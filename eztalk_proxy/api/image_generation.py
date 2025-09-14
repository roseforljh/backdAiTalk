from fastapi import APIRouter, HTTPException, Body
from ..models.image_generation_api_models import ImageGenerationRequest, ImageGenerationResponse
import httpx
import logging
import random
import re
import asyncio
from typing import Any, Dict, List
from pydantic import ValidationError

logger = logging.getLogger(__name__)
router = APIRouter()

def _is_google_official_api(api_address: str) -> bool:
    """Check if the API address is Google's official Gemini API"""
    if not api_address:
        return False
    
    google_domains = [
        "generativelanguage.googleapis.com",
        "ai.google.dev", 
        "googleapis.com"
    ]
    
    api_address_lower = api_address.lower()
    return any(domain in api_address_lower for domain in google_domains)

def _fallback_response(reason: str, user_text: str = None) -> ImageGenerationResponse:
    # 统一的兜底结构，避免前端反序列化报缺少必填字段
    logger.error(f"[IMG] Fallback response due to error: {reason}")
    return ImageGenerationResponse(
        images=[],
        text=user_text,
        timings={"inference": 0},
        seed=random.randint(1, 2**31 - 1)
    )

def _as_image_urls(ext_images: Any) -> List[Dict[str, str]]:
    urls: List[Dict[str, str]] = []
    if not isinstance(ext_images, list):
        return urls

    for item in ext_images:
        if isinstance(item, str) and item.startswith(('http://', 'https://', 'data:image/')):
            urls.append({"url": item})
        elif isinstance(item, dict):
            if "url" in item and isinstance(item["url"], str):
                urls.append({"url": item["url"]})
            elif "b64_json" in item and isinstance(item["b64_json"], str):
                urls.append({"url": f"data:image/png;base64,{item['b64_json']}"})
            # 兼容一些API将b64字符串直接放在image字段的情况
            elif "image" in item and isinstance(item["image"], str):
                urls.append({"url": f"data:image/png;base64,{item['image']}"})
            # 兼容一些API将b64字符串放在更深层嵌套的情况
            elif "image" in item and isinstance(item.get("image"), dict) and isinstance(item["image"].get("b64_json"), str):
                urls.append({"url": f"data:image/png;base64,{item['image']['b64_json']}"})
    return urls

def _normalize_response(data: Dict[str, Any], append_failure_hint: bool = False) -> ImageGenerationResponse:
    images_list: List[Dict[str, str]] = []
    text_parts: List[str] = []

    # Case 1: Provider wraps Gemini image response in an OpenAI chat completion format.
    if "choices" in data and isinstance(data.get("choices"), list) and data["choices"]:
        choice = data["choices"][0]
        if choice.get("finish_reason") == "content_filter":
            return ImageGenerationResponse(
                images=[],
                text="[CONTENT_FILTER]您的请求可能违反了相关的内容安全策略，已被拦截。请修改您的提示后重试。",
                timings={"inference": 0},
                seed=random.randint(1, 2**31 - 1)
            )

        message = choice.get("message", {})
        content = message.get("content", "")
        if isinstance(content, str):
            # Regex to find markdown image syntax with data URI or standard URL
            # ![...](...)
            url_matches = re.findall(r'!\[.*?\]\((data:image/[^;]+;base64,[^\s\)"]+|https?://[^\s\)]+)\)', content)
            for url in url_matches:
                images_list.append({"url": url})
            
            # Clean the image markdown from the text to get remaining text
            text_content = re.sub(r'!\[.*?\]\((data:image/[^;]+;base64,[^\s\)"]+|https?://[^\s\)]+)\)', "", content).strip()
            if text_content and text_content != '`':
                text_parts.append(text_content)
        elif isinstance(content, list):  # Handle list content (e.g. for Gemini Vision / OpenAI-compat multimodal)
            for part in content:
                if not isinstance(part, dict):
                    continue
                ptype = part.get("type")
                # 文本
                if ptype == "text" and isinstance(part.get("text"), str):
                    text_parts.append(part.get("text", ""))
                # OpenAI 兼容 image_url
                elif ptype == "image_url":
                    image_url_data = part.get("image_url", {})
                    if isinstance(image_url_data, dict) and "url" in image_url_data:
                        images_list.append({"url": image_url_data["url"]})
                # inline_data / inlineData（常见于Gemini返回或部分中转商）
                elif ptype in ("inline_data", "inlineData"):
                    data_obj = part.get("inline_data") or part.get("inlineData") or {}
                    if isinstance(data_obj, dict) and isinstance(data_obj.get("data"), str):
                        mime = data_obj.get("mime_type") or data_obj.get("mimeType") or "image/png"
                        images_list.append({"url": f"data:{mime};base64,{data_obj['data']}"})
                # image + b64_json/data
                elif ptype in ("image", "image_base64"):
                    img_obj = part.get("image", {})
                    if isinstance(img_obj, dict):
                        if isinstance(img_obj.get("b64_json"), str):
                            images_list.append({"url": f"data:image/png;base64,{img_obj['b64_json']}"})
                        elif isinstance(img_obj.get("data"), str):
                            data_val = img_obj["data"]
                            if data_val.startswith("data:image/"):
                                images_list.append({"url": data_val})
                            else:
                                images_list.append({"url": f"data:image/png;base64,{data_val}"})
                # 一些中转商可能直接把 data URI 放在字段 url 上
                elif isinstance(part.get("url"), str) and part["url"].startswith(("http://", "https://", "data:image/")):
                    images_list.append({"url": part["url"]})

        # Case 1.5: Handle OpenRouter's non-standard format for Gemini Image
        if not images_list and "images" in message and isinstance(message.get("images"), list):
            for img_item in message["images"]:
                if isinstance(img_item, dict) and img_item.get("type") == "image_url":
                    img_url_data = img_item.get("image_url", {})
                    if isinstance(img_url_data, dict) and "url" in img_url_data:
                        images_list.append({"url": img_url_data["url"]})

    # Case 2: Gemini's native format
    elif "candidates" in data and isinstance(data["candidates"], list):
        for candidate in data["candidates"]:
            if isinstance(candidate.get("content"), dict) and isinstance(candidate["content"].get("parts"), list):
                for part in candidate["content"]["parts"]:
                    if isinstance(part.get("inlineData"), dict) and isinstance(part["inlineData"].get("data"), str):
                        images_list.append({"url": f"data:image/png;base64,{part['inlineData']['data']}"})
                    if isinstance(part.get("text"), str):
                        text_parts.append(part["text"])

    # Case 3: Standard DALL-E/SD format (if no images found in other structures)
    if not images_list:
        if "images" in data:
            images_list = _as_image_urls(data.get("images"))
        elif "data" in data:
            images_list = _as_image_urls(data.get("data"))
        elif "output" in data and isinstance(data["output"], dict):
            images_list = _as_image_urls(data["output"].get("images"))
        elif "image" in data:  # Fallback for single image field
            images_list = _as_image_urls([data["image"]] if data["image"] else [])

    # Consolidate text and check for image generation failure patterns
    final_text = " ".join(text_parts).strip()
    if not images_list and final_text:
        # Common phrases indicating an intended but failed image generation
        failure_patterns = [
            "好的，这是", "好的，这是您要的", "这是您要的图片", "生成的图片如下", "这是我为您生成的",
            "here is the image", "here are the images", "i have generated", "voici l'image", "ecco l'immagine"
        ]
        # Use lowercasing for case-insensitive matching
        final_text_lower = final_text.lower()
        if append_failure_hint and any(p.lower() in final_text_lower for p in failure_patterns):
            # Append a user-friendly message about the potential failure
            failure_message = "\n\n(图片生成失败或被模型拒绝。请稍后重试或更换提示词。)"
            final_text += failure_message

    if not images_list and not final_text:
        raise ValueError("Downstream API did not return any recognizable images or text field")

    # Timings and Seed logic remains the same
    timings_obj = {}
    if isinstance(data.get("timings"), dict) and "inference" in data["timings"]:
        timings_obj = {"inference": int(data["timings"]["inference"])}
    else:
        inference_ms = None
        for key in ["inference", "inference_ms", "latency_ms", "runtime_ms"]:
            if isinstance(data.get(key), (int, float)):
                inference_ms = int(data[key])
                break
        timings_obj = {"inference": int(inference_ms or 0)}

    seed_val = data.get("seed")
    if not isinstance(seed_val, int):
        for k in ["meta", "metadata"]:
            maybe = data.get(k, {})
            if isinstance(maybe, dict) and isinstance(maybe.get("seed"), int):
                seed_val = maybe["seed"]
                break
        if not isinstance(seed_val, int):
            seed_val = random.randint(1, 2**31 - 1)

    normalized = {
        "images": images_list,
        "text": final_text if final_text else None,
        "timings": timings_obj,
        "seed": seed_val
    }
    return ImageGenerationResponse(**normalized)

async def _proxy_and_normalize(request: ImageGenerationRequest) -> ImageGenerationResponse:
    url = request.apiAddress
    headers = {
        "Authorization": f"Bearer {request.apiKey}",
        "Content-Type": "application/json",
        "User-Agent": "EzTalkProxy/1.9.9",
        "Accept": "application/json"
    }
    payload = {}

    model_lower = request.model.lower()
    is_gemini_image_model = "gemini" in model_lower and ("flash-image" in model_lower or "gemini-pro-vision" in model_lower)
    provider = request.provider or "openai compatible"
    provider_lower = provider.lower()
    
    # 调试信息：显示接收到的完整请求信息
    logger.info(f"[IMG DEBUG] Received ImageGenRequest:")
    logger.info(f"  - model: {request.model}")
    logger.info(f"  - provider/channel: {request.provider}")
    logger.info(f"  - apiAddress: {request.apiAddress}")
    logger.info(f"  - apiKey: {request.apiKey[:10]}..." if request.apiKey else "  - apiKey: None")

    # 根据用户选择的channel类型决定API格式
    # 支持中文和英文渠道名称映射
    channel_mapping = {
        "gemini": "gemini",
        "google": "gemini", 
        "Gemini": "gemini",
        "openai compatible": "openai_compatible",
        "OpenAI兼容": "openai_compatible",
        "openai": "openai_compatible"
    }
    normalized_channel = channel_mapping.get(provider, provider_lower)
    
    logger.info(f"[IMG DEBUG] Channel mapping - original: {provider} -> normalized: {normalized_channel}")
    
    # Gemini 渠道强制走原生（与前端/配置一致），不再依赖 host 判断
    if is_gemini_image_model and normalized_channel == "gemini":
        # Use Google's native Gemini API format
        logger.info(f"[IMG] Using Google native API format for {request.model} (provider: {provider})")
        
        # 同时提供 Authorization 与 x-goog-api-key，最大化兼容第三方聚合
        headers = {
            "Authorization": f"Bearer {request.apiKey}",
            "Content-Type": "application/json",
            "User-Agent": "EzTalkProxy/1.9.9",
            "Accept": "application/json",
            "x-goog-api-key": request.apiKey
        }
        
        content_parts = []
        if request.contents:  # Image editing mode with input images
            # Add text prompt first
            text_prompt = ""
            for part in request.contents:
                if "text" in part and part["text"]:
                    text_prompt = part["text"]
                    break
            
            if text_prompt:
                content_parts.append({"text": text_prompt})
            else:
                content_parts.append({"text": request.prompt or "Generate an image based on the provided image."})
            
            # Add inline images
            for part in request.contents:
                if "inline_data" in part:
                    inline_data = part["inline_data"]
                    mime_type = inline_data.get("mime_type", "image/jpeg")
                    b64_data = inline_data.get("data", "")
                    if b64_data:
                        content_parts.append({
                            "inline_data": {
                                "mime_type": mime_type,
                                "data": b64_data
                            }
                        })
        else:  # Pure text-to-image mode
            content_parts = [{"text": request.prompt or ""}]
        
        # Construct Google native API payload
        payload = {
            "contents": [{
                "parts": content_parts
            }]
        }
        
        # Add generation config if needed
        generation_config = {}
        if hasattr(request, 'image_size') and request.image_size:
            # Note: Google's API may have different parameter names
            generation_config["candidate_count"] = 1
            
        if generation_config:
            payload["generationConfig"] = generation_config
            
        # Construct the correct Google API URL
        model_name = request.model
        # Remove the /v1/images/generations suffix if present and replace with Google's format
        base_url = url
        if base_url.endswith('/v1/images/generations'):
            base_url = base_url[:-len('/v1/images/generations')]
        elif base_url.endswith('/v1/images/generations/'):
            base_url = base_url[:-len('/v1/images/generations/')]
        
        if not base_url.endswith('/'):
            base_url += '/'
        url = f"{base_url}v1beta/models/{model_name}:generateContent"
        # 追加 ?key= 以兼容官方与部分聚合实现
        delimiter = '&' if '?' in url else '?'
        if "key=" not in url:
            url = f"{url}{delimiter}key={request.apiKey}"
        
    elif is_gemini_image_model:
        # For OpenAI compatible format (even for Gemini models when provider is not "gemini")
        logger.info(f"[IMG] Using OpenAI-compatible format for Gemini model {request.model} (provider: {provider})")
        if "/images/generations" in url:
            url = url.replace("/images/generations", "/chat/completions")

        content_parts = []
        # 图像生成或编辑的核心逻辑
        if request.contents:  # 这是图像编辑模式
            text_prompt = ""
            # 首先找到文本部分
            for part in request.contents:
                if "text" in part and part["text"]:
                    text_prompt = part["text"]
                    break
            
            # OpenRouter文档要求文本部分在前
            if text_prompt:
                content_parts.append({"type": "text", "text": text_prompt})
            else:  # 如果没有文本，提供一个默认的
                content_parts.append({"type": "text", "text": "Edit the image."})
            
            # 然后添加图像部分
            for part in request.contents:
                if "inline_data" in part:
                    inline_data = part["inline_data"]
                    mime_type = inline_data.get("mime_type", "image/jpeg")
                    b64_data = inline_data.get("data", "")
                    if b64_data:
                        content_parts.append({
                            "type": "image_url",
                            "image_url": {"url": f"data:{mime_type};base64,{b64_data}"}
                        })
            
            payload = {
                "model": request.model,
                "messages": [{"role": "user", "content": content_parts}],
                "stream": False,
                "modalities": ["image"]
            }
        else:  # 这是纯文本图像生成模式
            # 使用多模态 parts 格式并显式声明 modalities，以触发中转商的图片输出
            content_parts = [{"type": "text", "text": request.prompt or ""}]
            payload = {
                "model": request.model,
                "messages": [{"role": "user", "content": content_parts}],
                "stream": False,
                "modalities": ["image"]
            }
    else:
        payload = request.model_dump(exclude={"apiAddress", "apiKey", "contents"})
        try:
            img_size = payload.get("image_size")
            if not isinstance(img_size, str) or not img_size.strip() or "<" in img_size:
                payload["image_size"] = "1024x1024"
        except Exception:
            payload["image_size"] = "1024x1024"

    payload = {k: v for k, v in payload.items() if v is not None}

    logger.info(f"[IMG] Proxying to upstream: {url} | model={payload.get('model')} | size={payload.get('image_size')} | batch={payload.get('batch_size')} | steps={payload.get('num_inference_steps')} | guidance={payload.get('guidance_scale')}")
    logger.debug(f"[IMG] Upstream payload: {payload}")
    if "x-goog-api-key" in headers and "Authorization" in headers:
        logger.info(f"[IMG] Request headers: Authorization & x-goog-api-key present, Content-Type: application/json")
    elif "x-goog-api-key" in headers:
        logger.info(f"[IMG] Request headers: x-goog-api-key: {request.apiKey[:10]}..., Content-Type: application/json")
    else:
        logger.info(f"[IMG] Request headers: Authorization: Bearer {request.apiKey[:10]}..., Content-Type: application/json")
 
    try:
        async with httpx.AsyncClient(timeout=httpx.Timeout(120.0), http2=True, follow_redirects=True) as client:
            resp = await client.post(url, headers=headers, json=payload)
    except httpx.RequestError as e:
        logger.error(f"[IMG] Upstream request error to {url}: {e}", exc_info=True)
        return _fallback_response(f"request_error: {e}", user_text="网络异常：无法连接上游服务，请稍后重试。")

    # 非 2xx：将上游响应体传回，便于前端/日志定位
    if resp.status_code < 200 or resp.status_code >= 300:
        text_preview = resp.text[:1000] if resp.text else "(empty)"
        logger.error(f"[IMG] Upstream non-2xx {resp.status_code}. Body preview: {text_preview}")
        logger.error(f"[IMG] Response headers: {dict(resp.headers)}")
        
        # 若为 Gemini 原生分支的 401，自动尝试不同认证/URL 组合回退，提高聚合商兼容性
        try:
            model_lower = request.model.lower()
            provider_lower = (request.provider or "").lower()
            is_gemini_image_model = "gemini" in model_lower and ("flash-image" in model_lower or "gemini-pro-vision" in model_lower)
            normalized_channel = {"gemini": "gemini", "google": "gemini", "openai compatible": "openai_compatible", "openai": "openai_compatible", "OpenAI兼容": "openai_compatible"}.get(request.provider or "openai compatible", provider_lower)
        except Exception:
            normalized_channel = provider_lower
            is_gemini_image_model = False
        
        if resp.status_code == 401 and is_gemini_image_model and normalized_channel == "gemini":
            logger.warning("[IMG] 401 from upstream in Gemini native branch, trying alternative auth strategies for compatibility...")
            # 生成三个尝试：
            # A) 仅 x-goog-api-key + URL ?key=    B) 仅 Authorization（移除 ?key）    C) x-api-key 头
            alt_attempts = []
            
            # 准备 URL 变体
            def remove_key_param(u: str) -> str:
                try:
                    # 移除 ?key= 或 &key=
                    return re.sub(r'([?&])key=[^&]*(&)?', lambda m: '?' if m.group(2) else '', u).rstrip('?')
                except Exception:
                    return u
            
            url_with_key = url if re.search(r'([?&])key=', url) else (f"{url}{'&' if '?' in url else '?'}key={request.apiKey}")
            url_without_key = remove_key_param(url)
            
            # A) 仅 x-goog-api-key
            a_headers = {k: v for k, v in headers.items() if k.lower() != "authorization"}
            a_headers["x-goog-api-key"] = request.apiKey
            alt_attempts.append(("A:x-goog-only", url_with_key, a_headers))
            
            # B) 仅 Authorization
            b_headers = {k: v for k, v in headers.items() if k.lower() != "x-goog-api-key"}
            b_headers["Authorization"] = f"Bearer {request.apiKey}"
            alt_attempts.append(("B:auth-only", url_without_key, b_headers))
            
            # C) x-api-key
            c_headers = {k: v for k, v in headers.items() if k.lower() not in ("x-goog-api-key",)}
            c_headers["x-api-key"] = request.apiKey
            alt_attempts.append(("C:x-api-key", url_without_key, c_headers))
            
            try:
                async with httpx.AsyncClient(timeout=httpx.Timeout(120.0), http2=True, follow_redirects=True) as client:
                    for idx, (label, alt_url, alt_headers) in enumerate(alt_attempts):
                        logger.info(f"[IMG] Retrying ({label}) -> {alt_url}")
                        alt_resp = await client.post(alt_url, headers=alt_headers, json=payload)
                        if 200 <= alt_resp.status_code < 300:
                            try:
                                raw = alt_resp.json()
                                raw_for_log = str(raw)
                                log_preview = raw_for_log[:1000] + ('...' if len(raw_for_log) > 1000 else '')
                                logger.info(f"[IMG] Alternative attempt succeeded ({label}). RAW preview: {log_preview}")
                                normalized = _normalize_response(raw, append_failure_hint=False)
                                if getattr(normalized, 'images', []) and len(normalized.images) > 0:
                                    logger.info(f"[IMG] Image generation normalized successfully after alternative attempt. Text: {normalized.text}, Images: {len(normalized.images)}")
                                    return normalized
                                # 若本次规范化仍无图片，只有在最后一次替代尝试后才附加失败提示
                                if idx == len(alt_attempts) - 1:
                                    normalized_final = _normalize_response(raw, append_failure_hint=True)
                                    logger.info(f"[IMG] Alternative attempts exhausted. Returning final normalized result. Text: {normalized_final.text}, Images: {len(normalized_final.images)}")
                                    return normalized_final
                                # 否则继续尝试下一种认证策略
                            except Exception as e_norm:
                                logger.error(f"[IMG] Alternative attempt succeeded but normalization failed: {e_norm}", exc_info=True)
                                return _fallback_response(f"normalize_error_after_alt: {e_norm}")
                        else:
                            logger.warning(f"[IMG] Alternative attempt ({label}) still non-2xx: {alt_resp.status_code}. Body: {alt_resp.text[:500] if alt_resp.text else '(empty)'}")
            except Exception as e_alt:
                logger.error(f"[IMG] Error during alternative auth retries: {e_alt}", exc_info=True)
        
        # 将上游错误提示透传给前端（例如地区限制）
        user_text = None
        try:
            err_json = resp.json()
            err_obj = err_json.get("error", {}) if isinstance(err_json, dict) else {}
            raw_msg = err_obj.get("message")
            raw_embedded = None
            meta = err_obj.get("metadata") if isinstance(err_obj, dict) else None
            if isinstance(meta, dict):
                raw_embedded = meta.get("raw")
            candidate_texts = [raw_msg, raw_embedded, text_preview]
            for ct in candidate_texts:
                if isinstance(ct, str) and ct:
                    if "User location is not supported" in ct or "FAILED_PRECONDITION" in ct:
                        user_text = "区域限制：上游拒绝该模型的调用。请更换可用地区节点/供应商，或选择其他模型。"
                        break
            if user_text is None and isinstance(raw_msg, str) and raw_msg:
                user_text = f"上游错误（{resp.status_code}）：{raw_msg}"
        except Exception:
            if "User location is not supported" in text_preview:
                user_text = "区域限制：上游拒绝该模型的调用。请更换可用地区节点/供应商，或选择其他模型。"
            else:
                user_text = f"上游错误（{resp.status_code}）：{text_preview}"
        # 返回兜底结构（HTTP 200 由路由层自动处理，因为我们返回模型对象）
        return _fallback_response(f"upstream_{resp.status_code}: {text_preview}", user_text=user_text)

    try:
        raw = resp.json()
        # 日志记录截断的响应，避免日志过长
        raw_for_log = str(raw)
        log_preview = raw_for_log[:1000] + ('...' if len(raw_for_log) > 1000 else '')
        logger.info(f"[IMG] Upstream RAW response from provider (preview): {log_preview}")
    except Exception as e:
        logger.error(f"[IMG] Upstream returned non-JSON body: {e}. Body preview: {resp.text[:500]}", exc_info=True)
        return _fallback_response(f"non_json_upstream: {e}", user_text=f"上游返回非JSON响应：{str(e)}")

    try:
        normalized = _normalize_response(raw, append_failure_hint=False)
        if getattr(normalized, 'images', []) and len(normalized.images) > 0:
            logger.info(f"[IMG] Image generation normalized successfully. Text: {normalized.text}, Images: {len(normalized.images)}")
            return normalized
        # 轻量级重试：首次规范化无图片、且非内容拦截时，重试一次
        if (normalized.text and not str(normalized.text).startswith("[CONTENT_FILTER]")):
            try:
                logger.info("[IMG] No images found after first normalization. Performing a lightweight retry once...")
                async with httpx.AsyncClient(timeout=httpx.Timeout(120.0), http2=True, follow_redirects=True) as client:
                    resp2 = await client.post(url, headers=headers, json=payload)
                if 200 <= resp2.status_code < 300:
                    try:
                        raw2 = resp2.json()
                        raw2_for_log = str(raw2)
                        log2_preview = raw2_for_log[:1000] + ('...' if len(raw2_for_log) > 1000 else '')
                        logger.info(f"[IMG] Upstream RAW response from provider (retry preview): {log2_preview}")
                        normalized2 = _normalize_response(raw2, append_failure_hint=True)
                        logger.info(f"[IMG] After retry, Text: {normalized2.text}, Images: {len(normalized2.images)}")
                        return normalized2
                    except Exception as e2:
                        logger.error(f"[IMG] Retry returned JSON but normalization failed: {e2}", exc_info=True)
                        # 回退到对第一次响应追加失败提示
                        normalized_final = _normalize_response(raw, append_failure_hint=True)
                        return normalized_final
                else:
                    logger.warning(f"[IMG] Retry non-2xx {resp2.status_code}. Body preview: {resp2.text[:500] if resp2.text else '(empty)'}")
                    normalized_final = _normalize_response(raw, append_failure_hint=True)
                    return normalized_final
            except Exception as er2:
                logger.error(f"[IMG] Lightweight retry error: {er2}", exc_info=True)
                normalized_final = _normalize_response(raw, append_failure_hint=True)
                return normalized_final
        # 若没有文本可提示或为内容拦截，直接返回追加提示后的结果（追加逻辑内部会自动判断是否需要提示）
        normalized_final = _normalize_response(raw, append_failure_hint=True)
        logger.info(f"[IMG] Returning final normalized result after no-image condition. Text: {normalized_final.text}, Images: {len(normalized_final.images)}")
        return normalized_final
    except Exception as e:
        logger.error(f"[IMG] Failed to normalize upstream response: {e}. Raw keys: {list(raw) if isinstance(raw, dict) else type(raw)}", exc_info=True)
        # 返回兜底结构，避免前端解析失败
        return _fallback_response(f"normalize_error: {e}")

# Support both with and without '/chat' prefix to be backward compatible
@router.post("/v1/images/generations", response_model=ImageGenerationResponse)
async def create_image_generation_v1(payload: Dict[str, Any] = Body(...)):
    try:
        req = ImageGenerationRequest(**payload)
    except ValidationError as e:
        raise HTTPException(status_code=400, detail={"message": "Invalid image generation request", "errors": e.errors()})
    return await _proxy_and_normalize(req)

@router.post("/chat/v1/images/generations", response_model=ImageGenerationResponse)
async def create_image_generation_chat_v1(payload: Dict[str, Any] = Body(...)):
    try:
        req = ImageGenerationRequest(**payload)
    except ValidationError as e:
        raise HTTPException(status_code=400, detail={"message": "Invalid image generation request", "errors": e.errors()})
    return await _proxy_and_normalize(req)