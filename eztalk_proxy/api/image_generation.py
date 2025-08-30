from fastapi import APIRouter, HTTPException, Body
from ..models.image_generation_api_models import ImageGenerationRequest, ImageGenerationResponse
import httpx
import logging
import random
import re
from typing import Any, Dict, List
from pydantic import ValidationError

logger = logging.getLogger(__name__)
router = APIRouter()

def _fallback_response(reason: str) -> ImageGenerationResponse:
    # 统一的兜底结构，避免前端反序列化报缺少必填字段
    logger.error(f"[IMG] Fallback response due to error: {reason}")
    return ImageGenerationResponse(
        images=[],
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

def _normalize_response(data: Dict[str, Any]) -> ImageGenerationResponse:
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
            url_matches = re.findall(r"!\[.*?\]\((data:image/[^;]+;base64,[^\s\)\"]+|https?://[^\s\)]+)\)", content)
            for url in url_matches:
                images_list.append({"url": url})
            
            # Clean the image markdown from the text to get remaining text
            text_content = re.sub(r"!\[.*?\]\((data:image/[^;]+;base64,[^\s\)\"]+|https?://[^\s\)]+)\)", "", content).strip()
            if text_content:
                text_parts.append(text_content)
        elif isinstance(content, list): # Handle list content (e.g. for Gemini Vision)
            for part in content:
                if isinstance(part, dict) and part.get("type") == "text":
                    text_parts.append(part.get("text", ""))
                if isinstance(part, dict) and part.get("type") == "image_url":
                    image_url_data = part.get("image_url", {})
                    if isinstance(image_url_data, dict) and "url" in image_url_data:
                         images_list.append({"url": image_url_data["url"]})


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
        elif "image" in data: # Fallback for single image field
             images_list = _as_image_urls([data["image"]] if data["image"] else [])

    if not images_list and not text_parts:
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
        "text": " ".join(text_parts) if text_parts else None,
        "timings": timings_obj,
        "seed": seed_val
    }
    return ImageGenerationResponse(**normalized)

async def _proxy_and_normalize(request: ImageGenerationRequest) -> ImageGenerationResponse:
    url = request.apiAddress
    headers = {
        "Authorization": f"Bearer {request.apiKey}",
        "Content-Type": "application/json"
    }
    payload = {}

    if "gemini" in request.model.lower():
        if "/images/generations" in url:
            url = url.replace("/images/generations", "/chat/completions")

        content_parts = []
        if request.contents:
            text_prompt = ""
            for part in request.contents:
                if "text" in part:
                    text_prompt = part["text"]

            if text_prompt:
                # 轻量级指令，以用户输入为主
                enhanced_text_prompt = f"{text_prompt}\n\n---\n instruction: Edit the provided image based on the text description above. Output only the edited image."
                content_parts.append({"type": "text", "text": enhanced_text_prompt})

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
                "stream": False
            }
        else:
            # 轻量级指令，以用户输入为主
            enhanced_prompt = f"{request.prompt}\n\n---\nInstruction: Generate a high-quality image based on the description above. Output the image directly."
            payload = {
                "model": request.model,
                "messages": [{"role": "user", "content": enhanced_prompt}],
                "stream": False
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
 
    try:
        async with httpx.AsyncClient(timeout=httpx.Timeout(120.0), http2=True, follow_redirects=True) as client:
            resp = await client.post(url, headers=headers, json=payload)
    except httpx.RequestError as e:
        logger.error(f"[IMG] Upstream request error to {url}: {e}", exc_info=True)
        return _fallback_response(f"request_error: {e}")

    # 非 2xx：将上游响应体传回，便于前端/日志定位
    if resp.status_code < 200 or resp.status_code >= 300:
        text_preview = resp.text[:1000] if resp.text else "(empty)"
        logger.error(f"[IMG] Upstream non-2xx {resp.status_code}. Body preview: {text_preview}")
        # 返回兜底结构（HTTP 200 由路由层自动处理，因为我们返回模型对象）
        return _fallback_response(f"upstream_{resp.status_code}: {text_preview}")

    try:
        raw = resp.json()
        logger.info(f"[IMG] Upstream RAW response from provider: {raw}")
    except Exception as e:
        logger.error(f"[IMG] Upstream returned non-JSON body: {e}. Body preview: {resp.text[:500]}", exc_info=True)
        return _fallback_response(f"non_json_upstream: {e}")

    try:
        normalized = _normalize_response(raw)
        logger.info("[IMG] Image generation normalized successfully")
        return normalized
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