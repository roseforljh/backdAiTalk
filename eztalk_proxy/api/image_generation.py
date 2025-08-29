from fastapi import APIRouter, HTTPException, Body
from ..models.image_generation_api_models import ImageGenerationRequest, ImageGenerationResponse
import httpx
import logging
import random
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
    # Common variants observed from different providers
    if "images" in data:
        images_list = _as_image_urls(data.get("images"))
    elif "data" in data:
        images_list = _as_image_urls(data.get("data"))
    elif "output" in data and isinstance(data["output"], dict):
        images_list = _as_image_urls(data["output"].get("images"))

    if not images_list:
        # Fallback: if a single field "image" exists
        single = data.get("image")
        converted = _as_image_urls([single] if single is not None else [])
        images_list = converted

    if not images_list:
        raise ValueError("Downstream API did not return any recognizable images field")

    timings_obj = {}
    # Try multiple common keys for inference time
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
        # Try nested or generate a pseudo seed
        for k in ["meta", "metadata"]:
            maybe = data.get(k, {})
            if isinstance(maybe, dict) and isinstance(maybe.get("seed"), int):
                seed_val = maybe["seed"]
                break
        if not isinstance(seed_val, int):
            seed_val = random.randint(1, 2**31 - 1)

    normalized = {
        "images": images_list,
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
    # 使用 pydantic v2 的导出，并进行必要的清洗
    payload = request.model_dump(exclude={"apiAddress", "apiKey"})
    # SiliconFlow 要求有效的分辨率字符串，避免占位符
    try:
        img_size = payload.get("image_size")
        if not isinstance(img_size, str) or not img_size.strip() or "<" in img_size:
            payload["image_size"] = "1024x1024"
    except Exception:
        payload["image_size"] = "1024x1024"
    # 清理 None 字段，避免发无效键
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