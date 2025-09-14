import logging
import uuid
from typing import List, Optional
from urllib.parse import urlparse

from fastapi import APIRouter, Depends, Request, HTTPException, File, UploadFile, Form
from fastapi.responses import StreamingResponse
import httpx
import orjson

from ..models.api_models import ChatRequestModel
from . import gemini, openai

logger = logging.getLogger("EzTalkProxy.Routers.Chat")
router = APIRouter()

def mask_api_key_for_log(api_key: Optional[str]) -> str:
    if not api_key:
        return "(empty)"
    head = api_key[:4]
    tail = api_key[-4:] if len(api_key) > 8 else "****"
    return f"{head}...{tail} (len={len(api_key)})"

def is_google_official_api(api_address: str) -> bool:
    """
    判断API地址是否为Google官方地址
    Google官方Gemini API地址通常包含：
    - generativelanguage.googleapis.com
    - aiplatform.googleapis.com
    - googleapis.com (通用Google API域名)
    """
    if not api_address:
        return False
    
    try:
        parsed_url = urlparse(api_address)
        domain = parsed_url.netloc.lower()
        
        # Google官方API域名列表
        google_domains = [
            'generativelanguage.googleapis.com',
            'aiplatform.googleapis.com',
            'googleapis.com',
            'ai.google.dev'  # Google AI Studio API
        ]
        
        # 检查是否为Google官方域名或其子域名
        for google_domain in google_domains:
            if domain == google_domain or domain.endswith('.' + google_domain):
                return True
                
        return False
    except Exception as e:
        logger.warning(f"Failed to parse API address '{api_address}': {e}")
        return False

async def get_http_client(request: Request) -> httpx.AsyncClient:
    client = getattr(request.app.state, "http_client", None)
    if client is None or (hasattr(client, 'is_closed') and client.is_closed):
        logger.error("HTTP client not available or closed in app.state.")
        raise HTTPException(status_code=503, detail="Service unavailable: HTTP client not initialized or closed.")
    return client

async def extract_chat_request_from_form(request: Request) -> tuple[str, List[UploadFile]]:
    """
    从 multipart/form-data 请求中提取聊天请求数据
    兼容各种客户端格式，包括缺少 name 属性的情况
    """
    try:
        form = await request.form()
        chat_request_json_str = None
        uploaded_files = []
        
        # 首先尝试标准方式获取 chat_request_json
        if "chat_request_json" in form:
            chat_request_json_str = form["chat_request_json"]
        
        # 如果标准方式失败，尝试从所有字符串值中查找有效的 JSON
        if not chat_request_json_str:
            for key, value in form.items():
                if isinstance(value, str):
                    try:
                        # 尝试解析为 JSON 并检查是否包含必要字段
                        potential_json = orjson.loads(value)
                        if isinstance(potential_json, dict) and "messages" in potential_json and "model" in potential_json:
                            chat_request_json_str = value
                            logger.info(f"Found chat_request_json in form field '{key}' (fallback method)")
                            break
                    except (orjson.JSONDecodeError, TypeError):
                        continue
        
        # 收集上传的文件
        for key, value in form.items():
            if hasattr(value, 'filename') and hasattr(value, 'file'):  # 这是一个文件
                uploaded_files.append(value)
        
        if not chat_request_json_str:
            raise HTTPException(status_code=400, detail="Missing 'chat_request_json' field in form data")
        
        return chat_request_json_str, uploaded_files
        
    except Exception as e:
        logger.error(f"Error extracting chat request from form: {e}", exc_info=True)
        raise HTTPException(status_code=400, detail=f"Failed to parse multipart form data: {e}")

def decide_chat_channel(chat_input: ChatRequestModel) -> tuple[str, str]:
    """
    决策文本聊天所用渠道：
    - 优先依据 provider 关键词
    - 次选依据 model 是否包含 'gemini'
    - 最后依据 apiAddress 是否为 Google 官方域名
    返回 (channel, reason)，channel ∈ {'gemini','openai'}
    reason ∈ {'provider','model','domain','fallback'}
    """
    provider_lower = (chat_input.provider or "").lower()
    gemini_keys = ["gemini", "google", "vertex", "aistudio", "google-gemini"]
    openai_keys = ["openai", "azure", "oai", "gpt", "openai-compatible", "openai_compatible"]

    if any(k in provider_lower for k in gemini_keys):
        return "gemini", "provider"
    if any(k in provider_lower for k in openai_keys):
        return "openai", "provider"

    model_lower = (chat_input.model or "").lower()
    if "gemini" in model_lower:
        return "gemini", "model"

    if is_google_official_api(chat_input.api_address or ""):
        return "gemini", "domain"

    return "openai", "fallback"

@router.post("/chat", response_class=StreamingResponse, summary="AI聊天完成代理", tags=["AI Proxy"])
async def chat_proxy_entrypoint(
    fastapi_request_obj: Request,
    chat_request_json_str: Optional[str] = Form(None, alias="chat_request_json"),
    http_client: httpx.AsyncClient = Depends(get_http_client),
    uploaded_documents: List[UploadFile] = File(default_factory=list)
):
    request_id = str(uuid.uuid4())
    log_prefix = f"RID-{request_id}"
    
    # 如果标准方式没有获取到 chat_request_json，使用兼容性方法
    if not chat_request_json_str:
        logger.info(f"{log_prefix}: Standard form parsing failed, trying compatibility method")
        chat_request_json_str, uploaded_documents = await extract_chat_request_from_form(fastapi_request_obj)
    
    logger.info(f"{log_prefix}: Received /chat request with {len(uploaded_documents)} documents.")

    try:
        chat_input_data = orjson.loads(chat_request_json_str)
        chat_input = ChatRequestModel(**chat_input_data)
        logger.info(f"{log_prefix}: Parsed ChatRequestModel for provider '{chat_input.provider}' and model '{chat_input.model}'.")
        try:
            masked = mask_api_key_for_log(getattr(chat_input, "api_key", None))
            logger.info(f"{log_prefix}: API key fingerprint: {masked}; apiAddress='{chat_input.api_address}'")
        except Exception as _e:
            logger.debug(f"{log_prefix}: Failed to log api key fingerprint safely: {_e}")
    except (orjson.JSONDecodeError, TypeError, ValueError) as e:
        logger.error(f"{log_prefix}: Failed to parse or validate chat request JSON: {e}", exc_info=True)
        raise HTTPException(status_code=400, detail=f"Invalid chat request data: {e}")

    # 使用新版分发：channel 优先（实现“文本模式 Gemini 可走聚合商链路”）
    channel, reason = decide_chat_channel_v2(chat_input)
    logger.info(f"{log_prefix}: Channel decided (v2) => {channel} (reason={reason}); "
                f"provider='{chat_input.provider}', model='{chat_input.model}', api='{chat_input.api_address}', channel_field='{getattr(chat_input, 'channel', None)}'.")

    if channel == "gemini":
        # 仅当确认为 Gemini 官方直连时才进入 gemini 处理器；不再强制覆盖为官方地址
        return await gemini.handle_gemini_request(
            gemini_chat_input=chat_input,
            uploaded_files=uploaded_documents,
            fastapi_request_obj=fastapi_request_obj,
            http_client=http_client,
            request_id=request_id,
        )
    else:
        # 其余（包括 Gemini + 聚合商）统一走 OpenAI 兼容分支（与图像模式行为保持一致）
        return await openai.handle_openai_compatible_request(
            chat_input=chat_input,
            uploaded_documents=uploaded_documents,
            fastapi_request_obj=fastapi_request_obj,
            http_client=http_client,
            request_id=request_id,
        )
def decide_chat_channel_v2(chat_input: ChatRequestModel) -> tuple[str, str]:
    """
    文本模式分发（channel 优先）：
    - 若 channel 明确为 OpenAI 兼容（含“openai”、“兼容”、“compatible”、“oai”、“azure”等），直接走 openai
    - 若 channel 明确为 Gemini 官方（含“gemini”、“google”、“aistudio”、“ai studio”、“官方”等）：
        * 若 apiAddress 是 Google 官方域名或未提供 → gemini
        * 否则（地址看起来是聚合商）→ openai（避免把聚合商强制改成直连 Google）
    - 如 channel 未明确：按 provider → model → 域名兜底
    返回 (channel, reason)，channel ∈ {'gemini','openai'}
    """
    channel_lower = (getattr(chat_input, "channel", None) or "").lower()
    provider_lower = (chat_input.provider or "").lower()
    model_lower = (chat_input.model or "").lower()
    api_addr = (chat_input.api_address or "")

    # 1) channel 明确优先
    if channel_lower:
        # OpenAI 兼容通道关键词
        openai_keys = ["openai", "兼容", "compatible", "oai", "azure"]
        if any(k in channel_lower for k in openai_keys):
            return "openai", "channel"

        # Gemini 官方通道关键词（按你的要求：只要 channel 表示 Gemini，就视为 Gemini 语义，不再因地址而回退到 OpenAI 兼容）
        gemini_channel_keys = ["gemini", "google", "aistudio", "ai studio", "官方"]
        if any(k in channel_lower for k in gemini_channel_keys):
            return "gemini", "channel"

    # 2) provider 推断
    # 常见聚合/代理关键词（尽量窄匹配，避免误杀）
    provider_is_aggregator = any(
        key in provider_lower
        for key in ["asb", "abs", "openrouter", "router", "done", "hub"]
    )
    if provider_is_aggregator:
        return "openai", "provider_agg"

    gemini_provider_keys = ["gemini", "google", "vertex", "aistudio", "google-gemini"]
    if any(k in provider_lower for k in gemini_provider_keys):
        # 若地址非 Google 官方，仍视为聚合商 → openai
        if api_addr and not is_google_official_api(api_addr):
            return "openai", "provider_non_google_address"
        return "gemini", "provider"

    # 3) model 推断
    if "gemini" in model_lower:
        # 非 Google 官方域名 → 当作聚合商，走 openai 兼容
        if api_addr and not is_google_official_api(api_addr):
            return "openai", "model_non_google"
        return "gemini", "model"

    # 4) 域名兜底
    if is_google_official_api(api_addr):
        return "gemini", "domain"

    return "openai", "fallback"