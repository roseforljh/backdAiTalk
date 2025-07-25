import logging
import uuid
from typing import List
from urllib.parse import urlparse

from fastapi import APIRouter, Depends, Request, HTTPException, File, UploadFile, Form
from fastapi.responses import StreamingResponse
import httpx
import orjson

from ..models.api_models import ChatRequestModel
from . import gemini, openai

logger = logging.getLogger("EzTalkProxy.Routers.Chat")
router = APIRouter()

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

@router.post("/chat", response_class=StreamingResponse, summary="AI聊天完成代理", tags=["AI Proxy"])
async def chat_proxy_entrypoint(
    fastapi_request_obj: Request,
    chat_request_json_str: str = Form(..., alias="chat_request_json"),
    http_client: httpx.AsyncClient = Depends(get_http_client),
    uploaded_documents: List[UploadFile] = File(default_factory=list)
):
    request_id = str(uuid.uuid4())
    log_prefix = f"RID-{request_id}"
    logger.info(f"{log_prefix}: Received /chat request with {len(uploaded_documents)} documents.")

    try:
        chat_input_data = orjson.loads(chat_request_json_str)
        chat_input = ChatRequestModel(**chat_input_data)
        logger.info(f"{log_prefix}: Parsed ChatRequestModel for provider '{chat_input.provider}' and model '{chat_input.model}'.")
    except (orjson.JSONDecodeError, TypeError, ValueError) as e:
        logger.error(f"{log_prefix}: Failed to parse or validate chat request JSON: {e}", exc_info=True)
        raise HTTPException(status_code=400, detail=f"Invalid chat request data: {e}")

    # 新的路由逻辑：根据API地址判断是否使用Gemini规则
    # 如果地址为Google官方，则走Gemini规则；否则全部走OpenAI兼容格式规则
    use_gemini_format = is_google_official_api(chat_input.api_address)
    
    if use_gemini_format:
        logger.info(f"{log_prefix}: API address '{chat_input.api_address}' is Google official, dispatching to Gemini handler for model {chat_input.model}.")
        return await gemini.handle_gemini_request(
            gemini_chat_input=chat_input,
            uploaded_files=uploaded_documents,
            fastapi_request_obj=fastapi_request_obj,
            http_client=http_client,
            request_id=request_id,
        )
    else:
        logger.info(f"{log_prefix}: API address '{chat_input.api_address}' is not Google official, dispatching to OpenAI compatible handler for model {chat_input.model}.")
        return await openai.handle_openai_compatible_request(
            chat_input=chat_input,
            uploaded_documents=uploaded_documents,
            fastapi_request_obj=fastapi_request_obj,
            http_client=http_client,
            request_id=request_id,
        )