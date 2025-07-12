import logging
import uuid
from typing import List

from fastapi import APIRouter, Depends, Request, HTTPException, File, UploadFile, Form
from fastapi.responses import StreamingResponse
import httpx
import orjson

from ..models.api_models import ChatRequestModel
from . import gemini, openai

logger = logging.getLogger("EzTalkProxy.Routers.Chat")
router = APIRouter()

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

    if chat_input.provider.lower() == "google" and chat_input.model.lower().startswith("gemini"):
        logger.info(f"{log_prefix}: Dispatching to Gemini handler.")
        return await gemini.handle_gemini_request(
            gemini_chat_input=chat_input,
            uploaded_files=uploaded_documents,
            fastapi_request_obj=fastapi_request_obj,
            http_client=http_client,
            request_id=request_id,
        )
    else:
        logger.info(f"{log_prefix}: Dispatching to OpenAI compatible handler.")
        return await openai.handle_openai_compatible_request(
            chat_input=chat_input,
            uploaded_documents=uploaded_documents,
            fastapi_request_obj=fastapi_request_obj,
            http_client=http_client,
            request_id=request_id,
        )