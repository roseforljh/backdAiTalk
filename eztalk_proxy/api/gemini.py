import os
import logging
import httpx
import orjson
import asyncio
import base64
import io
from typing import Optional, List

import docx
from fastapi import Request, UploadFile
from fastapi.responses import StreamingResponse

from ..models.api_models import (
    ChatRequestModel,
    AppStreamEventPy,
    PartsApiMessagePy,
    AbstractApiMessagePy,
    SimpleTextApiMessagePy,
    PyTextContentPart,
    PyInlineDataContentPart,
    IncomingApiContentPart,
    PyFileUriContentPart
)
from ..core.config import (
    GEMINI_ENABLE_GCS_UPLOAD,
    GCS_BUCKET_NAME,
    GCS_PROJECT_ID,
    API_TIMEOUT
)
from ..utils.helpers import (
    get_current_time_iso,
    orjson_dumps_bytes_wrapper,
    strip_potentially_harmful_html_and_normalize_newlines,
    extract_sse_lines,
    upload_to_gcs
)
from ..services.request_builder import prepare_gemini_rest_api_request
from ..services.stream_processor import (
    process_openai_like_sse_stream,
    handle_stream_error,
    handle_stream_cleanup,
    should_apply_custom_separator_logic
)
from ..services.web_search import perform_web_search, generate_search_context_message_content

logger = logging.getLogger("EzTalkProxy.Handlers.Gemini")

IMAGE_MIME_TYPES = ["image/png", "image/jpeg", "image/webp", "image/heic", "image/heif"]
DOCUMENT_MIME_TYPES = [
    "application/pdf",
    "application/x-javascript", "text/javascript",
    "application/x-python", "text/x-python",
    "text/plain",
    "text/html",
    "text/css",
    "text/md",
    "text/markdown",
    "text/csv",
    "text/xml",
    "text/rtf"
]
VIDEO_AUDIO_MIME_TYPES = [
    "video/mp4", "video/mpeg", "video/quicktime", "video/x-msvideo", "video/x-flv",
    "video/x-matroska", "video/webm", "video/x-ms-wmv", "video/3gpp", "video/x-m4v",
    "audio/wav", "audio/mpeg", "audio/aac", "audio/ogg", "audio/opus", "audio/flac"
]

async def sse_event_serializer_rest(event_data: AppStreamEventPy) -> bytes:
    return orjson_dumps_bytes_wrapper(event_data.model_dump(by_alias=True, exclude_none=True))

async def handle_gemini_request(
    gemini_chat_input: ChatRequestModel,
    uploaded_files: List[UploadFile],
    fastapi_request_obj: Request,
    http_client: httpx.AsyncClient,
    request_id: str,
):
    log_prefix = f"RID-{request_id}"
    active_messages_for_llm: List[AbstractApiMessagePy] = [msg.model_copy(deep=True) for msg in gemini_chat_input.messages]
    newly_created_multimodal_parts: List[IncomingApiContentPart] = []

    if gemini_chat_input.use_web_search:
        query = ""
        last_user_message_idx = -1
        for i in range(len(active_messages_for_llm) - 1, -1, -1):
            if active_messages_for_llm[i].role == 'user':
                last_user_message_idx = i
                break
        
        if last_user_message_idx != -1:
            last_user_message = active_messages_for_llm[last_user_message_idx]
            if isinstance(last_user_message, SimpleTextApiMessagePy):
                query = last_user_message.content
            elif isinstance(last_user_message, PartsApiMessagePy):
                query = " ".join([part.text for part in last_user_message.parts if isinstance(part, PyTextContentPart)])

        if query:
            search_results = await perform_web_search(query, request_id)
            if search_results:
                search_context_content = generate_search_context_message_content(query, search_results)
                context_part = PyTextContentPart(type="text_content", text=search_context_content)
                
                # Prepend the search context to the last user message
                if last_user_message_idx != -1:
                    user_msg = active_messages_for_llm[last_user_message_idx]
                    if isinstance(user_msg, PartsApiMessagePy):
                        user_msg.parts.insert(0, context_part)
                    elif isinstance(user_msg, SimpleTextApiMessagePy):
                        # Convert SimpleTextApiMessage to PartsApiMessage
                        original_text_part = PyTextContentPart(type="text_content", text=user_msg.content)
                        active_messages_for_llm[last_user_message_idx] = PartsApiMessagePy(
                            role="user", parts=[context_part, original_text_part]
                        )
                    logger.info(f"{log_prefix}: Injected web search context into the last user message for Gemini.")

    if uploaded_files:
        for uploaded_file in uploaded_files:
            mime_type = uploaded_file.content_type.lower() if uploaded_file.content_type else ""
            filename = uploaded_file.filename or "unknown"
            
            try:
                if mime_type in IMAGE_MIME_TYPES:
                    await uploaded_file.seek(0)
                    file_bytes = await uploaded_file.read()
                    base64_data = base64.b64encode(file_bytes).decode('utf-8')
                    newly_created_multimodal_parts.append(PyInlineDataContentPart(
                        type="inline_data_content", mimeType=mime_type, base64Data=base64_data
                    ))
                elif mime_type in DOCUMENT_MIME_TYPES:
                    logger.info(f"{log_prefix}: Processing document for Gemini: {filename} ({mime_type})")
                    await uploaded_file.seek(0)
                    file_bytes = await uploaded_file.read()
                    base64_data = base64.b64encode(file_bytes).decode('utf-8')
                    newly_created_multimodal_parts.append(PyInlineDataContentPart(
                        type="inline_data_content", mimeType=mime_type, base64Data=base64_data
                    ))
                elif mime_type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
                    logger.info(f"{log_prefix}: Extracting text from DOCX file for Gemini: {filename}")
                    await uploaded_file.seek(0)
                    file_bytes = await uploaded_file.read()
                    try:
                        doc_stream = io.BytesIO(file_bytes)
                        document = docx.Document(doc_stream)
                        full_text = "\n".join([para.text for para in document.paragraphs])
                        
                        extracted_text_content = f"\n\n--- START OF DOCUMENT: {filename} ---\n\n{full_text}\n\n--- END OF DOCUMENT: {filename} ---\n"
                        
                        newly_created_multimodal_parts.append(PyTextContentPart(
                            type="text_content", text=extracted_text_content
                        ))
                    except Exception as docx_e:
                        logger.error(f"{log_prefix}: Failed to extract text from DOCX file {filename}: {docx_e}", exc_info=True)

                elif mime_type in VIDEO_AUDIO_MIME_TYPES:
                    logger.info(f"{log_prefix}: Processing audio/video for Gemini: {filename} ({mime_type})")
                    await uploaded_file.seek(0)
                    file_bytes = await uploaded_file.read()
                    base64_data = base64.b64encode(file_bytes).decode('utf-8')
                    newly_created_multimodal_parts.append(PyInlineDataContentPart(
                        type="inline_data_content", mimeType=mime_type, base64Data=base64_data
                    ))
                else:
                    logger.warning(f"{log_prefix}: Skipping unsupported file type for Gemini: {filename} ({mime_type})")
            except Exception as e:
                logger.error(f"{log_prefix}: Error processing file {filename} for Gemini: {e}", exc_info=True)

    if newly_created_multimodal_parts:
        last_user_message_idx = -1
        for i in range(len(active_messages_for_llm) - 1, -1, -1):
            if active_messages_for_llm[i].role == "user":
                last_user_message_idx = i
                break
        
        if last_user_message_idx != -1:
            user_msg = active_messages_for_llm[last_user_message_idx]
            if isinstance(user_msg, PartsApiMessagePy):
                user_msg.parts.extend(newly_created_multimodal_parts)
            elif isinstance(user_msg, SimpleTextApiMessagePy):
                initial_text_part = [PyTextContentPart(type="text_content", text=user_msg.content)] if user_msg.content else []
                active_messages_for_llm[last_user_message_idx] = PartsApiMessagePy(
                    role="user", parts=initial_text_part + newly_created_multimodal_parts
                )
        else:
            active_messages_for_llm.append(PartsApiMessagePy(role="user", parts=newly_created_multimodal_parts))

    try:
        target_url, headers, json_payload = prepare_gemini_rest_api_request(
            chat_input=gemini_chat_input.model_copy(update={'messages': active_messages_for_llm}),
            request_id=request_id
        )
    except Exception as e_prepare:
        async def error_gen():
            yield await sse_event_serializer_rest(AppStreamEventPy(type="error", message=f"请求准备错误: {e_prepare}", timestamp=get_current_time_iso()))
            yield await sse_event_serializer_rest(AppStreamEventPy(type="finish", reason="request_error", timestamp=get_current_time_iso()))
        return StreamingResponse(error_gen(), media_type="text/event-stream")

    async def stream_generator():
        # This part is moved from the original position to be available for the whole function
        search_results = []
        if gemini_chat_input.use_web_search:
            # Step 1: Extract the original query from the last user message BEFORE any modification.
            original_query = ""
            last_user_message_for_query = next((msg for msg in reversed(active_messages_for_llm) if msg.role == 'user'), None)
            if last_user_message_for_query:
                if isinstance(last_user_message_for_query, SimpleTextApiMessagePy):
                    original_query = last_user_message_for_query.content
                elif isinstance(last_user_message_for_query, PartsApiMessagePy):
                    # Assume the user's typed text is the last text part.
                    # This is a safeguard against re-searching the injected context.
                    text_parts = [part.text for part in last_user_message_for_query.parts if isinstance(part, PyTextContentPart)]
                    if text_parts:
                        original_query = text_parts[-1]

            # Step 2: Perform web search if the original query is not empty.
            if original_query:
                yield await sse_event_serializer_rest(AppStreamEventPy(type="status_update", stage="Searching web..."))
                search_results = await perform_web_search(original_query, request_id)
                
                # Step 3: If search results are found, inject them into the message list.
                if search_results:
                    yield await sse_event_serializer_rest(AppStreamEventPy(type="web_search_results", results=search_results))
                    search_context_content = generate_search_context_message_content(original_query, search_results)
                    context_part = PyTextContentPart(type="text_content", text=search_context_content)
                    
                    # Find the last user message again to modify it.
                    last_user_message_idx = -1
                    for i in range(len(active_messages_for_llm) - 1, -1, -1):
                        if active_messages_for_llm[i].role == 'user':
                            last_user_message_idx = i
                            break
                    
                    if last_user_message_idx != -1:
                        user_msg_to_modify = active_messages_for_llm[last_user_message_idx]
                        if isinstance(user_msg_to_modify, PartsApiMessagePy):
                            user_msg_to_modify.parts.insert(0, context_part)
                        elif isinstance(user_msg_to_modify, SimpleTextApiMessagePy):
                            original_text_part = PyTextContentPart(type="text_content", text=user_msg_to_modify.content)
                            active_messages_for_llm[last_user_message_idx] = PartsApiMessagePy(
                                role="user", parts=[context_part, original_text_part]
                            )
                        logger.info(f"{log_prefix}: Injected web search context into the last user message for Gemini.")
                    yield await sse_event_serializer_rest(AppStreamEventPy(type="status_update", stage="Answering..."))

        processing_state = {}
        upstream_ok = False
        first_chunk_received = False
        try:
            async with http_client.stream("POST", target_url, headers=headers, json=json_payload, timeout=API_TIMEOUT) as response:
                upstream_ok = response.is_success
                if not upstream_ok:
                    error_body = await response.aread()
                    logger.error(f"{log_prefix}: Gemini upstream error {response.status_code}: {error_body.decode(errors='ignore')}")
                    response.raise_for_status()

                async for line in response.aiter_lines():
                    if not first_chunk_received:
                        first_chunk_received = True
                    
                    if line.startswith("data:"):
                        json_str = line[len("data:"):].strip()
                        try:
                            logger.debug(f"Received from Gemini: {json_str}")
                            sse_data = orjson.loads(json_str)
                            openai_like_sse = {"id": f"gemini-{request_id}", "choices": []}
                            
                            for candidate in sse_data.get("candidates", []):
                                content_parts = candidate.get("content", {}).get("parts", [])
                                is_thought_part = any("thought" in part for part in content_parts)

                                if is_thought_part:
                                    for part in content_parts:
                                        if "thought" in part and part.get("text"):
                                            reasoning_text = part["text"]
                                            yield await sse_event_serializer_rest(AppStreamEventPy(type="reasoning", text=reasoning_text))
                                else:
                                    delta = {}
                                    for part in content_parts:
                                        if "text" in part:
                                            delta["content"] = part["text"]
                                    
                                    if delta:
                                        choice = {
                                            "delta": delta,
                                            "finish_reason": candidate.get("finishReason")
                                        }
                                        openai_like_sse = {"id": f"gemini-{request_id}", "choices": [choice]}
                                        async for event in process_openai_like_sse_stream(openai_like_sse, processing_state, request_id):
                                            yield await sse_event_serializer_rest(AppStreamEventPy(**event))

                        except orjson.JSONDecodeError:
                            logger.warning(f"{log_prefix}: Skipping non-JSON line in Gemini stream: {line}")

        except Exception as e:
            logger.error(f"{log_prefix}: An error occurred during the Gemini stream: {e}", exc_info=True)
            async for error_event in handle_stream_error(e, request_id, upstream_ok, first_chunk_received):
                yield error_event
        finally:
            is_native_thinking = bool(gemini_chat_input.generation_config and gemini_chat_input.generation_config.thinking_config)
            use_custom_sep = should_apply_custom_separator_logic(gemini_chat_input, request_id, is_google_like_path=True, is_native_thinking_active=is_native_thinking)
            async for final_event in handle_stream_cleanup(processing_state, request_id, upstream_ok, use_custom_sep, gemini_chat_input.provider):
                yield final_event

    return StreamingResponse(stream_generator(), media_type="text/event-stream")