import os
import logging
import httpx
import orjson
import asyncio
import base64
import shutil
import uuid
from typing import Optional, Dict, Any, AsyncGenerator, List, Union

from fastapi import APIRouter, Depends, Request, HTTPException, File, UploadFile, Form
from fastapi.responses import StreamingResponse

from eztalk_proxy.models import (
    ChatRequestModel,
    SimpleTextApiMessagePy,
    PartsApiMessagePy,
    AppStreamEventPy
)
from eztalk_proxy.multimodal_models import (
    PyTextContentPart
)
from eztalk_proxy.config import (
    COMMON_HEADERS,
    TEMP_UPLOAD_DIR,
    MAX_DOCUMENT_UPLOAD_SIZE_MB,
    API_TIMEOUT
)
from eztalk_proxy.utils import (
    get_current_time_iso,
    orjson_dumps_bytes_wrapper,
    extract_text_from_uploaded_document,
    extract_sse_lines
)
from eztalk_proxy.api_helpers import prepare_openai_request
from eztalk_proxy.routers import multimodal_chat as multimodal_router
from eztalk_proxy.stream_processors import (
    process_openai_like_sse_stream,
    handle_stream_error,
    handle_stream_cleanup,
    should_apply_custom_separator_logic
)
from eztalk_proxy.web_search import perform_web_search, generate_search_context_message_content

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
    logger.info(f"{log_prefix}: Received /chat request.")

    try:
        chat_input_data = orjson.loads(chat_request_json_str)
        chat_input = ChatRequestModel(**chat_input_data)
        logger.info(f"{log_prefix}: Parsed ChatRequestModel successfully.")
    except orjson.JSONDecodeError as e_json:
        logger.error(f"{log_prefix}: Failed to parse chat_request_json_str: {e_json}. Data: {chat_request_json_str[:500]}")
        raise HTTPException(status_code=400, detail=f"Invalid JSON format in 'chat_request_json' field.")
    except Exception as e_pydantic:
        logger.error(f"{log_prefix}: Failed to validate ChatRequestModel from JSON: {e_pydantic}. Data: {chat_request_json_str[:500]}", exc_info=True)
        raise HTTPException(status_code=422, detail=f"Invalid data structure for chat request: {e_pydantic}")

    logger.info(
        f"{log_prefix}: Provider='{chat_input.provider}', "
        f"Model='{chat_input.model}', WebSearch={chat_input.use_web_search}, "
        f"Uploaded Items: {len(uploaded_documents)}"
    )

    processed_document_text_parts: List[str] = []
    temp_files_created_for_text_extraction: List[str] = []
    
    files_for_gemini_multimodal_direct_pass: List[UploadFile] = []
    base64_encoded_parts_for_openai: List[Dict[str, Any]] = []
    
    SUPPORTED_IMAGE_MIME_TYPES_FOR_OPENAI = ["image/jpeg", "image/png", "image/gif", "image/webp"]

    if uploaded_documents:
        logger.info(f"{log_prefix}: Processing {len(uploaded_documents)} uploaded items.")
        if not os.path.exists(TEMP_UPLOAD_DIR):
            try:
                os.makedirs(TEMP_UPLOAD_DIR)
                logger.info(f"{log_prefix}: Created temporary upload directory: {TEMP_UPLOAD_DIR}")
            except OSError as e_mkdir:
                logger.error(f"{log_prefix}: Failed to create temporary upload directory {TEMP_UPLOAD_DIR}: {e_mkdir}", exc_info=True)

        for doc_file in uploaded_documents:
            if not doc_file.filename:
                logger.warning(f"{log_prefix}: Skipping uploaded file with no filename.")
                try: await doc_file.close()
                except Exception: pass
                continue
            
            actual_size = -1
            try:
                if hasattr(doc_file, 'size') and doc_file.size is not None:
                     actual_size = doc_file.size
                elif hasattr(doc_file.file, 'tell') and hasattr(doc_file.file, 'seek'):
                    current_pos = doc_file.file.tell()
                    doc_file.file.seek(0, os.SEEK_END)
                    actual_size = doc_file.file.tell()
                    doc_file.file.seek(current_pos)
                
                if actual_size != -1 and actual_size > MAX_DOCUMENT_UPLOAD_SIZE_MB * 1024 * 1024:
                    logger.warning(f"{log_prefix}: Document '{doc_file.filename}' (size: {actual_size} B) exceeds max size ({MAX_DOCUMENT_UPLOAD_SIZE_MB} MB). Skipping.")
                    try: await doc_file.close()
                    except Exception: pass
                    continue
            except Exception as e_size:
                logger.warning(f"{log_prefix}: Could not reliably determine size of '{doc_file.filename}'. Error: {e_size}. Proceeding.")

            files_for_gemini_multimodal_direct_pass.append(doc_file)

            is_openai_compatible_vision_path = not (chat_input.provider.lower() == "google" and chat_input.model.lower().startswith("gemini"))
            should_attempt_text_extraction = True

            if is_openai_compatible_vision_path and doc_file.content_type and doc_file.content_type.lower() in SUPPORTED_IMAGE_MIME_TYPES_FOR_OPENAI:
                try:
                    logger.info(f"{log_prefix}: Processing image '{doc_file.filename}' for OpenAI Vision. MIME: {doc_file.content_type}")
                    await doc_file.seek(0)
                    image_bytes = await doc_file.read()
                    base64_encoded_data = base64.b64encode(image_bytes).decode('utf-8')
                    data_uri = f"data:{doc_file.content_type};base64,{base64_encoded_data}"
                    base64_encoded_parts_for_openai.append({"type": "image_url", "image_url": {"url": data_uri}})
                    await doc_file.seek(0)
                    logger.info(f"{log_prefix}: Successfully encoded '{doc_file.filename}' for OpenAI Vision.")
                    should_attempt_text_extraction = False
                except Exception as e_img_proc:
                    logger.error(f"{log_prefix}: Error processing image '{doc_file.filename}' for OpenAI Vision: {e_img_proc}", exc_info=True)
            
            if not should_attempt_text_extraction:
                continue

            _, file_extension = os.path.splitext(doc_file.filename)
            if not file_extension: file_extension = ".tmp"
            safe_original_filename_part = "".join(c if c.isalnum() or c in ['.', '_', '-'] else '_' for c in doc_file.filename.rsplit('.', 1)[0])[:50]
            temp_filename = f"{request_id}_{safe_original_filename_part}_{uuid.uuid4().hex[:8]}{file_extension}"
            temp_file_path = os.path.join(TEMP_UPLOAD_DIR, temp_filename)
            
            try:
                with open(temp_file_path, "wb") as buffer:
                    await doc_file.seek(0)
                    shutil.copyfileobj(doc_file.file, buffer)
                temp_files_created_for_text_extraction.append(temp_file_path)
                await doc_file.seek(0)

                document_text = await extract_text_from_uploaded_document(
                    temp_file_path,
                    doc_file.content_type,
                    doc_file.filename
                )
                if document_text:
                    doc_text_part = f"\n\n--- 来自文档: {doc_file.filename} ---\n{document_text}\n--- 文档结束: {doc_file.filename} ---\n"
                    processed_document_text_parts.append(doc_text_part)
                    logger.info(f"{log_prefix}: Successfully processed text from '{doc_file.filename}'. Length: {len(document_text)}")
                else:
                    logger.warning(f"{log_prefix}: Failed to extract text or no text found in '{doc_file.filename}'.")
            except Exception as e_doc_proc:
                logger.error(f"{log_prefix}: Error processing copy of document '{doc_file.filename}' for text extraction: {e_doc_proc}", exc_info=True)
    
    combined_document_text_for_prompt = "".join(processed_document_text_parts).strip() if processed_document_text_parts else None

    if chat_input.provider.lower() == "google" and chat_input.model.lower().startswith("gemini"):
        logger.info(f"{log_prefix}: Provider is 'google' and model '{chat_input.model}' is Gemini. Dispatching to Gemini REST API multimodal handler.")
        return StreamingResponse(
            multimodal_router.generate_gemini_rest_api_events_with_docs(
                gemini_chat_input=chat_input,
                uploaded_files_for_gemini=files_for_gemini_multimodal_direct_pass,
                additional_extracted_text=combined_document_text_for_prompt,
                fastapi_request_obj=fastapi_request_obj,
                http_client=http_client,
                request_id=request_id,
                temp_files_to_delete_after_stream=temp_files_created_for_text_extraction
            ),
            media_type="text/event-stream",
            headers=COMMON_HEADERS
        )
    else:
        logger.info(f"{log_prefix}: Model '{chat_input.model}' with provider '{chat_input.provider}' will be handled by non-Gemini-REST (OpenAI compatible) path.")
        
        messages_for_upstream: List[Dict[str, Any]] = []
        user_query_for_search = ""
        original_user_text_found = False

        for i, msg_abstract in enumerate(chat_input.messages):
            msg_dict: Dict[str, Any] = {"role": msg_abstract.role}
            is_last_user_message = (i == len(chat_input.messages) - 1 and msg_abstract.role == "user")
            current_user_text_content = ""

            if isinstance(msg_abstract, SimpleTextApiMessagePy):
                current_user_text_content = msg_abstract.content or ""
                msg_dict["content"] = current_user_text_content
                if msg_abstract.role == "user":
                    original_user_text_found = True
                    user_query_for_search = current_user_text_content.strip()
                
                if hasattr(msg_abstract, 'tool_calls') and msg_abstract.tool_calls:
                    msg_dict["tool_calls"] = [tc.model_dump(exclude_none=True) for tc in msg_abstract.tool_calls]
                if msg_abstract.role == "tool":
                    if hasattr(msg_abstract, 'tool_call_id') and msg_abstract.tool_call_id:
                        msg_dict["tool_call_id"] = msg_abstract.tool_call_id
                    if msg_abstract.name: msg_dict["name"] = msg_abstract.name

            elif isinstance(msg_abstract, PartsApiMessagePy):
                text_from_parts = " ".join([part.text for part in msg_abstract.parts if isinstance(part, PyTextContentPart) and part.text])
                current_user_text_content = text_from_parts.strip()
                msg_dict["content"] = current_user_text_content
                if msg_abstract.role == "user" and current_user_text_content:
                    original_user_text_found = True
                    user_query_for_search = current_user_text_content
            
            if is_last_user_message:
                final_content_parts_for_current_message: List[Dict[str, Any]] = []
                
                if current_user_text_content.strip():
                    final_content_parts_for_current_message.append({"type": "text", "text": current_user_text_content.strip()})
                
                if combined_document_text_for_prompt:
                    if final_content_parts_for_current_message and final_content_parts_for_current_message[0]["type"] == "text":
                        final_content_parts_for_current_message[0]["text"] += "\n" + combined_document_text_for_prompt
                    elif not final_content_parts_for_current_message:
                        final_content_parts_for_current_message.insert(0, {"type": "text", "text": combined_document_text_for_prompt})
                    else:
                         final_content_parts_for_current_message.append({"type": "text", "text": combined_document_text_for_prompt})
                    user_query_for_search = (user_query_for_search + "\n" + combined_document_text_for_prompt).strip()

                if base64_encoded_parts_for_openai:
                    if not final_content_parts_for_current_message or not any(p.get("type") == "text" and p.get("text","").strip() for p in final_content_parts_for_current_message):
                         final_content_parts_for_current_message.insert(0, {"type": "text", "text": "请描述或分析以下图片和/或文档内容："})
                    final_content_parts_for_current_message.extend(base64_encoded_parts_for_openai)
                
                if final_content_parts_for_current_message:
                    if len(final_content_parts_for_current_message) == 1 and final_content_parts_for_current_message[0]["type"] == "text":
                        msg_dict["content"] = final_content_parts_for_current_message[0]["text"]
                    else:
                        msg_dict["content"] = final_content_parts_for_current_message
                elif "content" not in msg_dict :
                     msg_dict["content"] = ""
            
            messages_for_upstream.append(msg_dict)
        
        if not original_user_text_found and (combined_document_text_for_prompt or base64_encoded_parts_for_openai):
            new_user_content_parts: List[Dict[str, Any]] = []
            default_prompt_text = "请基于以下文档内容和/或图片进行处理或回答："
            
            current_text_for_new_message = default_prompt_text
            if combined_document_text_for_prompt:
                current_text_for_new_message += "\n" + combined_document_text_for_prompt
            
            new_user_content_parts.append({"type": "text", "text": current_text_for_new_message.strip()})
            user_query_for_search = current_text_for_new_message.strip()

            if base64_encoded_parts_for_openai:
                new_user_content_parts.extend(base64_encoded_parts_for_openai)
            
            final_content_for_new_user_message: Union[str, List[Dict[str, Any]]]
            if len(new_user_content_parts) == 1 and new_user_content_parts[0]["type"] == "text":
                final_content_for_new_user_message = new_user_content_parts[0]["text"]
            else:
                final_content_for_new_user_message = new_user_content_parts
                
            messages_for_upstream.append({"role": "user", "content": final_content_for_new_user_message})

        if not messages_for_upstream or not any(m.get("role") != "system" for m in messages_for_upstream):
            has_valid_user_content_in_messages = any(
                (isinstance(m.get("content"), str) and m.get("content","").strip()) or
                (isinstance(m.get("content"), list) and any(p.get("type") == "text" and p.get("text","").strip() for p in m.get("content", []))) or
                (isinstance(m.get("content"), list) and any(p.get("type") == "image_url" for p in m.get("content", [])))
                for m in messages_for_upstream if m.get("role") == "user"
            )
            if not has_valid_user_content_in_messages and not any(m.get("role") == "system" and m.get("content","").strip() for m in messages_for_upstream):
                 logger.warning(f"{log_prefix}: No processable non-system messages or valid user content for non-Gemini-REST path.")
                 async def no_msg_gen_err():
                     yield orjson_dumps_bytes_wrapper(AppStreamEventPy(type="error", message="No processable messages for this model.", timestamp=get_current_time_iso()).model_dump(by_alias=True, exclude_none=True))
                     yield orjson_dumps_bytes_wrapper(AppStreamEventPy(type="finish", reason="bad_request", timestamp=get_current_time_iso()).model_dump(by_alias=True, exclude_none=True))
                     for temp_file in temp_files_created_for_text_extraction:
                         if os.path.exists(temp_file):
                             try: os.remove(temp_file)
                             except Exception as e_del_err: logger.error(f"{log_prefix}: Error deleting temp file {temp_file} in no_msg_gen_err: {e_del_err}")
                 return StreamingResponse(no_msg_gen_err(), media_type="text/event-stream", headers=COMMON_HEADERS)
        
        return StreamingResponse(
            generate_non_gemini_events(
                request_data=chat_input,
                processed_upstream_messages=messages_for_upstream,
                user_query_for_search=user_query_for_search,
                http_client=http_client,
                fastapi_request_obj=fastapi_request_obj,
                request_id=request_id,
                temp_files_to_delete_after_stream=temp_files_created_for_text_extraction
            ),
            media_type="text/event-stream",
            headers=COMMON_HEADERS
        )

async def generate_non_gemini_events(
    request_data: ChatRequestModel,
    processed_upstream_messages: List[Dict[str, Any]],
    user_query_for_search: str,
    http_client: httpx.AsyncClient,
    fastapi_request_obj: Request,
    request_id: str,
    temp_files_to_delete_after_stream: List[str]
) -> AsyncGenerator[bytes, None]:
    log_prefix = f"RID-{request_id}"
    final_messages_for_llm = list(processed_upstream_messages)
    search_results_generated_this_time = False

    if request_data.use_web_search and user_query_for_search:
        yield orjson_dumps_bytes_wrapper(AppStreamEventPy(type="status_update", stage="web_search_started", timestamp=get_current_time_iso()).model_dump(by_alias=True, exclude_none=True))
        
        search_results_list = await perform_web_search(user_query_for_search, request_id)
        
        if search_results_list:
            search_context_content = generate_search_context_message_content(user_query_for_search, search_results_list)
            new_system_message_dict = {"role": "system", "content": search_context_content}
            
            last_user_idx = -1
            for i, msg in reversed(list(enumerate(final_messages_for_llm))):
                if msg.get("role") == "user":
                    last_user_idx = i
                    break
            if last_user_idx != -1:
                final_messages_for_llm.insert(last_user_idx, new_system_message_dict)
            else:
                final_messages_for_llm.insert(0, new_system_message_dict)
            
            search_results_generated_this_time = True
            logger.info(f"{log_prefix}: (Non-Gemini-REST) Web search context injected.")
            yield orjson_dumps_bytes_wrapper(AppStreamEventPy(type="status_update", stage="web_search_complete_with_results", query=user_query_for_search, timestamp=get_current_time_iso()).model_dump(by_alias=True, exclude_none=True))
            yield orjson_dumps_bytes_wrapper(AppStreamEventPy(type="web_search_results", results=search_results_list, timestamp=get_current_time_iso()).model_dump(by_alias=True, exclude_none=True))
        else:
            logger.info(f"{log_prefix}: (Non-Gemini-REST) Web search yielded no results for query '{user_query_for_search[:100]}'.")
            yield orjson_dumps_bytes_wrapper(AppStreamEventPy(type="status_update", stage="web_search_complete_no_results", query=user_query_for_search, timestamp=get_current_time_iso()).model_dump(by_alias=True, exclude_none=True))
        
    try:
        current_api_url, current_api_headers, current_api_payload = prepare_openai_request(
            request_data=request_data,
            processed_messages=final_messages_for_llm,
            request_id=request_id
        )
    except Exception as e_prepare:
        logger.error(f"{log_prefix}: Error preparing non-Gemini-REST request: {e_prepare}", exc_info=True)
        yield orjson_dumps_bytes_wrapper(AppStreamEventPy(type="error", message=f"Request preparation error: {e_prepare}", timestamp=get_current_time_iso()).model_dump(by_alias=True, exclude_none=True))
        yield orjson_dumps_bytes_wrapper(AppStreamEventPy(type="finish", reason="request_error", timestamp=get_current_time_iso()).model_dump(by_alias=True, exclude_none=True))
        for temp_file in temp_files_to_delete_after_stream:
            if os.path.exists(temp_file):
                try: os.remove(temp_file)
                except Exception as e_del_prep: logger.error(f"{log_prefix}: Error deleting temp file {temp_file} after prep error: {e_del_prep}")
        return

    buffer = bytearray()
    upstream_ok_flag = False
    first_chunk_llm_received = False
    stream_proc_state: Dict[str, Any] = {
        "accumulated_openai_content": "", "accumulated_openai_reasoning": "",
        "openai_had_any_reasoning": False, "openai_had_any_content_or_tool_call": False,
        "openai_reasoning_finish_event_sent": False,
        "final_finish_event_sent_by_llm_reason": False,
        "final_finish_event_sent_flag_for_cleanup": False
    }
    use_old_custom_separator_branch_flag = should_apply_custom_separator_logic(
        request_data, request_id, False, False
    )

    try:
        content_previews = []
        for m_idx, m_val in enumerate(current_api_payload.get('messages',[])):
            content_item = m_val.get('content','')
            if isinstance(content_item, str):
                preview = content_item[:70] + ('...' if len(content_item)>70 else '')
            elif isinstance(content_item, list):
                part_previews = []
                for p_idx, p_val in enumerate(content_item):
                    if p_val.get("type") == "text":
                        part_text = p_val.get("text","")[:30] + "..."
                        part_previews.append(f"TextPart[{p_idx}]:'{part_text}'")
                    elif p_val.get("type") == "image_url":
                        part_previews.append(f"ImagePart[{p_idx}]")
                preview = f"MultiPart: [{', '.join(part_previews)}]"
            else:
                preview = "UnknownContentFormat"
            content_previews.append(f"Msg[{m_idx}]: {preview}")

        async with http_client.stream(
            "POST", current_api_url,
            headers=current_api_headers,
            json=current_api_payload,
            timeout=API_TIMEOUT
        ) as response:
            logger.info(f"{log_prefix}: (Non-Gemini-REST) Upstream LLM response status: {response.status_code}")
            if not (200 <= response.status_code < 300):
                err_body_bytes = await response.aread()
                err_text = err_body_bytes.decode("utf-8", errors="replace")
                logger.error(f"{log_prefix}: (Non-Gemini-REST) Upstream LLM error {response.status_code}: {err_text[:1000]}")
                yield orjson_dumps_bytes_wrapper(AppStreamEventPy(type="error", message=f"LLM API Error: {err_text[:200]}", upstream_status=response.status_code, timestamp=get_current_time_iso()).model_dump(by_alias=True, exclude_none=True))
                upstream_ok_flag = False
                return

            upstream_ok_flag = True
            async for raw_chunk_bytes in response.aiter_raw():
                if await fastapi_request_obj.is_disconnected():
                    logger.info(f"{log_prefix}: (Non-Gemini-REST) Client disconnected.")
                    break
                
                if not first_chunk_llm_received:
                    if request_data.use_web_search and user_query_for_search:
                        stage_after_search = "web_analysis_complete" if search_results_generated_this_time else "web_analysis_skipped_no_results"
                        yield orjson_dumps_bytes_wrapper(AppStreamEventPy(type="status_update", stage=stage_after_search, timestamp=get_current_time_iso()).model_dump(by_alias=True, exclude_none=True))
                    first_chunk_llm_received = True
                
                buffer.extend(raw_chunk_bytes)
                sse_lines, buffer = extract_sse_lines(buffer)

                for sse_line_bytes in sse_lines:
                    if not sse_line_bytes.strip(): continue
                    sse_data_bytes = b""
                    if sse_line_bytes.startswith(b"data: "):
                        sse_data_bytes = sse_line_bytes[len(b"data: "):].strip()
                    if not sse_data_bytes: continue
                    
                    if sse_data_bytes == b"[DONE]":
                        logger.info(f"{log_prefix}: Received [DONE] from non-Gemini endpoint.")
                        stream_proc_state["final_finish_reason_from_llm"] = stream_proc_state.get("final_finish_reason_from_llm","stop")
                        stream_proc_state["final_finish_event_sent_by_llm_reason"] = True
                        break

                    try:
                        parsed_sse_data = orjson.loads(sse_data_bytes)
                        async for event_dict in process_openai_like_sse_stream(parsed_sse_data, stream_proc_state, request_id):
                            yield orjson_dumps_bytes_wrapper(AppStreamEventPy(**event_dict).model_dump(by_alias=True, exclude_none=True))
                            if event_dict.get("type") == "finish" or stream_proc_state.get("final_finish_event_sent_by_llm_reason"):
                                return
                    except orjson.JSONDecodeError:
                        logger.warning(f"{log_prefix}: Failed to parse non-Gemini SSE JSON: {sse_data_bytes.decode(errors='ignore')[:100]}")
                    except Exception as e_proc_sse:
                        logger.error(f"{log_prefix}: Error processing non-Gemini SSE line: {sse_data_bytes.decode(errors='ignore')[:100]}, error: {e_proc_sse}", exc_info=True)
                
                if stream_proc_state.get("final_finish_event_sent_by_llm_reason"):
                    break
            
            if not stream_proc_state.get("final_finish_event_sent_by_llm_reason") and not stream_proc_state.get("final_finish_event_sent_flag_for_cleanup"):
                logger.info(f"{log_prefix}: (Non-Gemini-REST) Stream ended without explicit LLM finish signal.")

    except httpx.RequestError as e_req:
        logger.error(f"{log_prefix}: httpx.RequestError for non-Gemini model '{request_data.model}': {e_req}", exc_info=True)
        async for event_bytes in handle_stream_error(e_req, request_id, upstream_ok_flag, first_chunk_llm_received): yield event_bytes
        stream_proc_state["final_finish_event_sent_flag_for_cleanup"] = True
    except Exception as e_gen:
        logger.error(f"{log_prefix}: Generic error in non-Gemini stream for model '{request_data.model}': {e_gen}", exc_info=True)
        async for event_bytes in handle_stream_error(e_gen, request_id, upstream_ok_flag, first_chunk_llm_received): yield event_bytes
        stream_proc_state["final_finish_event_sent_flag_for_cleanup"] = True
    finally:
        logger.info(f"{log_prefix}: Cleaning up non-Gemini stream for model '{request_data.model}'.")
        async for event_bytes in handle_stream_cleanup(
            stream_proc_state, request_id, upstream_ok_flag,
            use_old_custom_separator_branch_flag,
            request_data.provider
        ):
            yield event_bytes
        
        logger.info(f"{log_prefix}: Deleting {len(temp_files_to_delete_after_stream)} temporary document file(s) for non-Gemini path.")
        for temp_file in temp_files_to_delete_after_stream:
            try:
                if os.path.exists(temp_file):
                    os.remove(temp_file)
            except Exception as e_del:
                logger.error(f"{log_prefix}: Error deleting temp file {temp_file}: {e_del}")