import os
import logging
import httpx
import orjson
import asyncio
import base64
from typing import Optional, Dict, Any, AsyncGenerator, List

from fastapi import Request, UploadFile
from fastapi.responses import StreamingResponse

from eztalk_proxy.models import (
    ChatRequestModel,
    AppStreamEventPy,
    PartsApiMessagePy,
    AbstractApiMessagePy,
    SimpleTextApiMessagePy
)
from eztalk_proxy.multimodal_models import (
    PyTextContentPart,
    PyInlineDataContentPart,
    IncomingApiContentPart,
    PyFileUriContentPart
)
from eztalk_proxy.config import (
    COMMON_HEADERS,
    API_TIMEOUT,
    GEMINI_SUPPORTED_UPLOAD_MIMETYPES,
    GEMINI_ENABLE_GCS_UPLOAD,
    GCS_BUCKET_NAME,
    GCS_PROJECT_ID
)
from eztalk_proxy.utils import (
    get_current_time_iso,
    orjson_dumps_bytes_wrapper,
    strip_potentially_harmful_html_and_normalize_newlines,
    extract_sse_lines,
    upload_to_gcs
)
from eztalk_proxy.multimodal_api_helpers import (
    prepare_gemini_rest_api_request
)
from eztalk_proxy.web_search import perform_web_search, generate_search_context_message_content

logger = logging.getLogger("EzTalkProxy.Routers.MultimodalChat")

IMAGE_MIME_TYPES = ["image/png", "image/jpeg", "image/webp", "image/heic", "image/heif"]
VIDEO_AUDIO_MIME_TYPES = [
    "video/mp4", "application/mp4", "video/mpeg", "video/quicktime", "video/x-msvideo",
    "video/x-flv", "video/x-matroska", "video/webm", "video/x-ms-wmv",
    "video/3gpp", "video/x-m4v", "audio/wav", "audio/x-wav", "audio/mpeg",
    "audio/aac", "audio/ogg", "audio/opus", "audio/flac", "audio/midi",
    "audio/amr", "audio/aiff", "audio/x-m4a"
]


async def sse_event_serializer_rest(event_data: AppStreamEventPy) -> bytes:
    return orjson_dumps_bytes_wrapper(event_data.model_dump(by_alias=True, exclude_none=True))

async def generate_gemini_rest_api_events_with_docs(
    gemini_chat_input: ChatRequestModel,
    fastapi_request_obj: Request,
    http_client: httpx.AsyncClient,
    request_id: str,
    uploaded_files_for_gemini: Optional[List[UploadFile]],
    additional_extracted_text: Optional[str],
    temp_files_to_delete_after_stream: List[str]
) -> AsyncGenerator[bytes, None]:
    log_prefix = f"RID-{request_id}"
    first_chunk_received_from_llm = False
    final_finish_event_sent = False
    _had_any_reasoning_event_sent_in_stream = False
    _reasoning_finish_event_sent_flag = False
    
    active_messages_for_llm: List[AbstractApiMessagePy] = []
    gcs_uris_created_this_request: List[str] = []

    for msg_abstract_orig in gemini_chat_input.messages:
        active_messages_for_llm.append(msg_abstract_orig.model_copy(deep=True))

    newly_created_multimodal_parts: List[IncomingApiContentPart] = []
    
    if uploaded_files_for_gemini:
        logger.info(f"{log_prefix}: Processing {len(uploaded_files_for_gemini)} uploaded files for Gemini multimodal content.")
        for uploaded_file in uploaded_files_for_gemini:
            mime_type = uploaded_file.content_type.lower() if uploaded_file.content_type else ""
            filename = uploaded_file.filename or "unknown_file"
            file_processed_for_gemini = False

            try:
                if GEMINI_ENABLE_GCS_UPLOAD and mime_type in VIDEO_AUDIO_MIME_TYPES and GCS_BUCKET_NAME:
                    logger.info(f"{log_prefix}: Attempting GCS upload for '{filename}' (MIME: {mime_type}).")
                    await uploaded_file.seek(0)
                    gcs_uri = await upload_to_gcs(
                        file_obj=uploaded_file,
                        original_filename=filename,
                        bucket_name=GCS_BUCKET_NAME,
                        project_id=GCS_PROJECT_ID,
                        content_type=mime_type,
                        request_id=request_id
                    )
                    if gcs_uri:
                        file_uri_part = PyFileUriContentPart(
                            type="file_uri_content",
                            uri=gcs_uri,
                            mimeType=mime_type
                        )
                        newly_created_multimodal_parts.append(file_uri_part)
                        gcs_uris_created_this_request.append(gcs_uri)
                        logger.info(f"{log_prefix}: Successfully uploaded '{filename}' to GCS: {gcs_uri}")
                        file_processed_for_gemini = True
                    else:
                        logger.warning(f"{log_prefix}: GCS upload failed for '{filename}'. Will not be used for Gemini content part.")
                
                if not file_processed_for_gemini and mime_type in IMAGE_MIME_TYPES:
                    logger.info(f"{log_prefix}: Attempting Base64 encoding for image '{filename}' (MIME: {mime_type}).")
                    await uploaded_file.seek(0)
                    file_content_bytes = await uploaded_file.read()
                    base64_encoded_data = base64.b64encode(file_content_bytes).decode('utf-8')
                    inline_part = PyInlineDataContentPart(
                        type="inline_data_content",
                        mimeType=mime_type,
                        base64Data=base64_encoded_data
                    )
                    newly_created_multimodal_parts.append(inline_part)
                    logger.info(f"{log_prefix}: Successfully Base64 encoded '{filename}' for Gemini.")
                    file_processed_for_gemini = True

                if not file_processed_for_gemini:
                    if mime_type and mime_type not in GEMINI_SUPPORTED_UPLOAD_MIMETYPES:
                         logger.warning(f"{log_prefix}: Skipping file '{filename}' with unsupported MIME type '{mime_type}' for Gemini direct multimodal input.")
                    elif not mime_type:
                         logger.warning(f"{log_prefix}: Skipping file '{filename}' due to missing MIME type for Gemini direct multimodal input.")
                    else:
                         logger.warning(f"{log_prefix}: File '{filename}' (MIME: {mime_type}) was not processed for Gemini content (GCS disabled/failed and not a Base64 image).")

            except Exception as e_file_proc:
                logger.error(f"{log_prefix}: Error processing file '{filename}' for Gemini: {e_file_proc}", exc_info=True)
            finally:
                pass
    
    if additional_extracted_text:
        logger.info(f"{log_prefix}: Adding additionally extracted text (len: {len(additional_extracted_text)}) as a text part for Gemini.")
        doc_text_part = PyTextContentPart(type="text_content", text=additional_extracted_text)
        newly_created_multimodal_parts.append(doc_text_part)

    if newly_created_multimodal_parts:
        last_user_message_idx = -1
        for i in range(len(active_messages_for_llm) - 1, -1, -1):
            if active_messages_for_llm[i].role == "user":
                last_user_message_idx = i
                break
        
        if last_user_message_idx != -1:
            user_msg_abstract = active_messages_for_llm[last_user_message_idx]
            if isinstance(user_msg_abstract, PartsApiMessagePy):
                updated_parts = list(user_msg_abstract.parts) + newly_created_multimodal_parts
                active_messages_for_llm[last_user_message_idx] = PartsApiMessagePy(
                    role=user_msg_abstract.role, parts=updated_parts, message_type="parts_message",
                    name=user_msg_abstract.name, tool_calls=user_msg_abstract.tool_calls,
                    tool_call_id=user_msg_abstract.tool_call_id
                )
            elif isinstance(user_msg_abstract, SimpleTextApiMessagePy):
                initial_text_part = [PyTextContentPart(type="text_content", text=user_msg_abstract.content)] if user_msg_abstract.content else []
                combined_parts = initial_text_part + newly_created_multimodal_parts
                active_messages_for_llm[last_user_message_idx] = PartsApiMessagePy(
                    role=user_msg_abstract.role, parts=combined_parts, message_type="parts_message",
                    name=user_msg_abstract.name, tool_calls=user_msg_abstract.tool_calls,
                    tool_call_id=user_msg_abstract.tool_call_id
                )
        else:
            default_prompt_for_multimodal = "请分析以下内容："
            is_only_extracted_text = len(newly_created_multimodal_parts) == 1 and \
                                     isinstance(newly_created_multimodal_parts[0], PyTextContentPart) and \
                                     newly_created_multimodal_parts[0].text == additional_extracted_text
            final_parts_for_new_message = []
            if not any(isinstance(p, PyTextContentPart) and p.text.strip() for p in newly_created_multimodal_parts) \
               or is_only_extracted_text :
                final_parts_for_new_message.append(PyTextContentPart(type="text_content", text=default_prompt_for_multimodal))
            final_parts_for_new_message.extend(newly_created_multimodal_parts)
            new_user_message = PartsApiMessagePy(
                role="user", parts=final_parts_for_new_message, message_type="parts_message"
            )
            active_messages_for_llm.append(new_user_message)

    user_query_for_search_gemini = ""
    search_results_generated_this_time = False
    if active_messages_for_llm:
        last_user_message_for_search = next((msg for msg in reversed(active_messages_for_llm) if msg.role == "user"), None)
        if last_user_message_for_search:
            if isinstance(last_user_message_for_search, PartsApiMessagePy):
                for part in last_user_message_for_search.parts:
                    if isinstance(part, PyTextContentPart) and part.text:
                        user_query_for_search_gemini += part.text.strip() + " "
                user_query_for_search_gemini = user_query_for_search_gemini.strip()
            elif isinstance(last_user_message_for_search, SimpleTextApiMessagePy):
                 user_query_for_search_gemini = last_user_message_for_search.content.strip()
    
    if gemini_chat_input.use_web_search and user_query_for_search_gemini:
        logger.info(f"{log_prefix}: Checking web search for Gemini. use_web_search: {gemini_chat_input.use_web_search}, user_query_for_search_gemini: '{user_query_for_search_gemini}'")
        yield await sse_event_serializer_rest(AppStreamEventPy(type="status_update", stage="web_search_started", timestamp=get_current_time_iso()))
        search_results_list = await perform_web_search(user_query_for_search_gemini, request_id)
        if search_results_list:
            search_context_content = generate_search_context_message_content(user_query_for_search_gemini, search_results_list)
            search_context_parts = [PyTextContentPart(type="text_content", text=search_context_content)]
            try:
                search_context_api_message = PartsApiMessagePy(
                    role="user", parts=search_context_parts, message_type="parts_message"
                )
                last_user_idx = -1
                for i, msg_abstract_loop in reversed(list(enumerate(active_messages_for_llm))):
                    if msg_abstract_loop.role == "user":
                        last_user_idx = i
                        break
                if last_user_idx != -1:
                    active_messages_for_llm.insert(last_user_idx, search_context_api_message)
                else:
                    active_messages_for_llm.insert(0, search_context_api_message)
                search_results_generated_this_time = True
                yield await sse_event_serializer_rest(AppStreamEventPy(type="status_update", stage="web_search_complete_with_results", query=user_query_for_search_gemini, timestamp=get_current_time_iso()))
                yield await sse_event_serializer_rest(AppStreamEventPy(type="web_search_results", results=search_results_list, timestamp=get_current_time_iso()))
            except Exception as e_instantiate_search:
                logger.error(f"{log_prefix}: FAILED to instantiate PartsApiMessagePy for search context. Error: {e_instantiate_search}", exc_info=True)
        else:
            yield await sse_event_serializer_rest(AppStreamEventPy(type="status_update", stage="web_search_complete_no_results", query=user_query_for_search_gemini, timestamp=get_current_time_iso()))

    web_analysis_complete_sent = not (gemini_chat_input.use_web_search and user_query_for_search_gemini)
    try:
        if not gemini_chat_input.api_key:
            yield await sse_event_serializer_rest(AppStreamEventPy(type="error", message="Gemini API Key未在请求中提供。", timestamp=get_current_time_iso()))
            final_finish_event_sent = True; yield await sse_event_serializer_rest(AppStreamEventPy(type="finish", reason="configuration_error", timestamp=get_current_time_iso())); return

        temp_chat_input_for_prepare = gemini_chat_input.model_copy(deep=True)
        temp_chat_input_for_prepare.messages = active_messages_for_llm

        try:
            target_url, headers, json_payload = prepare_gemini_rest_api_request(chat_input=temp_chat_input_for_prepare, request_id=request_id)
        except Exception as e_prepare:
            logger.error(f"{log_prefix}: (Gemini REST) Request preparation error: {e_prepare}", exc_info=True)
            yield await sse_event_serializer_rest(AppStreamEventPy(type="error", message=f"请求准备错误: {e_prepare}", timestamp=get_current_time_iso()))
            final_finish_event_sent = True; yield await sse_event_serializer_rest(AppStreamEventPy(type="finish", reason="request_error", timestamp=get_current_time_iso())); return
        
        if not json_payload.get("contents"):
            has_any_user_input_in_active = any(
                msg.role == "user" and (
                    (isinstance(msg, PartsApiMessagePy) and any(part for part in msg.parts)) or
                    (isinstance(msg, SimpleTextApiMessagePy) and msg.content and msg.content.strip())
                ) for msg in active_messages_for_llm
            )
            if not has_any_user_input_in_active:
                 logger.warning(f"{log_prefix}: (Gemini REST) No valid user content (text or multimodal) to send to Gemini model after processing.")
                 yield await sse_event_serializer_rest(AppStreamEventPy(type="error", message="没有有效内容发送给Gemini模型。", timestamp=get_current_time_iso()))
                 final_finish_event_sent = True; yield await sse_event_serializer_rest(AppStreamEventPy(type="finish", reason="no_content_error", timestamp=get_current_time_iso())); return
            else:
                logger.error(f"{log_prefix}: (Gemini REST) Contents are empty in json_payload despite having user/document text/parts in active_messages_for_llm. This is unexpected but proceeding. Active messages types: {[(m.role, m.message_type) for m in active_messages_for_llm]}")
        
        logger.info(f"{log_prefix}: (Gemini REST) Sending request to URL: {target_url.split('?key=')[0]}...")
        if "contents" in json_payload and json_payload["contents"]:
            contents_preview = []
            for c_idx, c_content in enumerate(json_payload["contents"]):
                role = c_content.get("role", "unknown_role")
                parts_preview_list = []
                for p_idx, p_part in enumerate(c_content.get("parts", [])):
                    if "text" in p_part:
                        part_text_preview = p_part["text"][:50] + '...' if len(p_part["text"]) > 50 else p_part["text"]
                        parts_preview_list.append(f"Part[{p_idx}]: Text='{part_text_preview}'")
                    elif "inlineData" in p_part:
                        parts_preview_list.append(f"Part[{p_idx}]: InlineData MIME='{p_part.get('inlineData',{}).get('mimeType')}'")
                    elif "fileData" in p_part:
                        parts_preview_list.append(f"Part[{p_idx}]: FileData URI='{p_part.get('fileData',{}).get('fileUri')}'")
                    else:
                        parts_preview_list.append(f"Part[{p_idx}]: UnknownPartStructure")
                contents_preview.append(f"Content[{c_idx}]: Role='{role}', Parts={parts_preview_list}")
        
        buffer = bytearray()
        async with http_client.stream("POST", target_url, headers=headers, json=json_payload, timeout=API_TIMEOUT) as response:
            logger.info(f"{log_prefix}: (Gemini REST) Upstream LLM response status: {response.status_code}")
            if not (200 <= response.status_code < 300):
                err_body_bytes = await response.aread()
                err_text = err_body_bytes.decode("utf-8", errors="replace")
                logger.error(f"{log_prefix}: (Gemini REST) Upstream LLM error {response.status_code}: {err_text[:1000]}")
                parsed_err_msg = err_text[:200]
                try:
                    err_json = orjson.loads(err_text)
                    parsed_err_msg = err_json.get("error", {}).get("message", parsed_err_msg)
                except orjson.JSONDecodeError:
                    pass
                except Exception as e_parse_err:
                     logger.warning(f"{log_prefix}: (Gemini REST) Unexpected error parsing error body: {e_parse_err}")
                yield await sse_event_serializer_rest(AppStreamEventPy(type="error", message=f"LLM API Error: {parsed_err_msg}", upstream_status=response.status_code, timestamp=get_current_time_iso()))
                final_finish_event_sent = True; yield await sse_event_serializer_rest(AppStreamEventPy(type="finish", reason="upstream_error", timestamp=get_current_time_iso())); return
            
            async for raw_chunk_bytes in response.aiter_raw():
                if await fastapi_request_obj.is_disconnected():
                    logger.info(f"{log_prefix}: (Gemini REST) Client disconnected."); break
                if not first_chunk_received_from_llm:
                    if not web_analysis_complete_sent and gemini_chat_input.use_web_search and user_query_for_search_gemini:
                        stage_after_search = "web_analysis_complete" if search_results_generated_this_time else "web_analysis_skipped_no_results"
                        yield await sse_event_serializer_rest(AppStreamEventPy(type="status_update", stage=stage_after_search, timestamp=get_current_time_iso()))
                        web_analysis_complete_sent = True
                    first_chunk_received_from_llm = True
                buffer.extend(raw_chunk_bytes)
                sse_lines, buffer = extract_sse_lines(buffer)
                for sse_line_bytes in sse_lines:
                    if not sse_line_bytes.strip(): continue
                    sse_data_bytes = b"";
                    if sse_line_bytes.startswith(b"data: "):
                        sse_data_bytes = sse_line_bytes[len(b"data: "):].strip()
                    if not sse_data_bytes: continue
                    try:
                        chunk_json = orjson.loads(sse_data_bytes)
                        if "candidates" in chunk_json and chunk_json["candidates"]:
                            for candidate in chunk_json["candidates"]:
                                if "content" in candidate and "parts" in candidate["content"]:
                                    for part_data in candidate["content"]["parts"]:
                                        part_text = part_data.get("text")
                                        is_thought = part_data.get("thought") is True
                                        if part_text:
                                            clean_text = strip_potentially_harmful_html_and_normalize_newlines(part_text)
                                            if not clean_text: continue
                                            if is_thought:
                                                yield await sse_event_serializer_rest(AppStreamEventPy(type="reasoning", text=clean_text, timestamp=get_current_time_iso()))
                                                _had_any_reasoning_event_sent_in_stream = True
                                            else:
                                                if _had_any_reasoning_event_sent_in_stream and not _reasoning_finish_event_sent_flag:
                                                   yield await sse_event_serializer_rest(AppStreamEventPy(type="reasoning_finish", timestamp=get_current_time_iso()))
                                                   _reasoning_finish_event_sent_flag = True
                                                yield await sse_event_serializer_rest(AppStreamEventPy(type="content", text=clean_text, timestamp=get_current_time_iso()))
                                if "thinkingResult" in candidate and isinstance(candidate["thinkingResult"], dict) and candidate["thinkingResult"].get("chunks"):
                                    for thought_chunk_data in candidate["thinkingResult"]["chunks"]:
                                        thought_text = thought_chunk_data.get("text")
                                        if thought_text:
                                            clean_thought_text = strip_potentially_harmful_html_and_normalize_newlines(thought_text)
                                            if clean_thought_text:
                                                yield await sse_event_serializer_rest(AppStreamEventPy(type="reasoning", text=clean_thought_text, timestamp=get_current_time_iso()))
                                                _had_any_reasoning_event_sent_in_stream = True
                                finish_reason = candidate.get("finishReason")
                                if finish_reason:
                                    if _had_any_reasoning_event_sent_in_stream and not _reasoning_finish_event_sent_flag:
                                         yield await sse_event_serializer_rest(AppStreamEventPy(type="reasoning_finish", timestamp=get_current_time_iso()))
                                    final_finish_event_sent = True
                                    yield await sse_event_serializer_rest(AppStreamEventPy(type="finish", reason=finish_reason.lower(), timestamp=get_current_time_iso()))
                                    return
                        if "promptFeedback" in chunk_json:
                            block_reason = chunk_json.get("promptFeedback", {}).get("blockReason")
                            if block_reason:
                                error_message_for_client = f"请求被模型提供方阻止：{block_reason}。"
                                safety_ratings = chunk_json.get("promptFeedback", {}).get("safetyRatings")
                                if safety_ratings:
                                    error_message_for_client += f" 安全评级详情: {str(safety_ratings)[:100]}"
                                logger.warning(f"{log_prefix}: (Gemini REST) Prompt blocked: {block_reason}, Ratings: {safety_ratings}")
                                yield await sse_event_serializer_rest(AppStreamEventPy(type="error", message=error_message_for_client, timestamp=get_current_time_iso()))
                                if not final_finish_event_sent:
                                    final_finish_event_sent = True
                                    mapped_finish_reason = f"blocked_{block_reason.lower()}" if block_reason else "blocked_unknown"
                                    yield await sse_event_serializer_rest(AppStreamEventPy(type="finish", reason=mapped_finish_reason, timestamp=get_current_time_iso()))
                                return
                    except orjson.JSONDecodeError:
                        logger.warning(f"{log_prefix}: (Gemini REST) Failed to parse SSE JSON: {sse_data_bytes.decode(errors='ignore')[:100]}")
                    except Exception as e_proc_sse:
                        logger.error(f"{log_prefix}: (Gemini REST) Error processing SSE line: {sse_data_bytes.decode(errors='ignore')[:100]}, error: {e_proc_sse}", exc_info=True)
            
            if await fastapi_request_obj.is_disconnected():
                logger.info(f"{log_prefix}: (Gemini REST) Client disconnected after stream completion.")
            elif not final_finish_event_sent:
                logger.info(f"{log_prefix}: (Gemini REST) Stream ended, but no explicit finish_reason received from LLM. Sending stream_end.")
                if _had_any_reasoning_event_sent_in_stream and not _reasoning_finish_event_sent_flag:
                   yield await sse_event_serializer_rest(AppStreamEventPy(type="reasoning_finish", timestamp=get_current_time_iso()))
                final_finish_event_sent = True
                yield await sse_event_serializer_rest(AppStreamEventPy(type="finish", reason="stream_end", timestamp=get_current_time_iso()))

    except httpx.RequestError as e_req:
        logger.error(f"{log_prefix}: (Gemini REST) HTTPX RequestError: {e_req}", exc_info=True)
        yield await sse_event_serializer_rest(AppStreamEventPy(type="error", message=f"网络请求错误: {e_req}", timestamp=get_current_time_iso()))
        if not final_finish_event_sent:
            final_finish_event_sent = True
            yield await sse_event_serializer_rest(AppStreamEventPy(type="finish", reason="network_error", timestamp=get_current_time_iso()))
    except Exception as e_gen:
        logger.error(f"{log_prefix}: (Gemini REST) General error in generate_gemini_rest_api_events_with_docs: {e_gen}", exc_info=True)
        yield await sse_event_serializer_rest(AppStreamEventPy(type="error", message=f"处理Gemini REST请求时发生未知错误: {str(e_gen)[:200]}", timestamp=get_current_time_iso()))
        if not final_finish_event_sent:
            final_finish_event_sent = True
            yield await sse_event_serializer_rest(AppStreamEventPy(type="finish", reason="unknown_error", timestamp=get_current_time_iso()))
    finally:
        if not final_finish_event_sent:
            logger.warning(f"{log_prefix}: (Gemini REST) Reached finally block without sending a finish event. Sending cleanup_stream_end.")
            if _had_any_reasoning_event_sent_in_stream and not _reasoning_finish_event_sent_flag:
                 yield await sse_event_serializer_rest(AppStreamEventPy(type="reasoning_finish", timestamp=get_current_time_iso()))
            yield await sse_event_serializer_rest(AppStreamEventPy(type="finish", reason="cleanup_stream_end_gemini_rest", timestamp=get_current_time_iso()))
        
        if temp_files_to_delete_after_stream:
            logger.info(f"{log_prefix}: Deleting {len(temp_files_to_delete_after_stream)} temporary document file(s) passed from caller (chat.py).")
            for temp_file in temp_files_to_delete_after_stream:
                try:
                    if os.path.exists(temp_file):
                        os.remove(temp_file)
                except Exception as e_del:
                    logger.error(f"{log_prefix}: Error deleting temp file {temp_file}: {e_del}")
        

async def handle_gemini_request_entry(
    gemini_chat_input: ChatRequestModel,
    raw_request: Request,
    http_client: httpx.AsyncClient,
    request_id: str,
):
    logger.warning(f"RID-{request_id}: handle_gemini_request_entry was called. Ensure it handles multimodal inputs correctly if files are involved via a different mechanism.")
    return StreamingResponse(
        generate_gemini_rest_api_events_with_docs(
             gemini_chat_input=gemini_chat_input,
             fastapi_request_obj=raw_request,
             http_client=http_client,
             request_id=request_id,
             uploaded_files_for_gemini=None,
             additional_extracted_text=None,
             temp_files_to_delete_after_stream=[]
        ),
        media_type="text/event-stream",
        headers=COMMON_HEADERS
    )