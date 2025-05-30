# eztalk_proxy/routers/multimodal_chat.py
import os
import logging
import httpx
import orjson
import asyncio
from typing import Optional, Dict, Any, AsyncGenerator, List

from fastapi import Request # Request might be needed for client disconnect check
from fastapi.responses import StreamingResponse # Not used directly here, but often in router files

from eztalk_proxy.models import ChatRequestModel, AppStreamEventPy, PartsApiMessagePy
from eztalk_proxy.multimodal_models import PyTextContentPart, PyInlineDataContentPart # Assuming PyFileUriContentPart might also exist

from eztalk_proxy.config import COMMON_HEADERS, API_TIMEOUT
from eztalk_proxy.utils import (
    get_current_time_iso,
    orjson_dumps_bytes_wrapper,
    strip_potentially_harmful_html_and_normalize_newlines,
    extract_sse_lines
)
from eztalk_proxy.multimodal_api_helpers import (
    prepare_gemini_rest_api_request
)
from eztalk_proxy.web_search import perform_web_search, generate_search_context_message_content

logger = logging.getLogger("EzTalkProxy.Routers.MultimodalChat")

async def sse_event_serializer_rest(event_data: AppStreamEventPy) -> bytes:
    return orjson_dumps_bytes_wrapper(event_data.model_dump(by_alias=True, exclude_none=True))

async def generate_gemini_rest_api_events_with_docs(
    gemini_chat_input: ChatRequestModel,
    fastapi_request_obj: Request,
    http_client: httpx.AsyncClient,
    request_id: str,
    extracted_document_text: Optional[str],
    temp_files_to_delete_after_stream: List[str]
) -> AsyncGenerator[bytes, None]:
    log_prefix = f"RID-{request_id}"
    first_chunk_received_from_llm = False
    final_finish_event_sent = False
    _had_any_reasoning_event_sent_in_stream = False
    _reasoning_finish_event_sent_flag = False
    original_user_text_found_in_parts = False

    active_messages_for_llm: List[PartsApiMessagePy] = []

    for msg_abstract in gemini_chat_input.messages:
        if isinstance(msg_abstract, PartsApiMessagePy):
            new_parts_for_gemini: List[Any] = []
            is_user_message = msg_abstract.role == "user"
            
            for original_part in msg_abstract.parts:
                if isinstance(original_part, PyTextContentPart):
                    new_parts_for_gemini.append(original_part.model_copy(deep=True) if hasattr(original_part, 'model_copy') else original_part.copy(deep=True))
                    if is_user_message and original_part.text and original_part.text.strip():
                        original_user_text_found_in_parts = True
                elif isinstance(original_part, PyInlineDataContentPart):
                    supported_inline_mimes = ["image/png", "image/jpeg", "image/webp", "image/heic", "image/heif", "video/mp4", "video/webm", "audio/mpeg", "audio/wav"] # Add more as needed
                    if original_part.mime_type.lower() in supported_inline_mimes:
                        new_parts_for_gemini.append(original_part.model_copy(deep=True) if hasattr(original_part, 'model_copy') else original_part.copy(deep=True))
                        if is_user_message:
                            original_user_text_found_in_parts = True
                    else:
                        logger.info(f"{log_prefix}: (Gemini REST) Ignoring inlineData part with unsupported MIME type '{original_part.mime_type}' for direct sending.")
            
            if new_parts_for_gemini or (is_user_message and not new_parts_for_gemini and not extracted_document_text and not original_user_text_found_in_parts):
                copied_msg_parts = list(new_parts_for_gemini) # Ensure it's a list
                copied_msg = PartsApiMessagePy(role=msg_abstract.role, parts=copied_msg_parts, type=msg_abstract.type)
                if hasattr(msg_abstract, 'name') and msg_abstract.name: copied_msg.name = msg_abstract.name
                active_messages_for_llm.append(copied_msg)

    if extracted_document_text:
        logger.info(f"{log_prefix}: (Gemini REST) Integrating extracted document text (length: {len(extracted_document_text)}).")
        doc_text_part = PyTextContentPart(type="text_content", text=extracted_document_text)
        
        last_user_message_index = -1
        for i in range(len(active_messages_for_llm) - 1, -1, -1):
            if active_messages_for_llm[i].role == "user":
                last_user_message_index = i
                break
        
        if last_user_message_index != -1:
            logger.debug(f"{log_prefix}: (Gemini REST) Appending extracted document text part to existing last user message parts.")
            active_messages_for_llm[last_user_message_index].parts.append(doc_text_part)
        else:
            logger.info(f"{log_prefix}: (Gemini REST) No prior user message found, creating new user message with extracted document text.")
            new_user_message_with_doc = PartsApiMessagePy(
                role="user",
                parts=[
                    PyTextContentPart(type="text_content", text="请基于以下文档内容进行处理或回答："),
                    doc_text_part
                ],
                type="parts_message"
            )
            active_messages_for_llm.append(new_user_message_with_doc)
        original_user_text_found_in_parts = True

    user_query_for_search_gemini = ""
    search_results_generated_this_time = False

    if active_messages_for_llm:
        last_user_message_for_search = next((msg for msg in reversed(active_messages_for_llm) if msg.role == "user"), None)
        if last_user_message_for_search:
            for part in last_user_message_for_search.parts:
                if isinstance(part, PyTextContentPart) and part.text:
                    user_query_for_search_gemini += part.text.strip() + " "
            user_query_for_search_gemini = user_query_for_search_gemini.strip()
    
    if gemini_chat_input.use_web_search and user_query_for_search_gemini:
        logger.info(f"{log_prefix}: (Gemini REST) Web search initiated for query: '{user_query_for_search_gemini[:100]}'")
        yield await sse_event_serializer_rest(AppStreamEventPy(type="status_update", stage="web_search_started", timestamp=get_current_time_iso()))
        
        search_results_list = await perform_web_search(user_query_for_search_gemini, request_id)
        
        if search_results_list:
            search_context_content = generate_search_context_message_content(user_query_for_search_gemini, search_results_list)
            search_context_parts = [PyTextContentPart(type="text_content", text=search_context_content)]
            search_context_api_message = PartsApiMessagePy(
                role="user", 
                parts=search_context_parts,
                type="parts_message"
            )
            
            last_user_idx = -1
            for i, msg_abstract in reversed(list(enumerate(active_messages_for_llm))):
                if msg_abstract.role == "user":
                    last_user_idx = i
                    break
            
            if last_user_idx != -1:
                active_messages_for_llm.insert(last_user_idx, search_context_api_message)
            else: 
                active_messages_for_llm.insert(0, search_context_api_message)

            search_results_generated_this_time = True
            logger.info(f"{log_prefix}: (Gemini REST) Web search context injected.")
            yield await sse_event_serializer_rest(AppStreamEventPy(type="status_update", stage="web_search_complete_with_results", query=user_query_for_search_gemini, timestamp=get_current_time_iso()))
            yield await sse_event_serializer_rest(AppStreamEventPy(type="web_search_results", results=search_results_list, timestamp=get_current_time_iso()))
        else:
            logger.info(f"{log_prefix}: (Gemini REST) Web search yielded no results for query '{user_query_for_search_gemini[:100]}'.")
            yield await sse_event_serializer_rest(AppStreamEventPy(type="status_update", stage="web_search_complete_no_results", query=user_query_for_search_gemini, timestamp=get_current_time_iso()))

    web_analysis_complete_sent = not (gemini_chat_input.use_web_search and user_query_for_search_gemini)

    try:
        if not gemini_chat_input.api_key:
            logger.error(f"{log_prefix}: (Gemini REST) API key is missing.")
            yield await sse_event_serializer_rest(AppStreamEventPy(type="error", message="Gemini API Key未在请求中提供。", timestamp=get_current_time_iso()))
            final_finish_event_sent = True; yield await sse_event_serializer_rest(AppStreamEventPy(type="finish", reason="configuration_error", timestamp=get_current_time_iso())); return

        temp_chat_input_for_prepare = gemini_chat_input.model_copy(deep=True) if hasattr(gemini_chat_input, 'model_copy') else gemini_chat_input.copy(deep=True)
        temp_chat_input_for_prepare.messages = active_messages_for_llm

        try:
            target_url, headers, json_payload = prepare_gemini_rest_api_request(
                chat_input=temp_chat_input_for_prepare, 
                request_id=request_id
            )
        except Exception as e_prepare:
            logger.error(f"{log_prefix}: (Gemini REST) Error preparing request: {e_prepare}", exc_info=True)
            yield await sse_event_serializer_rest(AppStreamEventPy(type="error", message=f"请求准备错误: {e_prepare}", timestamp=get_current_time_iso()))
            final_finish_event_sent = True; yield await sse_event_serializer_rest(AppStreamEventPy(type="finish", reason="request_error", timestamp=get_current_time_iso())); return

        if not json_payload.get("contents"):
            has_any_user_input = any(
                msg.role == "user" and any(isinstance(p, PyTextContentPart) and p.text and p.text.strip() for p in msg.parts)
                for msg in active_messages_for_llm
            )
            if not has_any_user_input:
                 logger.warning(f"{log_prefix}: (Gemini REST) No valid contents to send to model {gemini_chat_input.model}")
                 yield await sse_event_serializer_rest(AppStreamEventPy(type="error", message="没有有效内容发送给Gemini模型。", timestamp=get_current_time_iso()))
                 final_finish_event_sent = True; yield await sse_event_serializer_rest(AppStreamEventPy(type="finish", reason="no_content_error", timestamp=get_current_time_iso())); return
            else:
                logger.error(f"{log_prefix}: (Gemini REST) Contents are empty in json_payload despite having user/document text. This is unexpected but proceeding.")

        logger.info(f"{log_prefix}: (Gemini REST) Sending request to URL: {target_url.split('?key=')[0]}...") 
        logger.debug(f"{log_prefix}: (Gemini REST) Payload (contents preview): {[(c.get('role'), [p.get('text', 'NonTextPart')[:50] + '...' if len(p.get('text','')) > 50 else p.get('text','NonTextPart') for p in c.get('parts', [])]) for c in json_payload.get('contents', [])]}")

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
                except: pass
                yield await sse_event_serializer_rest(AppStreamEventPy(type="error", message=f"LLM API Error: {parsed_err_msg}", upstream_status=response.status_code, timestamp=get_current_time_iso()))
                final_finish_event_sent = True; yield await sse_event_serializer_rest(AppStreamEventPy(type="finish", reason="upstream_error", timestamp=get_current_time_iso())); return

            async for raw_chunk_bytes in response.aiter_raw():
                if await fastapi_request_obj.is_disconnected():
                    logger.info(f"{log_prefix}: (Gemini REST) Client disconnected.")
                    break

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
                    sse_data_bytes = b""
                    if sse_line_bytes.startswith(b"data: "):
                        sse_data_bytes = sse_line_bytes[len(b"data: "):].strip()
                    if not sse_data_bytes: continue
                    
                    logger.debug(f"{log_prefix}: (Gemini REST) Raw SSE Data: {sse_data_bytes!r}")

                    try:
                        chunk_json = orjson.loads(sse_data_bytes)
                        logger.debug(f"{log_prefix}: (Gemini REST) Parsed SSE Chunk JSON: {str(chunk_json)[:300]}")

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
                                                logger.debug(f"{log_prefix}: (Gemini REST Thought from part) '{clean_text[:100]}'")
                                                yield await sse_event_serializer_rest(AppStreamEventPy(type="reasoning", text=clean_text, timestamp=get_current_time_iso()))
                                                _had_any_reasoning_event_sent_in_stream = True
                                            else:
                                                if _had_any_reasoning_event_sent_in_stream and not _reasoning_finish_event_sent_flag:
                                                   logger.debug(f"{log_prefix}: (Gemini REST) Sending reasoning_finish as content starts after reasoning.")
                                                   yield await sse_event_serializer_rest(AppStreamEventPy(type="reasoning_finish", timestamp=get_current_time_iso()))
                                                   _reasoning_finish_event_sent_flag = True
                                                logger.debug(f"{log_prefix}: (Gemini REST Content) '{clean_text[:100]}'")
                                                yield await sse_event_serializer_rest(AppStreamEventPy(type="content", text=clean_text, timestamp=get_current_time_iso()))
                                
                                if "thinkingResult" in candidate and isinstance(candidate["thinkingResult"], dict) and candidate["thinkingResult"].get("chunks"):
                                    for thought_chunk_data in candidate["thinkingResult"]["chunks"]:
                                        thought_text = thought_chunk_data.get("text")
                                        if thought_text:
                                            clean_thought_text = strip_potentially_harmful_html_and_normalize_newlines(thought_text)
                                            if clean_thought_text:
                                                logger.debug(f"{log_prefix}: (Gemini REST Thought from thinkingResult) '{clean_thought_text[:100]}'")
                                                yield await sse_event_serializer_rest(AppStreamEventPy(type="reasoning", text=clean_thought_text, timestamp=get_current_time_iso()))
                                                _had_any_reasoning_event_sent_in_stream = True

                                finish_reason = candidate.get("finishReason")
                                if finish_reason:
                                    logger.info(f"{log_prefix}: (Gemini REST) Stream finished by LLM with reason: {finish_reason}")
                                    if _had_any_reasoning_event_sent_in_stream and not _reasoning_finish_event_sent_flag:
                                         logger.debug(f"{log_prefix}: (Gemini REST) Sending reasoning_finish before main finish due to LLM finish_reason.")
                                         yield await sse_event_serializer_rest(AppStreamEventPy(type="reasoning_finish", timestamp=get_current_time_iso()))
                                         _reasoning_finish_event_sent_flag = True
                                    final_finish_event_sent = True
                                    yield await sse_event_serializer_rest(AppStreamEventPy(type="finish", reason=finish_reason.lower(), timestamp=get_current_time_iso()))
                                    return 
                        
                        if "promptFeedback" in chunk_json:
                            logger.debug(f"{log_prefix}: (Gemini REST) Prompt Feedback: {chunk_json['promptFeedback']}")
                            block_reason = chunk_json.get("promptFeedback", {}).get("blockReason")
                            if block_reason:
                                logger.warning(f"{log_prefix}: (Gemini REST) Prompt blocked by API with reason: {block_reason}")
                                error_message_for_client = f"请求被模型提供方阻止：{block_reason}。"
                                safety_ratings = chunk_json.get("promptFeedback", {}).get("safetyRatings")
                                if safety_ratings:
                                    error_message_for_client += f" 安全评级详情: {str(safety_ratings)[:100]}"
                                
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
            
            if not final_finish_event_sent:
                logger.info(f"{log_prefix}: (Gemini REST) Stream iterator finished without explicit finish_reason from last chunk.")
                if _had_any_reasoning_event_sent_in_stream and not _reasoning_finish_event_sent_flag:
                   logger.debug(f"{log_prefix}: (Gemini REST) Sending reasoning_finish as stream ends.")
                   yield await sse_event_serializer_rest(AppStreamEventPy(type="reasoning_finish", timestamp=get_current_time_iso()))
                   _reasoning_finish_event_sent_flag = True
                final_finish_event_sent = True
                yield await sse_event_serializer_rest(AppStreamEventPy(type="finish", reason="stream_end", timestamp=get_current_time_iso()))

    except httpx.RequestError as e_req: 
        logger.error(f"{log_prefix}: (Gemini REST) httpx.RequestError for model '{gemini_chat_input.model}': {e_req}", exc_info=True)
        yield await sse_event_serializer_rest(AppStreamEventPy(type="error", message=f"网络请求错误: {e_req}", timestamp=get_current_time_iso()))
        if not final_finish_event_sent: final_finish_event_sent = True; yield await sse_event_serializer_rest(AppStreamEventPy(type="finish", reason="network_error", timestamp=get_current_time_iso()))
    except Exception as e_gen:
        logger.error(f"{log_prefix}: (Gemini REST) Generic error for model '{gemini_chat_input.model}': {e_gen}", exc_info=True)
        yield await sse_event_serializer_rest(AppStreamEventPy(type="error", message=f"处理Gemini REST请求时发生未知错误: {str(e_gen)[:200]}", timestamp=get_current_time_iso()))
        if not final_finish_event_sent: final_finish_event_sent = True; yield await sse_event_serializer_rest(AppStreamEventPy(type="finish", reason="unknown_error", timestamp=get_current_time_iso()))
    finally:
        logger.info(f"{log_prefix}: (Gemini REST) Stream generation loop for model {gemini_chat_input.model} concluded.")
        if not final_finish_event_sent:
            logger.warning(f"{log_prefix}: (Gemini REST) Forcing final 'finish' event in finally block.")
            if _had_any_reasoning_event_sent_in_stream and not _reasoning_finish_event_sent_flag:
                 logger.debug(f"{log_prefix}: (Gemini REST) Sending reasoning_finish in finally block.")
                 yield await sse_event_serializer_rest(AppStreamEventPy(type="reasoning_finish", timestamp=get_current_time_iso()))
            yield await sse_event_serializer_rest(AppStreamEventPy(type="finish", reason="cleanup_stream_end_gemini_rest", timestamp=get_current_time_iso()))
        
        logger.info(f"{log_prefix}: Deleting {len(temp_files_to_delete_after_stream)} temporary document file(s) for Gemini REST path.")
        for temp_file in temp_files_to_delete_after_stream:
            try:
                if os.path.exists(temp_file):
                    os.remove(temp_file)
                    logger.debug(f"{log_prefix}: Deleted temp file: {temp_file}")
            except Exception as e_del:
                logger.error(f"{log_prefix}: Error deleting temp file {temp_file}: {e_del}")

async def handle_gemini_request_entry(
    gemini_chat_input: ChatRequestModel,
    raw_request: Request,
    http_client: httpx.AsyncClient,
    request_id: str
):
    logger.warning(f"RID-{request_id}: handle_gemini_request_entry was called. This entry point assumes document text is already integrated or not applicable.")
    # This function is now primarily a legacy entry or for specific scenarios where documents are not handled via the main chat endpoint.
    # It will call generate_gemini_rest_api_events_with_docs without providing extracted text.
    return StreamingResponse(
        generate_gemini_rest_api_events_with_docs(
             gemini_chat_input=gemini_chat_input,
             fastapi_request_obj=raw_request,
             http_client=http_client,
             request_id=request_id,
             extracted_document_text=None, # No document text provided by this specific entry
             temp_files_to_delete_after_stream=[] # No files to delete managed by this specific entry
        ),
        media_type="text/event-stream",
        headers=COMMON_HEADERS
    )