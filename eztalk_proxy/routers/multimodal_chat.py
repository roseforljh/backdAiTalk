import os
import logging
import httpx
import orjson
import asyncio
from typing import Optional, Dict, Any, AsyncGenerator, List

from fastapi import Request
from fastapi.responses import StreamingResponse

# 确保从正确的路径导入
from eztalk_proxy.models import ChatRequestModel, AppStreamEventPy, PartsApiMessagePy, AbstractApiMessagePy, SimpleTextApiMessagePy
from eztalk_proxy.multimodal_models import PyTextContentPart, PyInlineDataContentPart, IncomingApiContentPart

# 假设这些在您的项目中存在
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
# 假设 web_search 模块和函数存在
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

    active_messages_for_llm: List[AbstractApiMessagePy] = [] # 使用 AbstractApiMessagePy 以便后续 prepare_gemini_rest_api_request 能正确处理

    # --- DEBUGGING msg_abstract in gemini_chat_input.messages ---
    logger.info(f"{log_prefix}: Initial gemini_chat_input.messages count: {len(gemini_chat_input.messages)}")
    for idx, msg_debug in enumerate(gemini_chat_input.messages):
        logger.info(f"{log_prefix}: DEBUG Initial msg_debug[{idx}] type: {type(msg_debug)}")
        if hasattr(msg_debug, 'model_dump'):
            try:
                logger.info(f"{log_prefix}: DEBUG Initial msg_debug[{idx}].model_dump(by_alias=True): {msg_debug.model_dump(by_alias=True, exclude_none=True)}")
                logger.info(f"{log_prefix}: DEBUG Initial msg_debug[{idx}].model_dump(by_alias=False): {msg_debug.model_dump(by_alias=False, exclude_none=True)}")
            except Exception as e_dump_initial:
                logger.error(f"{log_prefix}: DEBUG Initial Error dumping msg_debug[{idx}]: {e_dump_initial}")
        else:
            logger.info(f"{log_prefix}: DEBUG Initial msg_debug[{idx}] (raw, no model_dump): {msg_debug}")
    # --- END DEBUGGING ---

    for msg_abstract in gemini_chat_input.messages:
        # 使用 AbstractApiMessagePy 以便 Pydantic 的辨别器能正确工作
        if isinstance(msg_abstract, PartsApiMessagePy):
            new_parts_for_gemini: List[IncomingApiContentPart] = [] # 明确类型
            is_user_message = msg_abstract.role == "user"
            
            for original_part in msg_abstract.parts:
                if isinstance(original_part, PyTextContentPart):
                    # 使用 model_copy 替代 copy，Pydantic V2 推荐
                    new_parts_for_gemini.append(original_part.model_copy(deep=True))
                    if is_user_message and original_part.text and original_part.text.strip():
                        original_user_text_found_in_parts = True
                elif isinstance(original_part, PyInlineDataContentPart):
                    supported_inline_mimes = ["image/png", "image/jpeg", "image/webp", "image/heic", "image/heif", "video/mp4", "video/webm", "audio/mpeg", "audio/wav"]
                    if original_part.mime_type.lower() in supported_inline_mimes:
                        new_parts_for_gemini.append(original_part.model_copy(deep=True))
                        if is_user_message:
                            original_user_text_found_in_parts = True
                    else:
                        logger.info(f"{log_prefix}: (Gemini REST) Ignoring inlineData part with unsupported MIME type '{original_part.mime_type}' for direct sending.")
                # 可以添加对 PyFileUriContentPart 的处理，如果需要的话
            
            if new_parts_for_gemini or (is_user_message and not new_parts_for_gemini and not extracted_document_text and not original_user_text_found_in_parts):
                copied_msg_parts = list(new_parts_for_gemini) # 确保是列表副本
                
                # --- DEBUGGING right before PartsApiMessagePy instantiation ---
                logger.info(f"{log_prefix}: DEBUG (Loop): msg_abstract type: {type(msg_abstract)}")
                if hasattr(msg_abstract, 'model_dump'):
                    try:
                        logger.info(f"{log_prefix}: DEBUG (Loop): msg_abstract.model_dump(by_alias=True): {msg_abstract.model_dump(by_alias=True, exclude_none=True)}")
                        logger.info(f"{log_prefix}: DEBUG (Loop): msg_abstract.model_dump(by_alias=False): {msg_abstract.model_dump(by_alias=False, exclude_none=True)}")
                    except Exception as e_dump_loop:
                        logger.error(f"{log_prefix}: DEBUG (Loop): Error dumping msg_abstract: {e_dump_loop}")
                else:
                    logger.info(f"{log_prefix}: DEBUG (Loop): msg_abstract (raw, no model_dump): {msg_abstract}")
                logger.info(f"{log_prefix}: DEBUG (Loop): copied_msg_parts before PartsApiMessagePy: {copied_msg_parts}")
                logger.info(f"{log_prefix}: DEBUG (Loop): Instantiating PartsApiMessagePy with role='{msg_abstract.role}', message_type='parts_message'")
                # --- END DEBUGGING ---

                try:
                    copied_msg = PartsApiMessagePy(
                        role=msg_abstract.role,
                        parts=copied_msg_parts,
                        message_type="parts_message" # 显式提供 Python 字段名和其应有的值
                    )
                    # 保留可选字段
                    if hasattr(msg_abstract, 'name') and msg_abstract.name: copied_msg.name = msg_abstract.name
                    if hasattr(msg_abstract, 'tool_calls') and msg_abstract.tool_calls: copied_msg.tool_calls = msg_abstract.tool_calls
                    if hasattr(msg_abstract, 'tool_call_id') and msg_abstract.tool_call_id: copied_msg.tool_call_id = msg_abstract.tool_call_id
                    
                    active_messages_for_llm.append(copied_msg)
                except Exception as e_instantiate_copied:
                    logger.error(f"{log_prefix}: FAILED to instantiate PartsApiMessagePy (copied_msg). Error: {e_instantiate_copied}", exc_info=True)
                    # 如果这里失败，抛出或返回错误事件
                    yield await sse_event_serializer_rest(AppStreamEventPy(type="error", message=f"Internal error creating message: {e_instantiate_copied}", timestamp=get_current_time_iso()))
                    if not final_finish_event_sent: final_finish_event_sent = True; yield await sse_event_serializer_rest(AppStreamEventPy(type="finish", reason="internal_error", timestamp=get_current_time_iso())); return
                    # 继续下一个循环项可能不安全，或者直接返回
                    return

        elif isinstance(msg_abstract, SimpleTextApiMessagePy):
            # 如果也需要处理 SimpleTextApiMessagePy，可以复制并添加到 active_messages_for_llm
            active_messages_for_llm.append(msg_abstract.model_copy(deep=True))
        else:
            logger.warning(f"{log_prefix}: Unknown message type in gemini_chat_input.messages: {type(msg_abstract)}")


    if extracted_document_text:
        logger.info(f"{log_prefix}: (Gemini REST) Integrating extracted document text (length: {len(extracted_document_text)}).")
        doc_text_part = PyTextContentPart(type="text_content", text=extracted_document_text)
        
        last_user_message_index = -1
        # 查找最后一个 PartsApiMessagePy 类型的用户消息来追加文本
        for i in range(len(active_messages_for_llm) - 1, -1, -1):
            if active_messages_for_llm[i].role == "user" and isinstance(active_messages_for_llm[i], PartsApiMessagePy):
                last_user_message_index = i
                break
        
        if last_user_message_index != -1:
            # 确保 active_messages_for_llm[last_user_message_index] 是 PartsApiMessagePy
            # 并且其 parts 属性是可变的列表
            cast_msg = active_messages_for_llm[last_user_message_index]
            if isinstance(cast_msg, PartsApiMessagePy):
                logger.debug(f"{log_prefix}: (Gemini REST) Appending extracted document text part to existing last user message parts.")
                # Pydantic 模型字段通常是不可变的，除非显式复制或重新创建。
                # 为安全起见，我们创建一个新的 PartsApiMessagePy 实例或修改其 parts 列表（如果允许）
                # 这里假设 PartsApiMessagePy 的 parts 字段是 List，可以直接 append
                # 但更安全的方式是创建一个新的消息或新的parts列表
                current_parts = list(cast_msg.parts) # 创建副本
                current_parts.append(doc_text_part)
                active_messages_for_llm[last_user_message_index] = PartsApiMessagePy(
                    role=cast_msg.role,
                    parts=current_parts,
                    message_type="parts_message", # 显式提供
                    name=cast_msg.name,
                    tool_calls=cast_msg.tool_calls,
                    tool_call_id=cast_msg.tool_call_id
                )
            else: # 如果最后一个用户消息不是Parts类型，则创建一个新的
                logger.info(f"{log_prefix}: (Gemini REST) Last user message is not PartsApiMessagePy or no PartsApiMessagePy user message found, creating new user message with document text.")
                new_user_message_with_doc = PartsApiMessagePy(
                    role="user",
                    parts=[
                        PyTextContentPart(type="text_content", text="请基于以下文档内容进行处理或回答："),
                        doc_text_part
                    ],
                    message_type="parts_message" # 显式提供
                )
                active_messages_for_llm.append(new_user_message_with_doc)

        else: # 没有找到任何用户消息
            logger.info(f"{log_prefix}: (Gemini REST) No prior user message found, creating new user message with extracted document text.")
            new_user_message_with_doc = PartsApiMessagePy(
                role="user",
                parts=[
                    PyTextContentPart(type="text_content", text="请基于以下文档内容进行处理或回答："),
                    doc_text_part
                ],
                message_type="parts_message" # 显式提供
            )
            active_messages_for_llm.append(new_user_message_with_doc)
        original_user_text_found_in_parts = True # 因为我们添加了文档

    # ... (后续的 web_search 和 API 调用逻辑保持不变) ...
    # 例如:
    user_query_for_search_gemini = ""
    search_results_generated_this_time = False
    if active_messages_for_llm:
        last_user_message_for_search = next((msg for msg in reversed(active_messages_for_llm) if msg.role == "user"), None)
        if last_user_message_for_search and isinstance(last_user_message_for_search, PartsApiMessagePy): # 确保是Parts类型
            for part in last_user_message_for_search.parts:
                if isinstance(part, PyTextContentPart) and part.text: 
                    user_query_for_search_gemini += part.text.strip() + " "
            user_query_for_search_gemini = user_query_for_search_gemini.strip()
        elif last_user_message_for_search and isinstance(last_user_message_for_search, SimpleTextApiMessagePy):
             user_query_for_search_gemini = last_user_message_for_search.content.strip()
    
    if gemini_chat_input.use_web_search and user_query_for_search_gemini:
        yield await sse_event_serializer_rest(AppStreamEventPy(type="status_update", stage="web_search_started", timestamp=get_current_time_iso()))
        search_results_list = await perform_web_search(user_query_for_search_gemini, request_id)
        if search_results_list:
            search_context_content = generate_search_context_message_content(user_query_for_search_gemini, search_results_list)
            search_context_parts = [PyTextContentPart(type="text_content", text=search_context_content)]
            
            # --- DEBUGGING before search_context_api_message instantiation ---
            logger.info(f"{log_prefix}: DEBUG (Search): Instantiating PartsApiMessagePy for search context with role='user', message_type='parts_message'")
            # --- END DEBUGGING ---
            try:
                search_context_api_message = PartsApiMessagePy(
                    role="user", 
                    parts=search_context_parts,
                    message_type="parts_message" # 显式提供
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
                logger.error(f"{log_prefix}: FAILED to instantiate PartsApiMessagePy (search_context_api_message). Error: {e_instantiate_search}", exc_info=True)
                yield await sse_event_serializer_rest(AppStreamEventPy(type="error", message=f"Internal error creating search context message: {e_instantiate_search}", timestamp=get_current_time_iso()))
                # 可能需要决定是否继续
        else:
            yield await sse_event_serializer_rest(AppStreamEventPy(type="status_update", stage="web_search_complete_no_results", query=user_query_for_search_gemini, timestamp=get_current_time_iso()))

    web_analysis_complete_sent = not (gemini_chat_input.use_web_search and user_query_for_search_gemini)

    try:
        if not gemini_chat_input.api_key:
            yield await sse_event_serializer_rest(AppStreamEventPy(type="error", message="Gemini API Key未在请求中提供。", timestamp=get_current_time_iso()))
            final_finish_event_sent = True; yield await sse_event_serializer_rest(AppStreamEventPy(type="finish", reason="configuration_error", timestamp=get_current_time_iso())); return
        
        # 创建一个新的 ChatRequestModel 副本用于 prepare_gemini_rest_api_request
        # 因为 prepare_gemini_rest_api_request 可能期望 messages 是特定类型的列表
        temp_chat_input_for_prepare = gemini_chat_input.model_copy(deep=True)
        # 这里需要确保 active_messages_for_llm 中的元素是 prepare_gemini_rest_api_request 期望的类型
        # 假设 prepare_gemini_rest_api_request 也能处理 AbstractApiMessagePy 列表
        temp_chat_input_for_prepare.messages = active_messages_for_llm 

        try:
            target_url, headers, json_payload = prepare_gemini_rest_api_request(chat_input=temp_chat_input_for_prepare, request_id=request_id)
        except Exception as e_prepare:
            logger.error(f"{log_prefix}: (Gemini REST) Request preparation error: {e_prepare}", exc_info=True)
            yield await sse_event_serializer_rest(AppStreamEventPy(type="error", message=f"请求准备错误: {e_prepare}", timestamp=get_current_time_iso()))
            final_finish_event_sent = True; yield await sse_event_serializer_rest(AppStreamEventPy(type="finish", reason="request_error", timestamp=get_current_time_iso())); return

        if not json_payload.get("contents"):
            # 检查 active_messages_for_llm 是否真的没有用户输入
            has_any_user_input_in_active = False
            for msg_chk in active_messages_for_llm:
                if msg_chk.role == "user":
                    if isinstance(msg_chk, PartsApiMessagePy) and any(isinstance(p, PyTextContentPart) and p.text and p.text.strip() for p in msg_chk.parts):
                        has_any_user_input_in_active = True; break
                    elif isinstance(msg_chk, SimpleTextApiMessagePy) and msg_chk.content and msg_chk.content.strip():
                        has_any_user_input_in_active = True; break
            
            if not has_any_user_input_in_active:
                 logger.warning(f"{log_prefix}: (Gemini REST) No valid user content to send to Gemini model after processing.")
                 yield await sse_event_serializer_rest(AppStreamEventPy(type="error", message="没有有效内容发送给Gemini模型。", timestamp=get_current_time_iso()))
                 final_finish_event_sent = True; yield await sse_event_serializer_rest(AppStreamEventPy(type="finish", reason="no_content_error", timestamp=get_current_time_iso())); return
            else: 
                logger.error(f"{log_prefix}: (Gemini REST) Contents are empty in json_payload despite having user/document text in active_messages_for_llm. This is unexpected but proceeding. Active messages: {[(m.role, m.message_type) for m in active_messages_for_llm]}")
        
        logger.info(f"{log_prefix}: (Gemini REST) Sending request to URL: {target_url.split('?key=')[0]}...") 
        # 改进日志可读性
        contents_preview = []
        for c_idx, c in enumerate(json_payload.get('contents', [])):
            parts_preview_list = []
            for p_idx, p in enumerate(c.get('parts', [])):
                if 'text' in p:
                    part_text_preview = p['text'][:50] + '...' if len(p['text']) > 50 else p['text']
                    parts_preview_list.append(f"Part[{p_idx}]: Text='{part_text_preview}'")
                elif 'inlineData' in p:
                     parts_preview_list.append(f"Part[{p_idx}]: InlineData MIME='{p.get('inlineData',{}).get('mimeType')}'")
                else:
                    parts_preview_list.append(f"Part[{p_idx}]: NonTextOrInlinePart")
            contents_preview.append(f"Content[{c_idx}]: Role='{c.get('role')}', Parts={parts_preview_list}")
        logger.debug(f"{log_prefix}: (Gemini REST) Payload contents preview: {contents_preview}")
        
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
                    logger.debug(f"{log_prefix}: (Gemini REST) Failed to parse error body as JSON: {err_text[:200]}")
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
                    # logger.debug(f"{log_prefix}: (Gemini REST) Raw SSE Data: {sse_data_bytes!r}") # Potentially very verbose
                    try:
                        chunk_json = orjson.loads(sse_data_bytes)
                        # logger.debug(f"{log_prefix}: (Gemini REST) Parsed SSE Chunk JSON: {str(chunk_json)[:300]}") # Potentially very verbose
                        if "candidates" in chunk_json and chunk_json["candidates"]:
                            for candidate in chunk_json["candidates"]:
                                if "content" in candidate and "parts" in candidate["content"]:
                                    for part_data in candidate["content"]["parts"]:
                                        part_text = part_data.get("text")
                                        is_thought = part_data.get("thought") is True # Check if 'thought' key exists and is true
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
            
            # Stream finished
            if await fastapi_request_obj.is_disconnected():
                logger.info(f"{log_prefix}: (Gemini REST) Client disconnected after stream completion.")
            elif not final_finish_event_sent: # If not already sent by a finish_reason in a candidate
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
            logger.info(f"{log_prefix}: Deleting {len(temp_files_to_delete_after_stream)} temporary document file(s) for Gemini REST path.")
            for temp_file in temp_files_to_delete_after_stream:
                try:
                    if os.path.exists(temp_file): 
                        os.remove(temp_file)
                        # logger.debug(f"{log_prefix}: Deleted temp file {temp_file}")
                except Exception as e_del: 
                    logger.error(f"{log_prefix}: Error deleting temp file {temp_file}: {e_del}")
        else:
            logger.debug(f"{log_prefix}: No temporary files to delete for this request.")

async def handle_gemini_request_entry(
    gemini_chat_input: ChatRequestModel,
    raw_request: Request,
    http_client: httpx.AsyncClient,
    request_id: str
):
    # This entry point is problematic if it doesn't handle document extraction and temp file management.
    # It should ideally be integrated into the main chat router logic that does handle these.
    # For now, assuming it's called in a context where document processing is not needed or done elsewhere.
    logger.warning(f"RID-{request_id}: handle_gemini_request_entry was called. This entry point assumes document text is already integrated or not applicable, and it does not manage temporary files. This might lead to issues if documents were expected.")
    
    # Directly pass None for extracted_document_text and an empty list for temp_files_to_delete
    # This makes the function signature consistent but highlights that this path doesn't process docs.
    return StreamingResponse(
        generate_gemini_rest_api_events_with_docs(
             gemini_chat_input=gemini_chat_input,
             fastapi_request_obj=raw_request,
             http_client=http_client,
             request_id=request_id,
             extracted_document_text=None, # Explicitly None
             temp_files_to_delete_after_stream=[] # Explicitly empty
        ),
        media_type="text/event-stream",
        headers=COMMON_HEADERS # Ensure COMMON_HEADERS is defined in eztalk_proxy.config
    )