# eztalk_proxy/routers/multimodal_chat.py
import os
import logging
import httpx
import orjson
import asyncio # 确保导入 asyncio
from typing import Optional, Dict, Any, AsyncGenerator, List

from fastapi import APIRouter, Depends, Request, HTTPException
from fastapi.responses import StreamingResponse

from eztalk_proxy.models import ChatRequestModel, AppStreamEventPy, PartsApiMessagePy
from eztalk_proxy.multimodal_models import PyTextContentPart

from eztalk_proxy.config import COMMON_HEADERS
from eztalk_proxy.utils import (
    get_current_time_iso,
    orjson_dumps_bytes_wrapper,
    strip_potentially_harmful_html_and_normalize_newlines,
    extract_sse_lines
)
from eztalk_proxy.multimodal_api_helpers import (
    prepare_gemini_rest_api_request
)
# 导入 Web 搜索相关函数
from eztalk_proxy.web_search import perform_web_search, generate_search_context_message_content

logger = logging.getLogger("EzTalkProxy.Routers.MultimodalChat")

async def sse_event_serializer_rest(event_data: AppStreamEventPy) -> bytes:
    return orjson_dumps_bytes_wrapper(event_data.model_dump(by_alias=True, exclude_none=True))

async def generate_gemini_rest_api_events(
    gemini_chat_input: ChatRequestModel,
    fastapi_request_obj: Request,
    http_client: httpx.AsyncClient,
    request_id: str
) -> AsyncGenerator[bytes, None]:
    log_prefix = f"RID-{request_id}"
    first_chunk_received_from_llm = False
    final_finish_event_sent = False
    _had_any_reasoning_event_sent_in_stream = False
    _reasoning_finish_event_sent_flag = False

    # 创建一个可修改的消息列表副本
    # gemini_chat_input.messages 包含的是 PartsApiMessagePy 对象
    active_messages_for_llm: List[PartsApiMessagePy] = [
        msg for msg in gemini_chat_input.messages if isinstance(msg, PartsApiMessagePy)
    ]
    
    user_query_for_search_gemini = ""
    search_results_generated_this_time = False

    # 提取用户查询 (从 PartsApiMessagePy)
    if active_messages_for_llm:
        last_user_message = next((msg for msg in reversed(active_messages_for_llm) if msg.role == "user"), None)
        if last_user_message: # last_user_message 已经是 PartsApiMessagePy
            for part in last_user_message.parts:
                if isinstance(part, PyTextContentPart) and part.text:
                    user_query_for_search_gemini += part.text.strip() + " "
            user_query_for_search_gemini = user_query_for_search_gemini.strip()

    # --- Web Search (如果启用) ---
    if gemini_chat_input.use_web_search and user_query_for_search_gemini:
        logger.info(f"{log_prefix}: (Gemini REST) Web search initiated for query: '{user_query_for_search_gemini[:100]}'")
        yield await sse_event_serializer_rest(AppStreamEventPy(type="status_update", stage="web_search_started", timestamp=get_current_time_iso()))
        
        search_results_list = await perform_web_search(user_query_for_search_gemini, request_id)
        
        if search_results_list:
            search_context_content = generate_search_context_message_content(user_query_for_search_gemini, search_results_list)
            search_context_parts = [PyTextContentPart(type="text_content", text=search_context_content)]
            # 重要: PartsApiMessagePy 需要 message_type (type)
            search_context_api_message = PartsApiMessagePy(
                role="user", 
                parts=search_context_parts,
                type="parts_message" # Discriminator field
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
            yield await sse_event_serializer_rest(AppStreamEventPy(type="web_search_results", results=[r.model_dump() for r in search_results_list], timestamp=get_current_time_iso()))
        else:
            logger.info(f"{log_prefix}: (Gemini REST) Web search yielded no results for query '{user_query_for_search_gemini[:100]}'.")
            yield await sse_event_serializer_rest(AppStreamEventPy(type="status_update", stage="web_search_complete_no_results", query=user_query_for_search_gemini, timestamp=get_current_time_iso()))

    web_analysis_complete_sent = not (gemini_chat_input.use_web_search and user_query_for_search_gemini)

    try:
        if not gemini_chat_input.api_key:
            # ... (错误处理) ...
            logger.error(f"{log_prefix}: (Gemini REST) API key is missing.")
            yield await sse_event_serializer_rest(AppStreamEventPy(type="error", message="Gemini API Key未在请求中提供。", timestamp=get_current_time_iso()))
            final_finish_event_sent = True; yield await sse_event_serializer_rest(AppStreamEventPy(type="finish", reason="configuration_error", timestamp=get_current_time_iso())); return

        # 为 prepare_gemini_rest_api_request 创建一个临时的 ChatRequestModel 实例或修改副本
        # 这里我们直接修改传入的 gemini_chat_input 的 messages 字段的引用，但要小心副作用
        # 一个更安全的方式是克隆 gemini_chat_input 或让 prepare 函数接受消息列表
        original_messages_ref = gemini_chat_input.messages
        gemini_chat_input.messages = active_messages_for_llm # 使用可能被Web搜索修改过的消息列表

        try:
            target_url, headers, json_payload = prepare_gemini_rest_api_request(
                chat_input=gemini_chat_input, 
                request_id=request_id
            )
        except Exception as e_prepare:
            # ... (错误处理) ...
            logger.error(f"{log_prefix}: (Gemini REST) Error preparing request: {e_prepare}", exc_info=True)
            yield await sse_event_serializer_rest(AppStreamEventPy(type="error", message=f"请求准备错误: {e_prepare}", timestamp=get_current_time_iso()))
            final_finish_event_sent = True; yield await sse_event_serializer_rest(AppStreamEventPy(type="finish", reason="request_error", timestamp=get_current_time_iso())); return
        finally:
            gemini_chat_input.messages = original_messages_ref # 恢复原始消息引用

        if not json_payload.get("contents"): # contents 是 Gemini REST API 的字段
            # ... (错误处理) ...
            logger.warning(f"{log_prefix}: (Gemini REST) No valid contents for model {gemini_chat_input.model}")
            yield await sse_event_serializer_rest(AppStreamEventPy(type="error", message="没有有效内容发送给Gemini模型。", timestamp=get_current_time_iso()))
            final_finish_event_sent = True; yield await sse_event_serializer_rest(AppStreamEventPy(type="finish", reason="no_content_error", timestamp=get_current_time_iso())); return

        logger.info(f"{log_prefix}: (Gemini REST) Sending request to URL: {target_url.split('?key=')[0]}...") 
        logger.debug(f"{log_prefix}: (Gemini REST) Payload: {orjson.dumps(json_payload).decode('utf-8', errors='ignore')[:1000]}...") 

        buffer = bytearray()
        async with http_client.stream("POST", target_url, headers=headers, json=json_payload, timeout=300.0) as response:
            # ... (后续的 SSE 处理逻辑保持不变) ...
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
                    # ... (SSE line processing) ...
                    if not sse_line_bytes.strip(): continue
                    sse_data_bytes = b""
                    if sse_line_bytes.startswith(b"data: "):
                        sse_data_bytes = sse_line_bytes[len(b"data: "):].strip()
                    if not sse_data_bytes: continue
                    
                    logger.debug(f"{log_prefix}: (Gemini REST) Raw SSE Data: {sse_data_bytes!r}")

                    try:
                        chunk_json = orjson.loads(sse_data_bytes)
                        logger.debug(f"{log_prefix}: (Gemini REST) Parsed SSE Chunk JSON: {chunk_json}")

                        if "candidates" in chunk_json and chunk_json["candidates"]:
                            for candidate in chunk_json["candidates"]:
                                if "content" in candidate and "parts" in candidate["content"]:
                                    for part_data in candidate["content"]["parts"]: # Renamed to part_data
                                        part_text = part_data.get("text")
                                        is_thought = part_data.get("thought") is True 

                                        if part_text:
                                            clean_text = strip_potentially_harmful_html_and_normalize_newlines(part_text)
                                            if not clean_text: continue

                                            if is_thought:
                                                logger.debug(f"{log_prefix}: (Gemini REST Thought) '{clean_text[:100]}'")
                                                yield await sse_event_serializer_rest(AppStreamEventPy(type="reasoning", text=clean_text, timestamp=get_current_time_iso()))
                                                _had_any_reasoning_event_sent_in_stream = True
                                            else:
                                                if _had_any_reasoning_event_sent_in_stream and not _reasoning_finish_event_sent_flag:
                                                   logger.debug(f"{log_prefix}: (Gemini REST) Sending reasoning_finish as content starts after reasoning.")
                                                   yield await sse_event_serializer_rest(AppStreamEventPy(type="reasoning_finish", timestamp=get_current_time_iso()))
                                                   _reasoning_finish_event_sent_flag = True
                                                logger.debug(f"{log_prefix}: (Gemini REST Content) '{clean_text[:100]}'")
                                                yield await sse_event_serializer_rest(AppStreamEventPy(type="content", text=clean_text, timestamp=get_current_time_iso()))
                                
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
                # ... (处理流结束但LLM未发送finish_reason) ...
                logger.info(f"{log_prefix}: (Gemini REST) Stream iterator finished without explicit finish_reason from last chunk.")
                if _had_any_reasoning_event_sent_in_stream and not _reasoning_finish_event_sent_flag:
                   logger.debug(f"{log_prefix}: (Gemini REST) Sending reasoning_finish as stream ends.")
                   yield await sse_event_serializer_rest(AppStreamEventPy(type="reasoning_finish", timestamp=get_current_time_iso()))
                   _reasoning_finish_event_sent_flag = True
                final_finish_event_sent = True
                yield await sse_event_serializer_rest(AppStreamEventPy(type="finish", reason="stream_end", timestamp=get_current_time_iso()))


    except httpx.RequestError as e_req: 
        # ... (处理网络错误) ...
        logger.error(f"{log_prefix}: (Gemini REST) httpx.RequestError for model '{gemini_chat_input.model}': {e_req}", exc_info=True)
        yield await sse_event_serializer_rest(AppStreamEventPy(type="error", message=f"网络请求错误: {e_req}", timestamp=get_current_time_iso()))
        if not final_finish_event_sent: final_finish_event_sent = True; yield await sse_event_serializer_rest(AppStreamEventPy(type="finish", reason="network_error", timestamp=get_current_time_iso()))
    except Exception as e_gen:
        # ... (处理通用错误) ...
        logger.error(f"{log_prefix}: (Gemini REST) Generic error for model '{gemini_chat_input.model}': {e_gen}", exc_info=True)
        yield await sse_event_serializer_rest(AppStreamEventPy(type="error", message=f"处理Gemini REST请求时发生未知错误: {str(e_gen)}", timestamp=get_current_time_iso()))
        if not final_finish_event_sent: final_finish_event_sent = True; yield await sse_event_serializer_rest(AppStreamEventPy(type="finish", reason="unknown_error", timestamp=get_current_time_iso()))
    finally:
        # ... (清理逻辑) ...
        logger.info(f"{log_prefix}: (Gemini REST) Stream generation loop for model {gemini_chat_input.model} concluded.")
        if not final_finish_event_sent:
            logger.warning(f"{log_prefix}: (Gemini REST) Forcing final 'finish' event in finally block.")
            if _had_any_reasoning_event_sent_in_stream and not _reasoning_finish_event_sent_flag:
                 logger.debug(f"{log_prefix}: (Gemini REST) Sending reasoning_finish in finally block.")
                 yield await sse_event_serializer_rest(AppStreamEventPy(type="reasoning_finish", timestamp=get_current_time_iso()))
            yield await sse_event_serializer_rest(AppStreamEventPy(type="finish", reason="cleanup_stream_end_gemini_rest", timestamp=get_current_time_iso()))


async def handle_gemini_request_entry(
    gemini_chat_input: ChatRequestModel,
    raw_request: Request,
    http_client: httpx.AsyncClient,
    request_id: str
):
    log_prefix = f"RID-{request_id}"
    logger.info(f"{log_prefix}: Handling Google Gemini request (via REST API) for model {gemini_chat_input.model}.")
    return StreamingResponse(
        generate_gemini_rest_api_events(gemini_chat_input, raw_request, http_client, request_id),
        media_type="text/event-stream",
        headers=COMMON_HEADERS
    )