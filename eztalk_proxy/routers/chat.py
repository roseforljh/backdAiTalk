import os
import logging
import httpx
import orjson
from typing import Optional, Dict, Any, AsyncGenerator, List, Union # Added Union

from fastapi import APIRouter, Depends, Request
from fastapi.responses import StreamingResponse

# 从您的模型文件中导入更新后的模型
# 确保 TextContentIn, MultipartContentIn, ApiContentPart 也在 models.py 中定义
from ..models import ChatRequest, ApiMessage, TextContentIn, MultipartContentIn, ApiContentPart
from ..config import (
    GOOGLE_API_BASE_URL, COMMON_HEADERS
)
from ..utils import (
    extract_sse_lines, get_current_time_iso,
    orjson_dumps_bytes_wrapper, strip_potentially_harmful_html_and_normalize_newlines
)
from ..api_helpers import prepare_openai_request, prepare_google_request_payload_structure
from ..web_search import perform_web_search, generate_search_context_message_content
from ..stream_processors import (
    should_apply_custom_separator_logic, process_openai_response, process_google_response,
    handle_stream_error, handle_stream_cleanup
)

logger = logging.getLogger("EzTalkProxy.Routers.Chat")
router = APIRouter()

async def get_http_client(request: Request) -> Optional[httpx.AsyncClient]:
    return getattr(request.app.state, "http_client", None)

# 新增辅助函数：从新的 content 结构中提取纯文本
# 您也可以考虑将此函数移动到 utils.py 中
def extract_text_from_content(content: Union[TextContentIn, MultipartContentIn, str, None]) -> str:
    """
    Extracts plain text from the potentially complex content field of an ApiMessage.
    """
    if isinstance(content, TextContentIn):
        return content.text.strip() if content.text else ""
    elif isinstance(content, MultipartContentIn):
        text_parts = []
        for part in content.parts:
            if part.type == "text" and part.text:
                text_parts.append(part.text.strip())
        return " ".join(text_parts).strip()
    elif isinstance(content, str):
        return content.strip()
    return ""

@router.post("/chat", response_class=StreamingResponse, summary="AI聊天完成代理", tags=["AI Proxy"])
async def chat_proxy(
    request_data: ChatRequest, # FastAPI 会使用更新后的 ChatRequest 模型进行解析
    client: Optional[httpx.AsyncClient] = Depends(get_http_client)
):
    request_id = os.urandom(8).hex()
    logger.info(
        f"RID-{request_id}: Received /chat request: Provider='{request_data.provider}', "
        f"Model='{request_data.model}', WebSearch={request_data.use_web_search}, "
        # VVVVVV 字段名修正 VVVVVV
        f"ForceGoogleReasoning={request_data.force_google_reasoning_prompt}, "
        # ^^^^^^ 字段名修正 ^^^^^^
        f"CustomParams={request_data.custom_model_parameters is not None}"
    )

    if not client:
        logger.warning(f"RID-{request_id}: HTTP client not available from app.state for request {request_id}.")
        async def client_error_gen():
            yield orjson_dumps_bytes_wrapper({"type": "error", "message": "Service unavailable: HTTP client not initialized.", "timestamp": get_current_time_iso()})
            yield orjson_dumps_bytes_wrapper({"type": "finish", "reason": "service_unavailable", "timestamp": get_current_time_iso()})
        return StreamingResponse(client_error_gen(), media_type="text/event-stream", headers=COMMON_HEADERS)

    # api_messages_for_processing 已经包含了正确解析后的 ApiMessage 对象
    # 其中每个 message.content 可能是 TextContentIn, MultipartContentIn, str, 或 None
    api_messages_for_processing: List[ApiMessage] = [
        m.model_copy(deep=True) for m in request_data.messages
        if m.content is not None or m.tool_calls is not None or m.role == "system" # content is not None 仍然适用
    ]

    if not any(m.role != "system" for m in api_messages_for_processing):
        if not any(m.role == "system" and m.content for m in api_messages_for_processing): # 这里的 m.content 可能是复杂类型，但只要非 None 即可
            async def no_message_error_gen():
                yield orjson_dumps_bytes_wrapper({"type": "error", "message": "No processable messages provided (excluding empty system messages).", "timestamp": get_current_time_iso()})
                yield orjson_dumps_bytes_wrapper({"type": "finish", "reason": "bad_request", "timestamp": get_current_time_iso()})
            return StreamingResponse(no_message_error_gen(), media_type="text/event-stream", headers=COMMON_HEADERS)

    user_query_for_search = ""
    search_results_generated = False
    # ... (其他标志位定义保持不变)
    is_native_thinking_mode_active = False
    use_google_sse_parser_flag = False
    is_google_payload_format_used_flag = False
    is_google_like_path_active = False


    # VVVVVV 修改提取 user_query_for_search 的逻辑 VVVVVV
    if request_data.use_web_search:
        for msg_obj in reversed(api_messages_for_processing): # msg_obj 是 ApiMessage 类型
            if msg_obj.role == "user":
                # 使用新的辅助函数提取文本
                extracted_text = extract_text_from_content(msg_obj.content)
                if extracted_text: # 确保提取到非空文本
                    user_query_for_search = extracted_text
                    logger.info(f"RID-{request_id}: Extracted user query for web search: '{user_query_for_search[:100]}'")
                    break
        if not user_query_for_search:
            logger.warning(f"RID-{request_id}: Web search enabled but no processable user query found in messages.")
    # ^^^^^^ 修改提取 user_query_for_search 的逻辑结束 ^^^^^^

    if request_data.provider not in ["google", "openai"]:
        async def provider_error_gen():
            yield orjson_dumps_bytes_wrapper({"type": "error", "message": f"Unsupported provider: {request_data.provider}", "timestamp": get_current_time_iso()})
            yield orjson_dumps_bytes_wrapper({"type": "finish", "reason": "bad_request", "timestamp": get_current_time_iso()})
        return StreamingResponse(provider_error_gen(), media_type="text/event-stream", headers=COMMON_HEADERS)
    elif request_data.provider == "google":
        logger.info(f"RID-{request_id}: Path: Google Direct. Model: {request_data.model}")
        is_google_payload_format_used_flag = True
        use_google_sse_parser_flag = True
        is_google_like_path_active = True
    elif request_data.provider == "openai":
        logger.info(f"RID-{request_id}: Path: OpenAI Provider ('{request_data.model}').")

    async def stream_generator() -> AsyncGenerator[bytes, None]:
        nonlocal api_messages_for_processing, search_results_generated, user_query_for_search
        nonlocal is_native_thinking_mode_active, use_google_sse_parser_flag
        nonlocal is_google_payload_format_used_flag, is_google_like_path_active

        if not client:
            logger.error(f"RID-{request_id}: CRITICAL - HTTP client is None within stream_generator scope.")
            yield orjson_dumps_bytes_wrapper({"type": "error", "message": "Internal error: HTTP client unavailable in stream.", "timestamp": get_current_time_iso()})
            yield orjson_dumps_bytes_wrapper({"type": "finish", "reason": "internal_error", "timestamp": get_current_time_iso()})
            return
        
        try:
            if request_data.use_web_search and user_query_for_search:
                yield orjson_dumps_bytes_wrapper({"type": "status_update", "stage": "web_search_started", "timestamp": get_current_time_iso()})
                search_results_list = await perform_web_search(user_query_for_search, request_id)
                if search_results_list:
                    # generate_search_context_message_content 应该返回一个字符串
                    search_context_content_str = generate_search_context_message_content(user_query_for_search, search_results_list)
                    # 系统消息的 content 通常是字符串，我们的 ApiMessage 模型也允许 content 为 str
                    new_system_message = ApiMessage(role="system", content=search_context_content_str)
                    # ... (注入系统消息的逻辑保持不变)
                    last_user_message_index = -1
                    for i, msg in reversed(list(enumerate(api_messages_for_processing))):
                        if msg.role == "user":
                            last_user_message_index = i
                            break
                    if last_user_message_index != -1:
                        api_messages_for_processing.insert(last_user_message_index, new_system_message)
                    else:
                        api_messages_for_processing.append(new_system_message)
                    search_results_generated = True
                    logger.info(f"RID-{request_id}: Web search context injected into messages.")
                    yield orjson_dumps_bytes_wrapper({"type": "status_update", "stage": "web_search_complete_with_results", "query": user_query_for_search, "timestamp": get_current_time_iso()})
                else:
                    logger.info(f"RID-{request_id}: Web search yielded no results for query '{user_query_for_search[:100]}'.")
                    yield orjson_dumps_bytes_wrapper({"type": "status_update", "stage": "web_search_complete_no_results", "query": user_query_for_search, "timestamp": get_current_time_iso()})
                yield orjson_dumps_bytes_wrapper({"type": "status_update", "stage": "web_analysis_started", "timestamp": get_current_time_iso()})

            current_api_url: str; current_api_headers: Dict[str,str]; current_api_payload: Dict[str,Any]; current_api_params: Optional[Dict[str,str]] = None

            # ========================== 重要提示 ==========================
            # 下面的 prepare_google_request_payload_structure 和 prepare_openai_request 函数
            # (可能在您的 api_helpers.py 文件中) 现在接收到的 api_messages_for_processing
            # 列表中的每个 ApiMessage 对象的 .content 属性可能是 TextContentIn, MultipartContentIn, str, 或 None。
            # 您必须修改这些辅助函数，使其能够：
            # 1. 检查 message.content 的类型。
            # 2. 如果是 TextContentIn，则使用 message.content.text 作为文本内容。
            # 3. 如果是 MultipartContentIn，则根据目标API（Google 或 OpenAI）的要求，
            #    将其中的 parts (特别是 text 和 image_url 类型) 转换成相应的多部件/多模态格式。
            #    例如，OpenAI 的 vision 模型期望 content 是一个包含特定结构对象的列表。
            # 4. 如果是 str，则直接使用。
            # ============================================================

            if request_data.provider == "google":
                payload_dict, native_thinking_flag = prepare_google_request_payload_structure(
                    request_data, api_messages_for_processing, request_id
                )
                # ... (后续逻辑不变)
                is_native_thinking_mode_active = native_thinking_flag
                current_api_payload = payload_dict
                current_api_url = f"{GOOGLE_API_BASE_URL}/v1beta/models/{request_data.model}:streamGenerateContent"
                current_api_params = {"key": request_data.api_key, "alt": "sse"}
                current_api_headers = {"Content-Type": "application/json"}
            elif request_data.provider == "openai":
                current_api_url, current_api_headers, current_api_payload = prepare_openai_request(
                    request_data, api_messages_for_processing, request_id
                )
            
            # ... (stream_generator 的其余内容，包括 stream 处理、错误处理、finally 块等，保持不变) ...
            # ... (因为这部分代码主要处理流式响应的字节流，不直接依赖 content 的内部结构，) ...
            # ... (依赖的是 prepare_... 函数正确构造了 current_api_payload) ...

            use_old_custom_separator_branch_flag = should_apply_custom_separator_logic(
                request_data, request_id,
                is_google_like_path_active,
                is_native_thinking_mode_active
            )
            logger.info(f"RID-{request_id}: Final Logic Flags: GooglePayloadUsed={is_google_payload_format_used_flag}, ExpectGoogleResponseSSE={use_google_sse_parser_flag}, NativeThinkingActiveForGooglePath={is_native_thinking_mode_active}, UseOldSeparatorLogic={use_old_custom_separator_branch_flag}")
            
            # 调试日志：payload 打印时，如果 messages 里的 content 是复杂对象，str() 的输出可能不直观
            # 但这只是日志记录，不影响功能
            payload_messages_preview = []
            raw_payload_messages = current_api_payload.get('messages', current_api_payload.get('contents',[]))
            if isinstance(raw_payload_messages, list):
                for msg_idx, msg_content in enumerate(raw_payload_messages):
                    if msg_idx < 2: # Log first 2 messages for brevity
                         # This assumes prepare_... functions have already formatted content for the target LLM
                        payload_messages_preview.append(str(msg_content)[:200]) # Log string representation
                    else:
                        payload_messages_preview.append("...")
                        break
            else: # if not list, log as is
                payload_messages_preview.append(str(raw_payload_messages)[:500])

            logger.debug(f"RID-{request_id}: Sending to URL: {current_api_url}. Headers: {current_api_headers}. Payload preview (messages/contents): {' | '.join(payload_messages_preview)}")


            buffer = bytearray()
            upstream_ok = False
            first_chunk_llm = False
            state: Dict[str, Any] = {
                "accumulated_openai_content": "", "accumulated_openai_reasoning": "",
                "openai_had_any_reasoning": False, "openai_had_any_content_or_tool_call": False,
                "openai_reasoning_finish_event_sent": False,
                "accumulated_google_thought": "", "accumulated_google_text": "",
                "google_native_had_thoughts": False, "google_native_had_answer": False,
                "accumulated_text_custom": "", "full_yielded_reasoning_custom": "",
                "full_yielded_content_custom": "", "found_separator_custom": False,
            }

            async with client.stream("POST", current_api_url, headers=current_api_headers, json=current_api_payload, params=current_api_params) as resp:
                logger.info(f"RID-{request_id}: Upstream LLM response status: {resp.status_code}")
                if not (200 <= resp.status_code < 300):
                    err_body_bytes = await resp.aread()
                    err_text = err_body_bytes.decode("utf-8", errors="replace")
                    logger.error(f"RID-{request_id}: Upstream LLM error {resp.status_code}: {err_text[:1000]}")
                    try:
                        err_data = orjson.loads(err_text)
                        msg_detail = err_data.get("error", {}).get("message", str(err_data))
                    except:
                        msg_detail = err_text[:200]
                    yield orjson_dumps_bytes_wrapper({"type": "error", "message": f"LLM API Error: {msg_detail}", "upstream_status": resp.status_code, "timestamp": get_current_time_iso()})
                    yield orjson_dumps_bytes_wrapper({"type": "finish", "reason": "upstream_error", "timestamp": get_current_time_iso()})
                    return

                upstream_ok = True
                async for raw_chunk_bytes in resp.aiter_raw():
                    if not raw_chunk_bytes: continue
                    if not first_chunk_llm:
                        if request_data.use_web_search and user_query_for_search:
                            yield orjson_dumps_bytes_wrapper({"type": "status_update", "stage": "web_analysis_complete", "timestamp": get_current_time_iso()})
                        first_chunk_llm = True
                    buffer.extend(raw_chunk_bytes)
                    sse_lines, buffer = extract_sse_lines(buffer)

                    for sse_line_bytes in sse_lines:
                        if not sse_line_bytes.strip(): continue
                        sse_data_bytes = b""
                        if sse_line_bytes.startswith(b"data: "):
                            sse_data_bytes = sse_line_bytes[len(b"data: "):].strip()
                        if not sse_data_bytes: continue
                        logger.debug(f"RID-{request_id}, Raw SSE Data Line: {sse_data_bytes!r}")

                        if sse_data_bytes == b"[DONE]":
                            if not use_google_sse_parser_flag: # OpenAI or compatible
                                logger.info(f"RID-{request_id}: Received [DONE] from OpenAI-like endpoint.")
                                # Flush any remaining OpenAI accumulated data
                                if state.get("accumulated_openai_reasoning"):
                                    processed_reasoning = strip_potentially_harmful_html_and_normalize_newlines(state["accumulated_openai_reasoning"])
                                    if processed_reasoning: yield orjson_dumps_bytes_wrapper({"type": "reasoning", "text": processed_reasoning, "timestamp": get_current_time_iso()})
                                    state["accumulated_openai_reasoning"] = ""
                                if state.get("openai_had_any_reasoning") and not state.get("openai_reasoning_finish_event_sent"):
                                    yield orjson_dumps_bytes_wrapper({"type": "reasoning_finish", "timestamp": get_current_time_iso()})
                                    state["openai_reasoning_finish_event_sent"] = True
                                if state.get("accumulated_openai_content"):
                                    processed_content = strip_potentially_harmful_html_and_normalize_newlines(state["accumulated_openai_content"])
                                    if processed_content: yield orjson_dumps_bytes_wrapper({"type": "content", "text": processed_content, "timestamp": get_current_time_iso()})
                                    state["accumulated_openai_content"] = ""
                                yield orjson_dumps_bytes_wrapper({"type": "finish", "reason": "stop_openai_done", "timestamp": get_current_time_iso()})
                            else: # Google path, but received [DONE] - unexpected
                                logger.warning(f"RID-{request_id}: Received [DONE] but was expecting Google format SSE. Treating as end.")
                                # Flush any remaining Google accumulated data
                                if state.get("accumulated_google_thought"):
                                    processed_thought = strip_potentially_harmful_html_and_normalize_newlines(state["accumulated_google_thought"])
                                    if processed_thought: yield orjson_dumps_bytes_wrapper({"type": "reasoning", "text": processed_thought, "timestamp": get_current_time_iso()})
                                    state["accumulated_google_thought"] = ""
                                if is_native_thinking_mode_active and state.get('google_native_had_thoughts') and not state.get('openai_reasoning_finish_event_sent'): # using openai_reasoning_finish_event_sent as generic flag here
                                    yield orjson_dumps_bytes_wrapper({"type": "reasoning_finish", "timestamp": get_current_time_iso()})
                                    state["openai_reasoning_finish_event_sent"] = True
                                if state.get("accumulated_google_text"):
                                    processed_text = strip_potentially_harmful_html_and_normalize_newlines(state["accumulated_google_text"])
                                    if processed_text: yield orjson_dumps_bytes_wrapper({"type": "content", "text": processed_text, "timestamp": get_current_time_iso()})
                                    state["accumulated_google_text"] = ""
                                yield orjson_dumps_bytes_wrapper({"type": "finish", "reason": "google_stream_ended_with_unexpected_done_signal", "timestamp": get_current_time_iso()})
                            return # End processing after [DONE]

                        try:
                            parsed_sse_data = orjson.loads(sse_data_bytes)
                            logger.debug(f"RID-{request_id}, Parsed SSE Data JSON: {parsed_sse_data}")
                        except orjson.JSONDecodeError:
                            logger.warning(f"RID-{request_id}: Failed to parse SSE JSON data: {sse_data_bytes[:100]!r}"); continue

                        if use_google_sse_parser_flag:
                            async for event in process_google_response(parsed_sse_data, state, request_id, is_native_thinking_mode_active, use_old_custom_separator_branch_flag):
                                yield event
                                if event: # Check if event is not None before trying to parse
                                    try:
                                        event_data = orjson.loads(event)
                                        if event_data.get("type") == "finish": return
                                    except orjson.JSONDecodeError: pass # Ignore if parsing fails, it might be a partial event or non-JSON string
                        else: # OpenAI or compatible
                            async for event in process_openai_response(parsed_sse_data, state, request_id):
                                yield event
                                # No need to check for "finish" here as OpenAI uses [DONE] signal

        except Exception as e:
            async for event in handle_stream_error(e, request_id, upstream_ok, first_chunk_llm):
                yield event
        finally:
            if upstream_ok : # Only process final flushes if upstream connection was successful
                if not use_google_sse_parser_flag : # OpenAI or compatible
                    if state.get("accumulated_openai_reasoning"):
                        logger.info(f"RID-{request_id}: FINALLY flushing OpenAI reasoning: '{state['accumulated_openai_reasoning'][:100]}'")
                        processed_reasoning = strip_potentially_harmful_html_and_normalize_newlines(state["accumulated_openai_reasoning"])
                        if processed_reasoning: yield orjson_dumps_bytes_wrapper({"type": "reasoning", "text": processed_reasoning, "timestamp": get_current_time_iso()})
                    if state.get("openai_had_any_reasoning") and not state.get("openai_reasoning_finish_event_sent"):
                        logger.info(f"RID-{request_id}: FINALLY sending OpenAI reasoning_finish.")
                        yield orjson_dumps_bytes_wrapper({"type": "reasoning_finish", "timestamp": get_current_time_iso()})
                        state["openai_reasoning_finish_event_sent"] = True # Mark as sent
                    if state.get("accumulated_openai_content"):
                        logger.info(f"RID-{request_id}: FINALLY flushing OpenAI content: '{state['accumulated_openai_content'][:100]}'")
                        processed_content = strip_potentially_harmful_html_and_normalize_newlines(state["accumulated_openai_content"])
                        if processed_content: yield orjson_dumps_bytes_wrapper({"type": "content", "text": processed_content, "timestamp": get_current_time_iso()})
                elif use_google_sse_parser_flag: # Google path
                    if state.get("accumulated_google_thought"):
                        logger.info(f"RID-{request_id}: FINALLY flushing Google thought: '{state['accumulated_google_thought'][:100]}'")
                        processed_thought = strip_potentially_harmful_html_and_normalize_newlines(state["accumulated_google_thought"])
                        if processed_thought: yield orjson_dumps_bytes_wrapper({"type": "reasoning", "text": processed_thought, "timestamp": get_current_time_iso()})
                    if is_native_thinking_mode_active and state.get('google_native_had_thoughts') and not state.get('openai_reasoning_finish_event_sent'): # using openai_reasoning_finish_event_sent as generic flag
                        logger.info(f"RID-{request_id}: FINALLY sending Google reasoning_finish (native thoughts).")
                        yield orjson_dumps_bytes_wrapper({"type": "reasoning_finish", "timestamp": get_current_time_iso()})
                        state["openai_reasoning_finish_event_sent"] = True # Mark as sent
                    if state.get("accumulated_google_text"):
                        logger.info(f"RID-{request_id}: FINALLY flushing Google text: '{state['accumulated_google_text'][:100]}'")
                        processed_text = strip_potentially_harmful_html_and_normalize_newlines(state["accumulated_google_text"])
                        if processed_text: yield orjson_dumps_bytes_wrapper({"type": "content", "text": processed_text, "timestamp": get_current_time_iso()})

            # Ensure a finish event is always sent if not already done by specific conditions above
            # This state key needs to be set by the stream processors or [DONE] handling if stream finishes "naturally"
            # For now, we add a general cleanup based on state.
            state["_is_native_thinking_final_log"] = is_native_thinking_mode_active if is_google_like_path_active else False # Pass this info to cleanup
            async for event in handle_stream_cleanup(
                state, request_id, upstream_ok,
                use_old_custom_separator_branch_flag,
                request_data.provider # Pass provider to cleanup
            ):
                yield event

    return StreamingResponse(stream_generator(), media_type="text/event-stream", headers=COMMON_HEADERS)