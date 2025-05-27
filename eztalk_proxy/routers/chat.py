# eztalk_proxy/routers/chat.py
import os
import logging
import httpx
import orjson # 使用 orjson 进行更快的JSON操作
from typing import Optional, Dict, Any, AsyncGenerator, List

from fastapi import APIRouter, Depends, Request, HTTPException
from fastapi.responses import StreamingResponse

# 导入更新后的 Pydantic 模型
from ..models import ChatRequestModel, SimpleTextApiMessagePy, PartsApiMessagePy, AppStreamEventPy
from ..config import (
    # GOOGLE_API_BASE_URL, # 这个可能移到 multimodal_chat.py 或由 prepare_... 函数处理
    COMMON_HEADERS # 确保这个在 config.py 中有定义
)
from ..utils import (
    extract_sse_lines, get_current_time_iso,
    orjson_dumps_bytes_wrapper, # 确保 utils 中有这个函数
    strip_potentially_harmful_html_and_normalize_newlines
    # is_gemini_2_5_model # 这个判断现在直接在 chat_proxy 中进行
)
# 导入特定于 provider 的请求准备函数
from ..api_helpers import prepare_openai_request # 这个处理非Gemini的OpenAI请求
# prepare_google_request_payload_structure 这个函数可能需要重命名或调整，
# 因为Google的非Gemini模型（如果有）和Gemini模型的payload结构不同。
# 我们将在 multimodal_chat.py 中有专门的 prepare_google_multimodal_request

# 导入流处理器
from ..stream_processors import (
    should_apply_custom_separator_logic, # 这个逻辑可能需要重新评估
    process_openai_like_sse_stream, # 用于处理OpenAI兼容API的SSE流
    # process_google_response, # Google原生响应处理将移至 multimodal_chat.py
    handle_stream_error, handle_stream_cleanup
)

# 导入即将创建的 multimodal_chat 模块中的处理函数
from ..routers import multimodal_chat as multimodal_router # 别名，避免与当前模块名冲突

logger = logging.getLogger("EzTalkProxy.Routers.Chat")
router = APIRouter()

async def get_http_client(request: Request) -> Optional[httpx.AsyncClient]:
    # 从 app.state 获取全局 HTTP 客户端实例
    client = getattr(request.app.state, "http_client", None)
    if client is None:
        logger.error("HTTP client not found in app.state. Ensure lifespan event handler is configured correctly.")
        # 可以选择抛出异常或返回一个表示错误的特殊值，让调用者处理
    elif hasattr(client, 'is_closed') and client.is_closed:
        logger.error("HTTP client in app.state is closed. Application might be shutting down or an error occurred.")
        # 同上，处理错误
    return client

@router.post("/chat", response_class=StreamingResponse, summary="AI聊天完成代理", tags=["AI Proxy"])
async def chat_proxy(
    chat_input: ChatRequestModel, # 使用更新后的 ChatRequestModel
    fastapi_request_obj: Request, # 重命名以区分 client (httpx)
    http_client: Optional[httpx.AsyncClient] = Depends(get_http_client) # 依赖注入全局客户端
):
    request_id = os.urandom(8).hex() # 或者您有其他的 request_id 生成方式
    log_prefix = f"RID-{request_id}"

    logger.info(
        f"{log_prefix}: Received /chat request: Provider='{chat_input.provider}', "
        f"Model='{chat_input.model}', WebSearch={chat_input.use_web_search}"
    )

    if not http_client:
        logger.error(f"{log_prefix}: HTTP client not available. Service might be misconfigured or shutting down.")
        async def client_error_gen():
            yield orjson_dumps_bytes_wrapper(AppStreamEventPy(type="error", message="Service unavailable: HTTP client not initialized.", timestamp=get_current_time_iso()).model_dump())
            yield orjson_dumps_bytes_wrapper(AppStreamEventPy(type="finish", reason="service_unavailable", timestamp=get_current_time_iso()).model_dump())
        return StreamingResponse(client_error_gen(), media_type="text/event-stream", headers=COMMON_HEADERS)

    # --- 核心分发逻辑：判断是否为 Gemini 模型 ---
    if chat_input.model.lower().startswith("gemini"):
        logger.info(f"{log_prefix}: Model '{chat_input.model}' identified as Gemini. Dispatching to multimodal handler.")
        # 调用 multimodal_chat.py 中的处理函数
        # 这个函数需要与这里的 chat_proxy 有相似的签名，并返回 StreamingResponse
        return await multimodal_router.handle_gemini_request_entry( # 我们将在 multimodal_chat.py 中定义此函数
            gemini_chat_input=chat_input, # 参数名区分
            raw_request=fastapi_request_obj,
            http_client=http_client,
            request_id=request_id # 传递 request_id
        )
    else:
        # --- 非 Gemini 模型的处理逻辑 ---
        logger.info(f"{log_prefix}: Model '{chat_input.model}' is non-Gemini. Processing with standard handler.")

        # 1. 提取和准备非Gemini模型的上游消息 (期望 SimpleTextApiMessagePy)
        upstream_api_messages: List[Dict[str, Any]] = []
        user_query_for_search = "" # 用于联网搜索

        for i, msg_abstract in enumerate(chat_input.messages):
            if isinstance(msg_abstract, SimpleTextApiMessagePy):
                # 这是非Gemini模型期望的简单文本消息
                upstream_api_messages.append({"role": msg_abstract.role, "content": msg_abstract.content})
                if msg_abstract.role == "user" and msg_abstract.content: # 获取最后一个用户查询用于搜索
                    user_query_for_search = msg_abstract.content.strip()
                # 如果 SimpleTextApiMessagePy 也支持 tool_calls, 在这里处理
                if msg_abstract.tool_calls:
                     upstream_api_messages[-1]["tool_calls"] = [tc.model_dump(exclude_none=True) for tc in msg_abstract.tool_calls]
                if msg_abstract.role == "tool" and msg_abstract.tool_call_id:
                     upstream_api_messages[-1]["tool_call_id"] = msg_abstract.tool_call_id
                if msg_abstract.name : #  OpenAI 的 function role 现在是 tool role 带着 name
                     if msg_abstract.role == "tool" :
                         upstream_api_messages[-1]["name"] = msg_abstract.name

            elif isinstance(msg_abstract, PartsApiMessagePy):
                # 非Gemini路径不应该收到PartsApiMessage，如果收到，说明路由或前端逻辑可能还有问题
                # 或者，我们可以尝试从中提取纯文本部分作为降级处理
                logger.warning(f"{log_prefix}: Non-Gemini handler received a PartsApiMessage for model '{chat_input.model}'. Attempting to extract text.")
                text_from_parts = ""
                for part_wrapper in msg_abstract.parts:
                    if isinstance(part_wrapper, TextPartWrapper):
                        text_from_parts += part_wrapper.text_content.text + " "
                if text_from_parts.strip():
                    upstream_api_messages.append({"role": msg_abstract.role, "content": text_from_parts.strip()})
                    if msg_abstract.role == "user":
                        user_query_for_search = text_from_parts.strip()
                else:
                    logger.warning(f"{log_prefix}: Could not extract usable text from PartsApiMessage for non-Gemini model.")
            else:
                logger.error(f"{log_prefix}: Unknown message type in messages list: {type(msg_abstract)}")


        if not upstream_api_messages or not any(m.get("role") != "system" for m in upstream_api_messages):
             if not any(m.get("role") == "system" and m.get("content") for m in upstream_api_messages): #允许只有system prompt
                logger.warning(f"{log_prefix}: No processable non-system messages for non-Gemini model '{chat_input.model}'.")
                async def no_msg_gen():
                    yield orjson_dumps_bytes_wrapper(AppStreamEventPy(type="error", message="No processable messages.", timestamp=get_current_time_iso()).model_dump())
                    yield orjson_dumps_bytes_wrapper(AppStreamEventPy(type="finish", reason="bad_request", timestamp=get_current_time_iso()).model_dump())
                return StreamingResponse(no_msg_gen(), media_type="text/event-stream", headers=COMMON_HEADERS)

        # --- stream_generator 内部逻辑 (大部分与您提供的 chat.py 相似) ---
        async def stream_generator_non_gemini() -> AsyncGenerator[bytes, None]:
            nonlocal upstream_api_messages, user_query_for_search # 确保能修改这些变量
            
            # Web search logic (if enabled for non-Gemini)
            search_results_generated_flag = False # 本地flag，用于判断 web_analysis_complete 阶段
            if chat_input.use_web_search and user_query_for_search:
                yield orjson_dumps_bytes_wrapper(AppStreamEventPy(type="status_update", stage="web_search_started", timestamp=get_current_time_iso()).model_dump())
                # search_results_list = await perform_web_search(user_query_for_search, request_id) # 假设 perform_web_search 返回 List[WebSearchResultModel]
                # WebSearchResultModel 需要定义，或者直接返回 List[Dict]
                # ... (注入搜索结果到 upstream_api_messages 的逻辑，类似您之前的实现) ...
                # search_results_generated_flag = True
                # yield orjson_dumps_bytes_wrapper(AppStreamEventPy(type="web_search_results", results=..., timestamp=get_current_time_iso()).model_dump())
                # yield orjson_dumps_bytes_wrapper(AppStreamEventPy(type="status_update", stage="web_analysis_started", timestamp=get_current_time_iso()).model_dump())
                pass # Web search 逻辑占位

            # 准备API请求 (主要针对OpenAI兼容路径，因为非Gemini的Google路径基本被Gemini路径覆盖)
            # 您原来的 chat.py 中，provider=="openai" 的分支逻辑会放在这里
            if chat_input.provider != "openai" and not chat_input.api_address: # 简单的例子，您可能有更复杂的provider逻辑
                logger.warning(f"{log_prefix}: Non-Gemini model '{chat_input.model}' with provider '{chat_input.provider}' but no api_address. Assuming OpenAI compatible structure but this might fail.")
            
            # 调用 api_helpers.prepare_openai_request (或类似函数)
            # 这个函数应该接收 upstream_api_messages (已经是 List[Dict[str,str]])
            final_api_url, final_api_headers, final_api_payload = prepare_openai_request(
                request_data=chat_input, # 传递整个请求数据，以便 prepare 函数能获取所有参数
                processed_messages=upstream_api_messages, # 传递处理过的、符合OpenAI格式的消息
                request_id=request_id
            )
            logger.info(f"{log_prefix}: Sending to non-Gemini (OpenAI-like) endpoint: {final_api_url}")
            logger.debug(f"{log_prefix}: Payload for non-Gemini: {str(final_api_payload)[:1000]}")


            # --- 流式请求和处理SSE ---
            # (这部分与您原chat.py中的SSE处理循环非常相似)
            buffer = bytearray()
            upstream_ok_flag = False
            first_chunk_received_flag = False
            # stream_processors.py 中的 state 初始化可能需要根据上下文调整
            stream_proc_state: Dict[str, Any] = {
                "accumulated_openai_content": "", "accumulated_openai_reasoning": "",
                "openai_had_any_reasoning": False, "openai_had_any_content_or_tool_call": False,
                "openai_reasoning_finish_event_sent": False,
                # Google相关的state在这里不激活
            }

            try:
                async with http_client.stream(
                    "POST", final_api_url,
                    headers=final_api_headers,
                    json=final_api_payload, # prepare_openai_request 应返回可以直接json序列化的payload
                    # params=... 如果有需要的话
                    timeout=300.0 # 或来自配置
                ) as response:
                    logger.info(f"{log_prefix}: Non-Gemini upstream status: {response.status_code}")
                    if not (200 <= response.status_code < 300):
                        err_body = await response.aread()
                        # ... (错误处理，yield error AppStreamEventPy) ...
                        logger.error(f"{log_prefix}: Non-Gemini upstream error {response.status_code}: {err_body.decode(errors='ignore')[:500]}")
                        yield orjson_dumps_bytes_wrapper(AppStreamEventPy(type="error", message=f"LLM API Error: {response.status_code}", upstream_status=response.status_code, timestamp=get_current_time_iso()).model_dump())
                        return

                    upstream_ok_flag = True
                    async for raw_chunk_bytes in response.aiter_raw():
                        if await fastapi_request_obj.is_disconnected():
                            logger.info(f"{log_prefix}: Client disconnected (non-Gemini stream).")
                            break
                        if not first_chunk_received_flag:
                            if chat_input.use_web_search and user_query_for_search:
                                stage = "web_analysis_complete" if search_results_generated_flag else "web_analysis_skipped_no_results"
                                yield orjson_dumps_bytes_wrapper(AppStreamEventPy(type="status_update", stage=stage, timestamp=get_current_time_iso()).model_dump())
                            first_chunk_received_flag = True
                        
                        buffer.extend(raw_chunk_bytes)
                        sse_lines, buffer = extract_sse_lines(buffer)

                        for sse_line_bytes in sse_lines:
                            # ... (您的SSE行处理逻辑，调用 process_openai_like_sse_stream)
                            # 确保 process_openai_like_sse_stream 返回的是 AppStreamEventPy 兼容的字典或直接是模型实例
                            if not sse_line_bytes.strip(): continue
                            sse_data_bytes = b""
                            if sse_line_bytes.startswith(b"data: "):
                                sse_data_bytes = sse_line_bytes[len(b"data: "):].strip()
                            if not sse_data_bytes: continue
                            
                            if sse_data_bytes == b"[DONE]": # OpenAI [DONE]
                                logger.info(f"{log_prefix}: Received [DONE] from non-Gemini endpoint.")
                                # Flush any remaining buffered content from stream_proc_state
                                if stream_proc_state.get("accumulated_openai_content"):
                                    processed_content = strip_potentially_harmful_html_and_normalize_newlines(stream_proc_state["accumulated_openai_content"])
                                    if processed_content: yield orjson_dumps_bytes_wrapper(AppStreamEventPy(type="content", text=processed_content, timestamp=get_current_time_iso()).model_dump())
                                yield orjson_dumps_bytes_wrapper(AppStreamEventPy(type="finish", reason="stop_openai_done", timestamp=get_current_time_iso()).model_dump())
                                return # Stream finished

                            try:
                                parsed_sse_data = orjson.loads(sse_data_bytes)
                                async for event_dict in process_openai_like_sse_stream(parsed_sse_data, stream_proc_state, request_id): # 假设它返回 AppStreamEventPy 兼容的字典
                                    yield orjson_dumps_bytes_wrapper(AppStreamEventPy(**event_dict).model_dump()) # 转换为Pydantic模型再dump
                                    if event_dict.get("type") == "finish": return # 如果处理器内部决定结束
                            except Exception as e_proc:
                                logger.error(f"{log_prefix}: Error processing SSE data for non-Gemini: {sse_data_bytes.decode(errors='ignore')[:100]}, error: {e_proc}")
                                # continue or yield error
            
            except httpx.RequestError as e_req:
                async for event_bytes in handle_stream_error(e_req, request_id, upstream_ok_flag, first_chunk_received_flag): yield event_bytes
            except Exception as e_gen:
                async for event_bytes in handle_stream_error(e_gen, request_id, upstream_ok_flag, first_chunk_received_flag): yield event_bytes
            finally:
                # Final cleanup, flush any remaining data
                async for event_bytes in handle_stream_cleanup(
                    stream_proc_state, request_id, upstream_ok_flag,
                    False, # use_old_custom_separator_branch_flag - 假设非Gemini不用旧逻辑
                    chat_input.provider # 传递原始provider
                ):
                    yield event_bytes
        
        return StreamingResponse(stream_generator_non_gemini(), media_type="text/event-stream", headers=COMMON_HEADERS)