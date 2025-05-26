# routers/multimodal_chat.py
import os
import logging
import httpx
import orjson
from typing import Optional, Dict, Any, AsyncGenerator, List

from fastapi import APIRouter, Depends, Request
from fastapi.responses import StreamingResponse

# 导入后端 Pydantic 模型 (确保它们支持多模态 content)
from ..models import ChatRequest, ApiMessage, ApiContentPart, ApiImageUrlPart 
from ..config import GOOGLE_API_BASE_URL, COMMON_HEADERS
from ..utils import (
    extract_sse_lines, get_current_time_iso,
    orjson_dumps_bytes_wrapper, strip_potentially_harmful_html_and_normalize_newlines
)
# 导入多模态相关的辅助函数
from ..multimodal_models import is_model_multimodal 
from ..multimodal_api_helpers import prepare_openai_multimodal_request, prepare_google_multimodal_request
# Web搜索和流处理器可能也需要，或者为多模态做特定调整
from ..web_search import perform_web_search, generate_search_context_message_content
from ..stream_processors import ( # 这些处理器可能需要知道如何处理从多模态模型返回的响应
    process_openai_response, process_google_response, # 可能需要多模态版本
    handle_stream_error, handle_stream_cleanup,
    should_apply_custom_separator_logic # 这个可能也需要考虑多模态
)

logger = logging.getLogger("EzTalkProxy.Routers.MultimodalChat") # 新的 logger 名称
router = APIRouter() # 新的 router 实例

async def get_http_client(request: Request) -> Optional[httpx.AsyncClient]:
    return getattr(request.app.state, "http_client", None)

@router.post("/chat_multimodal", response_class=StreamingResponse, summary="AI多模态聊天代理", tags=["AI Proxy Multimodal"])
async def multimodal_chat_proxy(
    request_data: ChatRequest, # ChatRequest Pydantic 模型应该能处理多模态 content
    client: Optional[httpx.AsyncClient] = Depends(get_http_client)
):
    request_id = os.urandom(8).hex()
    logger.info(
        f"RID-{request_id}: Received /chat_multimodal request: Provider='{request_data.provider}', Model='{request_data.model}'"
    )

    if not client:
        # ... (client 错误处理) ...
        pass

    # 1. 检查模型是否确实支持多模态 (可选，但推荐)
    if not is_model_multimodal(request_data.provider, request_data.model):
        logger.warning(f"RID-{request_id}: Model '{request_data.model}' called on multimodal endpoint but not listed as multimodal.")
        # 可以选择报错，或者尝试按多模态处理（如果 prepare 函数能优雅降级）
        # For now, we proceed, assuming prepare_..._multimodal_request handles it or the list is comprehensive.

    # 2. 将前端 ApiMessage 转换为通用的 List[Dict[str, Any]] 结构
    #    这个结构将包含原始的 content (List[ApiContentPart Pydantic模型] 或 str)
    messages_for_llm_preparation: List[Dict[str, Any]] = []
    user_query_for_search_parts: List[str] = []

    for msg_model in request_data.messages:
        if msg_model.content is None and msg_model.tool_calls is None and msg_model.role != "system":
            continue
        
        msg_dict_generic = {"role": msg_model.role}
        current_content_for_generic = []
        
        if isinstance(msg_model.content, list): # Pydantic 解析为 List[ApiContentPart]
            for part_model in msg_model.content:
                part_dict = {"type": part_model.type}
                if part_model.type == "text" and part_model.text:
                    part_dict["text"] = part_model.text
                    if msg_model.role == "user": user_query_for_search_parts.append(part_model.text)
                elif part_model.type == "image_url" and part_model.image_url and part_model.image_url.url:
                    part_dict["image_url"] = {"url": part_model.image_url.url, "detail": part_model.image_url.detail or "auto"}
                current_content_for_generic.append(part_dict)
        elif isinstance(msg_model.content, str) and msg_model.content.strip(): # 兼容纯文本部分
            current_content_for_generic.append({"type": "text", "text": msg_model.content.strip()})
            if msg_model.role == "user": user_query_for_search_parts.append(msg_model.content.strip())
        
        if current_content_for_generic:
            msg_dict_generic["content"] = current_content_for_generic
        elif msg_model.role == "system" and not current_content_for_generic :
             msg_dict_generic["content"] = []


        if msg_model.tool_calls:
            msg_dict_generic["tool_calls"] = [tc.model_dump(exclude_none=True) for tc in msg_model.tool_calls]
        if msg_model.role == "tool":
            if msg_model.tool_call_id: msg_dict_generic["tool_call_id"] = msg_model.tool_call_id
            if msg_model.name: msg_dict_generic["name"] = msg_model.name
            if isinstance(msg_model.content, str) and "content" not in msg_dict_generic : # tool content is string
                msg_dict_generic["content"] = msg_model.content


        if "content" in msg_dict_generic or "tool_calls" in msg_dict_generic or \
           (msg_model.role == "system" and "content" in msg_dict_generic) or \
           (msg_model.role == "tool" and "tool_call_id" in msg_dict_generic) :
            messages_for_llm_preparation.append(msg_dict_generic)

    user_query_for_search = " ".join(user_query_for_search_parts).strip()

    if not any(m.get("role") != "system" for m in messages_for_llm_preparation):
        if not any(m.get("role") == "system" and m.get("content") for m in messages_for_llm_preparation):
            # ... (no_message_error_gen)
            pass
    
    # --- Provider 和路径选择逻辑 (与原 chat.py 类似，但调用多模态的 prepare 函数) ---
    is_native_thinking_mode_active = False # 这些 flag 可能在多模态 prepare 函数中设置
    use_google_sse_parser_flag = False
    # is_google_payload_format_used_flag = False # 这个 flag 可能更多地与 prepare 函数的内部实现有关
    # is_google_like_path_active = False


    if request_data.provider == "google":
        use_google_sse_parser_flag = True # Google 通常使用其特定的 SSE 格式
        # is_google_like_path_active = True # 这个flag的原始用途需要审视
    
    # stream_generator 内部逻辑与原 chat.py 非常相似，但 prepare_... 函数调用不同
    async def stream_generator() -> AsyncGenerator[bytes, None]:
        nonlocal messages_for_llm_preparation, user_query_for_search # 使用新处理的消息列表
        # ... (声明其他需要 nonlocal 的 flags)
        nonlocal is_native_thinking_mode_active, use_google_sse_parser_flag

        if not client:
            # ... (client error)
            pass
        
        current_api_payload: Dict[str, Any]
        current_api_url: str
        current_api_headers: Dict[str, str]
        current_api_params: Optional[Dict[str, str]] = None

        effective_messages_for_api = messages_for_llm_preparation # 初始值

        # Web 搜索逻辑 (如果多模态请求也支持 Web 搜索)
        if request_data.use_web_search and user_query_for_search:
            yield orjson_dumps_bytes_wrapper({"type": "status_update", "stage": "web_search_started", "timestamp": get_current_time_iso()})
            search_results_list = await perform_web_search(user_query_for_search, request_id)
            if search_results_list:
                search_context_content = generate_search_context_message_content(user_query_for_search, search_results_list)
                new_system_message_dict = {"role": "system", "content": [{"type": "text", "text": search_context_content}]}
                
                temp_messages_with_search = effective_messages_for_api.copy() # 操作副本
                last_user_message_idx_for_injection = -1
                for i, msg_dict_item in reversed(list(enumerate(temp_messages_with_search))):
                    if msg_dict_item.get("role") == "user":
                        last_user_message_idx_for_injection = i
                        break
                if last_user_message_idx_for_injection != -1:
                    temp_messages_with_search.insert(last_user_message_idx_for_injection, new_system_message_dict)
                else:
                    temp_messages_with_search.insert(0, new_system_message_dict)
                effective_messages_for_api = temp_messages_with_search # 更新

                logger.info(f"RID-{request_id}: (Multimodal) Web search context injected.")
                yield orjson_dumps_bytes_wrapper({"type": "status_update", "stage": "web_search_complete_with_results", "query": user_query_for_search, "timestamp": get_current_time_iso()})
                yield orjson_dumps_bytes_wrapper({"type": "web_search_results", "results": [r.model_dump() for r in search_results_list], "timestamp": get_current_time_iso()})
            else:
                # ... (no results)
                pass
            yield orjson_dumps_bytes_wrapper({"type": "status_update", "stage": "web_analysis_started", "timestamp": get_current_time_iso()})


        # 调用特定于多模态的 prepare 函数
        if request_data.provider == "google":
            current_api_payload, is_native_thinking_mode_active = prepare_google_multimodal_request(
                request_data, effective_messages_for_api, request_id
            )
            current_api_url = f"{GOOGLE_API_BASE_URL}/v1beta/models/{request_data.model}:streamGenerateContent" # 或 vision 特定端点
            current_api_params = {"key": request_data.api_key, "alt": "sse"}
            current_api_headers = {"Content-Type": "application/json"}

        elif request_data.provider == "openai":
            current_api_url, current_api_headers, current_api_payload = prepare_openai_multimodal_request(
                request_data, effective_messages_for_api, request_id
            )
        else: # 理论上不应到达这里，因为前面有 provider 检查
            yield orjson_dumps_bytes_wrapper({"type": "error", "message": f"Multimodal handler: Unsupported provider '{request_data.provider}'", "timestamp": get_current_time_iso()})
            yield orjson_dumps_bytes_wrapper({"type": "finish", "reason": "bad_request", "timestamp": get_current_time_iso()})
            return

        # 你的 should_apply_custom_separator_logic 可能对多模态不适用或需要调整
        # use_old_custom_separator_branch_flag = should_apply_custom_separator_logic(...)

        # 日志、buffer、state 初始化
        # ... (与原 chat.py 类似)
        logger.debug(f"RID-{request_id}: (Multimodal) Sending to URL: {current_api_url}...")


        # VVVVVV 从这里开始的 async with client.stream(...) 到 finally 块 VVVVVV
        # 这部分流式处理、SSE解析、调用 process_openai_response/process_google_response、
        # 错误处理、清理逻辑，与你现有的 chat.py 中的 stream_generator 内部的这部分代码
        # **几乎完全相同或非常相似**。
        #
        # 主要区别可能在于：
        # 1. `use_google_sse_parser_flag` 的设置。
        # 2. `process_openai_response` 和 `process_google_response` 是否需要为多模态模型的响应做特殊处理
        #    (通常多模态模型的文本响应部分与纯文本模型相似，但可能包含对图片的引用或描述)。
        # 3. `should_apply_custom_separator_logic` 的适用性。
        #
        # **为了不重复大量代码，你可以考虑将这个通用的流处理循环也提取到一个辅助函数中，**
        # 或者确保这里的实现与 `chat.py` 中的保持一致（如果响应格式没有本质区别）。
        #
        # 我将复制粘贴你之前提供的 `chat.py` 中 `stream_generator` 的 `try/except/finally` 块，
        # 你需要仔细检查并调整 `use_google_sse_parser_flag` 和 `state` 的初始化，
        # 以及 `process_..._response` 是否能正确处理。
        #
        # [ 这里应该是你原有的 try/except/finally 流处理逻辑 ]
        # 为了代码完整性，我将粘贴一个结构，你需要用你的详细逻辑填充
        buffer = bytearray()
        upstream_ok = False
        first_chunk_llm = False
        state: Dict[str, Any] = { # 根据你的 stream_processors 初始化 state
            "accumulated_openai_content": "", "accumulated_openai_reasoning": "",
            "openai_had_any_reasoning": False, "openai_had_any_content_or_tool_call": False,
            "openai_reasoning_finish_event_sent": False,
            "accumulated_google_thought": "", "accumulated_google_text": "",
            "google_native_had_thoughts": False, "google_native_had_answer": False,
            "accumulated_text_custom": "", "full_yielded_reasoning_custom": "",
            "full_yielded_content_custom": "", "found_separator_custom": False,
        }
        _use_old_custom_separator_branch_flag = False # 假设多模态默认不用旧逻辑，或根据模型判断

        try:
            async with client.stream("POST", current_api_url, headers=current_api_headers, json=current_api_payload, params=current_api_params) as resp:
                logger.info(f"RID-{request_id}: (Multimodal) Upstream LLM response status: {resp.status_code}")
                if not (200 <= resp.status_code < 300):
                    # ... (错误处理)
                    return

                upstream_ok = True
                async for raw_chunk_bytes in resp.aiter_raw():
                    if not raw_chunk_bytes: continue
                    if not first_chunk_llm:
                        if request_data.use_web_search and user_query_for_search and search_results_generated:
                            yield orjson_dumps_bytes_wrapper({"type": "status_update", "stage": "web_analysis_complete", "timestamp": get_current_time_iso()})
                        first_chunk_llm = True
                    buffer.extend(raw_chunk_bytes)
                    sse_lines, buffer = extract_sse_lines(buffer)

                    for sse_line_bytes in sse_lines:
                        # ... (SSE行处理，与原chat.py相同)
                        if not sse_line_bytes.strip(): continue
                        sse_data_bytes = b""
                        if sse_line_bytes.startswith(b"data: "):
                            sse_data_bytes = sse_line_bytes[len(b"data: "):].strip()
                        if not sse_data_bytes: continue

                        if sse_data_bytes == b"[DONE]":
                            # ... (与原chat.py相同的[DONE]处理逻辑)
                            return

                        try: parsed_sse_data = orjson.loads(sse_data_bytes)
                        except: # ...
                            continue
                        
                        if use_google_sse_parser_flag: # Google
                            async for event in process_google_response(parsed_sse_data, state, request_id, is_native_thinking_mode_active, _use_old_custom_separator_branch_flag): # 传递正确的flag
                                yield event
                                # ... (检查 type == "finish")
                        else: # OpenAI
                            async for event in process_openai_response(parsed_sse_data, state, request_id): # _use_old_custom_separator_branch_flag 对 OpenAI 可能不适用
                                yield event
                                # ... (检查 type == "finish")

        except Exception as e:
            async for event in handle_stream_error(e, request_id, upstream_ok, first_chunk_llm):
                yield event
        finally:
            async for event in handle_stream_cleanup(
                state, request_id, upstream_ok,
                _use_old_custom_separator_branch_flag, # 传递正确的flag
                request_data.provider
            ):
                yield event
    return StreamingResponse(stream_generator(), media_type="text/event-stream", headers=COMMON_HEADERS)