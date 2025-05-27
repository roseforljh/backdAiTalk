# eztalk_proxy/stream_processors.py
import logging
import orjson
from typing import Dict, Any, AsyncGenerator, List, Optional

from .models import AppStreamEventPy, ChatRequestModel # 导入 AppStreamEventPy 和 ChatRequestModel
from .utils import (
    get_current_time_iso,
    orjson_dumps_bytes_wrapper,
    strip_potentially_harmful_html_and_normalize_newlines
)
from .config import THINKING_PROCESS_SEPARATOR, MIN_FLUSH_LENGTH_HEURISTIC # 导入配置

logger = logging.getLogger("EzTalkProxy.StreamProcessors")

async def process_openai_like_sse_stream(
    parsed_sse_data: Dict[str, Any],
    current_processing_state: Dict[str, Any],
    request_id: str
) -> AsyncGenerator[Dict[str, Any], None]:
    log_prefix = f"RID-{request_id}"
    delta = parsed_sse_data.get("choices", [{}])[0].get("delta", {})
    text_chunk: Optional[str] = delta.get("content")
    role_chunk: Optional[str] = delta.get("role")
    tool_calls_chunk: Optional[List[Dict[str, Any]]] = delta.get("tool_calls")
    finish_reason: Optional[str] = parsed_sse_data.get("choices", [{}])[0].get("finish_reason")

    current_processing_state.setdefault("accumulated_openai_content", "")
    current_processing_state.setdefault("openai_had_any_content_or_tool_call", False)
    # 这个状态用于标记是否由LLM的finish_reason触发了最终的清理，以避免重复发送finish事件
    current_processing_state.setdefault("final_finish_event_sent_by_llm_reason", False)


    if role_chunk == "assistant" and not text_chunk and not tool_calls_chunk:
        logger.debug(f"{log_prefix}: Received role-only chunk for assistant.")
        pass # 通常可以忽略纯角色块

    if text_chunk:
        current_processing_state["accumulated_openai_content"] += text_chunk
        current_processing_state["openai_had_any_content_or_tool_call"] = True
        if len(current_processing_state["accumulated_openai_content"]) >= MIN_FLUSH_LENGTH_HEURISTIC or "\n" in current_processing_state["accumulated_openai_content"]:
            processed_text = strip_potentially_harmful_html_and_normalize_newlines(current_processing_state["accumulated_openai_content"])
            if processed_text:
                 yield AppStreamEventPy(type="content", text=processed_text, timestamp=get_current_time_iso()).model_dump(by_alias=True, exclude_none=True)
            current_processing_state["accumulated_openai_content"] = ""

    if tool_calls_chunk:
        current_processing_state["openai_had_any_content_or_tool_call"] = True
        yield AppStreamEventPy(
            type="tool_calls_chunk",
            data=tool_calls_chunk,
            timestamp=get_current_time_iso()
        ).model_dump(by_alias=True, exclude_none=True)

    if finish_reason:
        logger.info(f"{log_prefix}: OpenAI-like stream finished with reason from LLM: {finish_reason}")
        # 记录LLM的完成原因，供cleanup参考
        current_processing_state["final_finish_reason_from_llm"] = finish_reason
        # 注意：我们不在这个函数内部直接发送 type="finish" 的 AppStreamEventPy。
        # 这个函数只负责处理单个SSE块。
        # 调用者（routers/chat.py）在检测到[DONE]信号或这个包含finish_reason的块后，
        # 会调用 handle_stream_cleanup 来发送累积内容和最终的 "finish" 事件。
        # 但我们可以标记一下，以便 cleanup 知道LLM已明确结束。
        current_processing_state["final_finish_event_sent_by_llm_reason"] = True # 表示LLM端已结束


def should_apply_custom_separator_logic(
    request_data: ChatRequestModel,
    request_id: str,
    is_google_like_path: bool, # 这个参数的含义可能需要重新审视，因为Gemini和非Gemini现在路径分开了
    is_native_thinking_active: bool # 这个参数主要用于Gemini路径
) -> bool:
    log_prefix = f"RID-{request_id}"
    if request_data.force_custom_reasoning_prompt:
        logger.info(f"{log_prefix}: Custom separator logic FORCED by request.")
        return True
    # 对于非Gemini路径 (即此stream_processor主要服务的路径)，通常不使用Google原生思考
    if is_google_like_path and is_native_thinking_active:
        logger.info(f"{log_prefix}: Custom separator logic OFF (Google native thinking active for a non-Gemini path - unusual).")
        return False
    logger.info(f"{log_prefix}: Custom separator logic OFF by default for non-Gemini model '{request_data.model}'.")
    return False


async def handle_stream_error(
    error: Exception,
    request_id: str,
    upstream_responded_ok: bool,
    first_chunk_from_llm_received: bool
) -> AsyncGenerator[bytes, None]:
    log_prefix = f"RID-{request_id}"
    logger.error(f"{log_prefix}: Stream error: {type(error).__name__} - {error}", exc_info=True)
    error_message = f"Stream processing error: {type(error).__name__}"
    if isinstance(error, httpx.TimeoutException):
        error_message = "Request to LLM API timed out."
    elif isinstance(error, httpx.NetworkError):
        error_message = f"Network error while communicating with LLM API: {error}"
    else:
        error_message = f"Unexpected error during stream: {str(error)[:200]}" # 限制错误消息长度

    yield orjson_dumps_bytes_wrapper(AppStreamEventPy(type="error", message=error_message, timestamp=get_current_time_iso()).model_dump(by_alias=True, exclude_none=True))
    yield orjson_dumps_bytes_wrapper(AppStreamEventPy(type="finish", reason="error_in_stream", timestamp=get_current_time_iso()).model_dump(by_alias=True, exclude_none=True))


async def handle_stream_cleanup(
    processing_state: Dict[str, Any],
    request_id: str,
    upstream_responded_ok: bool,
    use_old_custom_separator_logic: bool, # <--- 确保这个参数被定义和使用
    provider: str                          # <--- 这是第5个参数
) -> AsyncGenerator[bytes, None]:
    log_prefix = f"RID-{request_id}"
    logger.info(f"{log_prefix}: Stream cleanup. Provider: {provider}. Upstream OK: {upstream_responded_ok}. CustomSep: {use_old_custom_separator_logic}")

    # 清理 OpenAI 类型的流 (非Gemini)
    if provider == "openai" or not processing_state.get("_is_google_path_flag_for_cleanup"): # 假设 provider 是 'openai' for non-Gemini
        # 冲洗剩余的 reasoning (如果有且未发送 finish)
        if processing_state.get("accumulated_openai_reasoning"):
            processed_reasoning = strip_potentially_harmful_html_and_normalize_newlines(processing_state["accumulated_openai_reasoning"])
            if processed_reasoning:
                logger.info(f"{log_prefix}: Cleanup: Flushing OpenAI reasoning: '{processed_reasoning[:100]}...'")
                yield orjson_dumps_bytes_wrapper(AppStreamEventPy(type="reasoning", text=processed_reasoning, timestamp=get_current_time_iso()).model_dump(by_alias=True, exclude_none=True))
        
        if processing_state.get("openai_had_any_reasoning") and not processing_state.get("openai_reasoning_finish_event_sent"):
            logger.info(f"{log_prefix}: Cleanup: Sending OpenAI reasoning_finish event.")
            yield orjson_dumps_bytes_wrapper(AppStreamEventPy(type="reasoning_finish", timestamp=get_current_time_iso()).model_dump(by_alias=True, exclude_none=True))
            # processing_state["openai_reasoning_finish_event_sent"] = True # 标记已发送，以防重复

        # 冲洗剩余的 content
        if processing_state.get("accumulated_openai_content"):
            processed_content = strip_potentially_harmful_html_and_normalize_newlines(processing_state["accumulated_openai_content"])
            if processed_content:
                logger.info(f"{log_prefix}: Cleanup: Flushing OpenAI content: '{processed_content[:100]}...'")
                yield orjson_dumps_bytes_wrapper(AppStreamEventPy(type="content", text=processed_content, timestamp=get_current_time_iso()).model_dump(by_alias=True, exclude_none=True))
    
    # 如果上游从未成功响应过，并且之前没有发送过错误类型的finish事件
    if not upstream_responded_ok and not processing_state.get("final_finish_event_sent_by_llm_reason") and not processing_state.get("error_event_sent_flag_for_cleanup"): # 需要一个标记来避免重复发送错误
        logger.warning(f"{log_prefix}: Upstream never responded successfully. Sending connection failed finish event.")
        yield orjson_dumps_bytes_wrapper(AppStreamEventPy(type="finish", reason="upstream_connection_failed", timestamp=get_current_time_iso()).model_dump(by_alias=True, exclude_none=True))
        processing_state["final_finish_event_sent_flag_for_cleanup"] = True # 标记已发送最终的finish

    # 只有当LLM没有明确给出finish_reason时，我们才发送一个通用的 "stream_end"
    # 并且确保之前没有因为错误或其他原因发送过 finish
    elif not processing_state.get("final_finish_event_sent_by_llm_reason") and not processing_state.get("final_finish_event_sent_flag_for_cleanup"):
        final_reason = processing_state.get("final_finish_reason_from_llm", "stream_end") # 使用LLM的，或默认
        logger.info(f"{log_prefix}: Cleanup: Sending final finish event with reason '{final_reason}'.")
        yield orjson_dumps_bytes_wrapper(AppStreamEventPy(type="finish", reason=final_reason, timestamp=get_current_time_iso()).model_dump(by_alias=True, exclude_none=True))
        processing_state["final_finish_event_sent_flag_for_cleanup"] = True