import logging
import asyncio
import httpx
from typing import Dict, Any, AsyncGenerator

from ..models.api_models import AppStreamEventPy, ChatRequestModel
from ..utils.helpers import (
    get_current_time_iso,
    orjson_dumps_bytes_wrapper,
    strip_potentially_harmful_html_and_normalize_newlines
)

logger = logging.getLogger("EzTalkProxy.StreamProcessors")

MIN_REASONING_FLUSH_CHUNK_SIZE = 1
MIN_CONTENT_FLUSH_CHUNK_SIZE = 20

async def process_openai_like_sse_stream(
    parsed_sse_data: Dict[str, Any],
    current_processing_state: Dict[str, Any],
    request_id: str
) -> AsyncGenerator[Dict[str, Any], None]:
    log_prefix = f"RID-{request_id}"
    
    state = current_processing_state
    state.setdefault("had_any_reasoning", False)
    state.setdefault("reasoning_finish_event_sent", False)
    state.setdefault("final_finish_event_sent_by_llm_reason", False)

    for choice in parsed_sse_data.get('choices', []):
        delta = choice.get('delta', {})
        finish_reason = choice.get("finish_reason")

        reasoning_chunk = delta.get("reasoning_content")
        content_chunk = delta.get("content")
        tool_calls_chunk = delta.get("tool_calls")

        # Send reasoning chunk immediately if it exists
        if reasoning_chunk:
            processed_reasoning = strip_potentially_harmful_html_and_normalize_newlines(str(reasoning_chunk))
            if processed_reasoning:
                yield {"type": "reasoning", "text": processed_reasoning, "timestamp": get_current_time_iso()}
                state["had_any_reasoning"] = True

        # Send content chunk immediately if it exists
        if content_chunk:
            # If we receive content and haven't sent reasoning_finish, send it now.
            if state["had_any_reasoning"] and not state["reasoning_finish_event_sent"]:
                yield {"type": "reasoning_finish", "timestamp": get_current_time_iso()}
                state["reasoning_finish_event_sent"] = True
            
            # 直接传递原始文本块，不做任何清洗。
            # 前端渲染库（如MarkdownView）通常有自己的安全机制。
            # 后端清洗可能会破坏合法的格式，如LaTeX。
            if content_chunk:
                yield {"type": "content", "text": str(content_chunk), "timestamp": get_current_time_iso()}

        # Handle tool calls and finish reason
        if tool_calls_chunk or finish_reason:
            # If there was any reasoning, ensure the finish event is sent before the final block.
            if state["had_any_reasoning"] and not state["reasoning_finish_event_sent"]:
                yield {"type": "reasoning_finish", "timestamp": get_current_time_iso()}
                state["reasoning_finish_event_sent"] = True
            if tool_calls_chunk:
                yield {"type": "tool_calls_chunk", "data": tool_calls_chunk, "timestamp": get_current_time_iso()}
            
            if finish_reason:
                logger.info(f"{log_prefix}: OpenAI-like choice finish_reason: {finish_reason}.")
                yield {"type": "finish", "reason": finish_reason, "timestamp": get_current_time_iso()}
                state["final_finish_event_sent_by_llm_reason"] = True

def should_apply_custom_separator_logic(
    request_data: ChatRequestModel,
    request_id: str,
    is_google_like_path: bool,
    is_native_thinking_active: bool
) -> bool:
    log_prefix = f"RID-{request_id}"
    if request_data.force_custom_reasoning_prompt:
        logger.info(f"{log_prefix}: Custom separator logic FORCED by request.")
        return True
    if is_google_like_path and is_native_thinking_active:
        logger.info(f"{log_prefix}: Custom separator logic OFF (Google native thinking via part.thought active).")
        return False
    if "deepseek" in request_data.model.lower() or request_data.provider.lower() == "mke":
        logger.info(f"{log_prefix}: Custom separator logic OFF for Deepseek/MKE (using reasoning_content or <think> tags).")
        return False
    logger.info(f"{log_prefix}: Custom separator logic OFF by default for model '{request_data.model}'.")
    return False


async def handle_stream_error(
    error: Exception, request_id: str, upstream_responded_ok: bool, first_chunk_from_llm_received: bool
) -> AsyncGenerator[bytes, None]:
    log_prefix = f"RID-{request_id}"
    logger.error(f"{log_prefix}: Stream error: {type(error).__name__} - {error}", exc_info=True)
    error_message = f"Stream processing error: {type(error).__name__}"
    if isinstance(error, httpx.TimeoutException): error_message = "Request to LLM API timed out."
    elif isinstance(error, httpx.RequestError): error_message = f"Network error: {error}"
    elif isinstance(error, asyncio.CancelledError): logger.info(f"{log_prefix}: Stream cancelled."); return
    else: error_message = f"Unexpected error: {str(error)[:200]}"
    yield orjson_dumps_bytes_wrapper(AppStreamEventPy(type="error", message=error_message, timestamp=get_current_time_iso()).model_dump(by_alias=True, exclude_none=True))
    yield orjson_dumps_bytes_wrapper(AppStreamEventPy(type="finish", reason="error_in_stream", timestamp=get_current_time_iso()).model_dump(by_alias=True, exclude_none=True))

async def handle_stream_cleanup(
    processing_state: Dict[str, Any], request_id: str,
    upstream_ok: bool, use_old_custom_separator_logic: bool, provider: str
) -> AsyncGenerator[bytes, None]:
    log_prefix = f"RID-{request_id}"
    state = processing_state
    logger.info(f"{log_prefix}: Stream cleanup. Provider: {provider}. Upstream OK: {upstream_ok}. CustomSep: {use_old_custom_separator_logic}")

    accumulated_reasoning = state.get("accumulated_reasoning", state.get("accumulated_openai_reasoning", ""))
    accumulated_content = state.get("accumulated_content", state.get("accumulated_openai_content", ""))
    had_any_reasoning = state.get("had_any_reasoning", state.get("openai_had_any_reasoning", False))
    reasoning_finish_sent = state.get("reasoning_finish_event_sent", state.get("openai_reasoning_finish_event_sent", False))


    if accumulated_reasoning:
        processed_r = strip_potentially_harmful_html_and_normalize_newlines(accumulated_reasoning)
        if processed_r:
            logger.info(f"{log_prefix}: Cleanup: Flushing remaining reasoning: '{processed_r[:100]}...'")
            yield orjson_dumps_bytes_wrapper(AppStreamEventPy(type="reasoning", text=processed_r, timestamp=get_current_time_iso()).model_dump(by_alias=True, exclude_none=True))
    
    if had_any_reasoning and not reasoning_finish_sent:
        logger.info(f"{log_prefix}: Cleanup: Sending reasoning_finish event.")
        yield orjson_dumps_bytes_wrapper(AppStreamEventPy(type="reasoning_finish", timestamp=get_current_time_iso()).model_dump(by_alias=True, exclude_none=True))

    if accumulated_content:
        processed_c = strip_potentially_harmful_html_and_normalize_newlines(accumulated_content)
        if processed_c and processed_c.strip():
            logger.info(f"{log_prefix}: Cleanup: Flushing remaining content: '{processed_c[:100]}...'")
            yield orjson_dumps_bytes_wrapper(AppStreamEventPy(type="content", text=processed_c, timestamp=get_current_time_iso()).model_dump(by_alias=True, exclude_none=True))
    
    if use_old_custom_separator_logic and state.get("accumulated_text_custom","").strip() and upstream_ok:
        pass

    if not upstream_ok and not state.get("final_finish_event_sent_by_llm_reason") and not state.get("final_finish_event_sent_flag_for_cleanup", False):
        yield orjson_dumps_bytes_wrapper(AppStreamEventPy(type="finish", reason="upstream_error_or_connection_failed", timestamp=get_current_time_iso()).model_dump(by_alias=True, exclude_none=True))
    elif not state.get("final_finish_event_sent_by_llm_reason") and not state.get("final_finish_event_sent_flag_for_cleanup", False):
        final_reason = state.get("final_finish_reason_from_llm", "stream_end")
        yield orjson_dumps_bytes_wrapper(AppStreamEventPy(type="finish", reason=final_reason, timestamp=get_current_time_iso()).model_dump(by_alias=True, exclude_none=True))