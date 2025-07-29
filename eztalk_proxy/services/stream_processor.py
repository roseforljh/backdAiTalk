import logging
import asyncio
import httpx
from typing import Dict, Any, AsyncGenerator

from ..models.api_models import AppStreamEventPy, ChatRequestModel
from ..utils.helpers import (
    get_current_time_iso,
    orjson_dumps_bytes_wrapper
)

logger = logging.getLogger("EzTalkProxy.StreamProcessors")

MIN_REASONING_FLUSH_CHUNK_SIZE = 1
MIN_CONTENT_FLUSH_CHUNK_SIZE = 20

def is_excessive_whitespace(text: str) -> bool:
    """
    检查文本是否只包含过多的空白字符
    针对OpenAI兼容接口经常产生大量空白段落的问题
    """
    if not text:
        return True
    
    # 如果文本只包含空白字符且长度超过10个字符，认为是过多的空白
    whitespace_only = text.replace('\n', '').replace(' ', '').replace('\t', '')
    if not whitespace_only and len(text) > 10:
        return True
    
    # 如果连续换行符超过3个，认为是过多的空白
    if '\n\n\n' in text:
        return True
        
    return False

def should_skip_content_chunk(content_str: str, accumulated_content: str) -> bool:
    """
    判断是否应该跳过当前的内容块
    """
    if is_excessive_whitespace(content_str):
        return True
    
    if content_str.strip() == '' and len(content_str) > 5:
        return True
        
    return False

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
    state.setdefault("accumulated_content", "")

    for choice in parsed_sse_data.get('choices', []):
        delta = choice.get('delta', {})
        finish_reason = choice.get("finish_reason")

        reasoning_chunk = delta.get("reasoning_content")
        content_chunk = delta.get("content")
        tool_calls_chunk = delta.get("tool_calls")

        # Send reasoning chunk immediately if it exists
        if reasoning_chunk:
            if reasoning_chunk:
                yield {"type": "reasoning", "text": str(reasoning_chunk), "timestamp": get_current_time_iso()}
                state["had_any_reasoning"] = True

        # Accumulate content and flush when it reaches a certain size
        if content_chunk is not None:
            # Convert to string and check if it's not just None or empty
            content_str = str(content_chunk)
            if content_str and not should_skip_content_chunk(content_str, state["accumulated_content"]):
                state["accumulated_content"] += content_str
                # If we receive content and haven't sent reasoning_finish, send it now.
                if state["had_any_reasoning"] and not state["reasoning_finish_event_sent"]:
                    yield {"type": "reasoning_finish", "timestamp": get_current_time_iso()}
                    state["reasoning_finish_event_sent"] = True
                
                if len(state["accumulated_content"]) >= MIN_CONTENT_FLUSH_CHUNK_SIZE:
                    # Send accumulated content (preserve formatting)
                    yield {"type": "content", "text": state["accumulated_content"], "timestamp": get_current_time_iso()}
                    state["accumulated_content"] = ""

        # Handle tool calls and finish reason
        if tool_calls_chunk or finish_reason:
            # Before handling finish or tool calls, flush any remaining content.
            if state["accumulated_content"]:
                yield {"type": "content", "text": state["accumulated_content"], "timestamp": get_current_time_iso()}
                state["accumulated_content"] = ""

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

    accumulated_content = state.get("accumulated_content", "")
    had_any_reasoning = state.get("had_any_reasoning", False)
    reasoning_finish_sent = state.get("reasoning_finish_event_sent", False)

    # If there was reasoning, ensure the finish event is sent.
    if had_any_reasoning and not reasoning_finish_sent:
        logger.info(f"{log_prefix}: Cleanup: Sending reasoning_finish event.")
        yield orjson_dumps_bytes_wrapper(AppStreamEventPy(type="reasoning_finish", timestamp=get_current_time_iso()).model_dump(by_alias=True, exclude_none=True))

    # Flush any remaining content in the buffer
    if accumulated_content:
        # Send all remaining content (preserve formatting)
        logger.info(f"{log_prefix}: Cleanup: Flushing remaining content: '{accumulated_content[:100]}...'")
        yield orjson_dumps_bytes_wrapper(AppStreamEventPy(type="content", text=accumulated_content, timestamp=get_current_time_iso()).model_dump(by_alias=True, exclude_none=True))

    # Send the final finish event if it hasn't been sent already by a finish_reason from the LLM.
    if not state.get("final_finish_event_sent_by_llm_reason"):
        final_reason = "stream_end"
        if not upstream_ok:
            final_reason = "upstream_error_or_connection_failed"
        
        logger.info(f"{log_prefix}: Cleanup: Sending final finish event with reason '{final_reason}'.")
        yield orjson_dumps_bytes_wrapper(AppStreamEventPy(type="finish", reason=final_reason, timestamp=get_current_time_iso()).model_dump(by_alias=True, exclude_none=True))