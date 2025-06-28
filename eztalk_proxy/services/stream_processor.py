import logging
import orjson
import re
import asyncio
import httpx
from typing import Dict, Any, AsyncGenerator, List, Optional

from ..models.api_models import AppStreamEventPy, ChatRequestModel
from ..utils.helpers import (
    get_current_time_iso,
    orjson_dumps_bytes_wrapper,
    strip_potentially_harmful_html_and_normalize_newlines
)
from ..core.config import THINKING_PROCESS_SEPARATOR, MIN_FLUSH_LENGTH_HEURISTIC

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
    state.setdefault("accumulated_content", "")
    state.setdefault("accumulated_reasoning", "")
    state.setdefault("had_any_reasoning", False)
    state.setdefault("had_any_content", False)
    state.setdefault("reasoning_finish_event_sent", False)
    state.setdefault("final_finish_event_sent_by_llm_reason", False)
    state.setdefault("in_think_tag_mode", False)

    for choice in parsed_sse_data.get('choices', []):
        delta = choice.get('delta', {})
        finish_reason = choice.get("finish_reason")

        current_reasoning_delta_from_chunk = ""
        current_content_delta_from_chunk = ""

        reasoning_chunk_field: Optional[str] = delta.get("reasoning_content")
        content_chunk_raw: Optional[str] = delta.get("content")
        role_chunk: Optional[str] = delta.get("role")
        tool_calls_chunk: Optional[List[Dict[str, Any]]] = delta.get("tool_calls")

        if role_chunk and not reasoning_chunk_field and not content_chunk_raw and not tool_calls_chunk:
            continue

        if reasoning_chunk_field is not None:
            if not isinstance(reasoning_chunk_field, str): reasoning_chunk_field = str(reasoning_chunk_field)
            current_reasoning_delta_from_chunk += reasoning_chunk_field
            state["had_any_reasoning"] = True
        
        if content_chunk_raw is not None:
            if not isinstance(content_chunk_raw, str): content_chunk_raw = str(content_chunk_raw)
            
            chunk_to_process_tags = content_chunk_raw
            
            if reasoning_chunk_field is None:
                entering_think_mode = "<think>" in chunk_to_process_tags and not state["in_think_tag_mode"]
                exiting_think_mode = "</think>" in chunk_to_process_tags and state["in_think_tag_mode"]

                if entering_think_mode:
                    state["in_think_tag_mode"] = True
                    parts = chunk_to_process_tags.split("<think>", 1)
                    if parts[0]: current_content_delta_from_chunk += parts[0]
                    chunk_to_process_tags = parts[1]
                    logger.info(f"{log_prefix}: Detected <think>. Enter think_tag_mode. Remainder: '{chunk_to_process_tags[:50]}'")
                    state["had_any_reasoning"] = True
                
                if exiting_think_mode:
                    parts = chunk_to_process_tags.split("</think>", 1)
                    if parts[0]: current_reasoning_delta_from_chunk += parts[0]
                    
                    full_reasoning_to_flush = state["accumulated_reasoning"] + current_reasoning_delta_from_chunk
                    if full_reasoning_to_flush:
                        clean_r_text = strip_potentially_harmful_html_and_normalize_newlines(full_reasoning_to_flush)
                        if clean_r_text: yield {"type": "reasoning", "text": clean_r_text, "timestamp": get_current_time_iso()}
                    state["accumulated_reasoning"] = ""
                    current_reasoning_delta_from_chunk = ""
                    
                    if state["had_any_reasoning"] and not state.get("reasoning_finish_event_sent"):
                        logger.info(f"{log_prefix}: Detected </think>. Sending reasoning_finish.")
                        yield {"type": "reasoning_finish", "timestamp": get_current_time_iso()}
                        state["reasoning_finish_event_sent"] = True
                    
                    state["in_think_tag_mode"] = False
                    chunk_to_process_tags = parts[1]
                    logger.info(f"{log_prefix}: Exited think_tag_mode. Remainder for content: '{chunk_to_process_tags[:50]}'")

                if state["in_think_tag_mode"]:
                    if chunk_to_process_tags: current_reasoning_delta_from_chunk += chunk_to_process_tags
                else:
                    if chunk_to_process_tags: current_content_delta_from_chunk += chunk_to_process_tags
            
            else:
                 if content_chunk_raw: current_content_delta_from_chunk += content_chunk_raw
        
        if current_reasoning_delta_from_chunk:
            state["accumulated_reasoning"] += current_reasoning_delta_from_chunk
        
        if current_content_delta_from_chunk:
            if state["had_any_reasoning"] and not state.get("reasoning_finish_event_sent"):
                if state["accumulated_reasoning"]:
                    processed_r = strip_potentially_harmful_html_and_normalize_newlines(state["accumulated_reasoning"])
                    if processed_r: yield {"type": "reasoning", "text": processed_r, "timestamp": get_current_time_iso()}
                    state["accumulated_reasoning"] = ""
                yield {"type": "reasoning_finish", "timestamp": get_current_time_iso()}
                state["reasoning_finish_event_sent"] = True
            state["accumulated_content"] += current_content_delta_from_chunk
            if current_content_delta_from_chunk.strip(): state["had_any_content"] = True

        if state["accumulated_reasoning"] and \
           (len(state["accumulated_reasoning"]) >= MIN_REASONING_FLUSH_CHUNK_SIZE or \
            "\n" in state["accumulated_reasoning"] or \
            finish_reason or tool_calls_chunk or \
            (state["had_any_content"] and not state["in_think_tag_mode"])):
            
            processed_text = strip_potentially_harmful_html_and_normalize_newlines(state["accumulated_reasoning"])
            if processed_text:
                yield {"type": "reasoning", "text": processed_text, "timestamp": get_current_time_iso()}
            state["accumulated_reasoning"] = ""
            if state["had_any_reasoning"] and not state.get("reasoning_finish_event_sent"):
                if finish_reason or tool_calls_chunk or (state["had_any_content"] and not state["in_think_tag_mode"]):
                    yield {"type": "reasoning_finish", "timestamp": get_current_time_iso()}
                    state["reasoning_finish_event_sent"] = True
        
        if state["accumulated_content"] and \
           (len(state["accumulated_content"]) >= MIN_CONTENT_FLUSH_CHUNK_SIZE or \
            "\n" in state["accumulated_content"] or \
            finish_reason or tool_calls_chunk):
            
            if state.get("had_any_reasoning") and not state.get("reasoning_finish_event_sent"):
                yield {"type": "reasoning_finish", "timestamp": get_current_time_iso()}
                state["reasoning_finish_event_sent"] = True
                
            processed_text = strip_potentially_harmful_html_and_normalize_newlines(state["accumulated_content"])
            if processed_text:
                yield {"type": "content", "text": processed_text, "timestamp": get_current_time_iso()}
            state["accumulated_content"] = ""
        
        if tool_calls_chunk or finish_reason:
            if state["accumulated_reasoning"]:
                processed_r = strip_potentially_harmful_html_and_normalize_newlines(state["accumulated_reasoning"])
                if processed_r: yield {"type": "reasoning", "text": processed_r, "timestamp": get_current_time_iso()}
                state["accumulated_reasoning"] = ""
            if state["accumulated_content"]:
                processed_c = strip_potentially_harmful_html_and_normalize_newlines(state["accumulated_content"])
                if processed_c: yield {"type": "content", "text": processed_c, "timestamp": get_current_time_iso()}
                state["accumulated_content"] = ""
            
            if state.get("had_any_reasoning") and not state.get("reasoning_finish_event_sent"):
                yield {"type": "reasoning_finish", "timestamp": get_current_time_iso()}
                state["reasoning_finish_event_sent"] = True
            
            if tool_calls_chunk:
                yield {"type": "tool_calls_chunk", "data": tool_calls_chunk, "timestamp": get_current_time_iso()}
                state["had_any_content_or_tool_call"] = True
            
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
    
    # 检查是否是通过OpenAI兼容接口使用的Gemini模型
    is_openai_gemini = not is_google_like_path and "gemini" in request_data.model.lower()
    
    if request_data.force_custom_reasoning_prompt:
        logger.info(f"{log_prefix}: Custom separator logic FORCED by request.")
        return True
    if (is_google_like_path or is_openai_gemini) and is_native_thinking_active:
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
        if processed_c:
            logger.info(f"{log_prefix}: Cleanup: Flushing remaining content: '{processed_c[:100]}...'")
            yield orjson_dumps_bytes_wrapper(AppStreamEventPy(type="content", text=processed_c, timestamp=get_current_time_iso()).model_dump(by_alias=True, exclude_none=True))
    
    if use_old_custom_separator_logic and state.get("accumulated_text_custom","").strip() and upstream_ok:
        pass

    if not upstream_ok and not state.get("final_finish_event_sent_by_llm_reason") and not state.get("final_finish_event_sent_flag_for_cleanup", False):
        yield orjson_dumps_bytes_wrapper(AppStreamEventPy(type="finish", reason="upstream_error_or_connection_failed", timestamp=get_current_time_iso()).model_dump(by_alias=True, exclude_none=True))
    elif not state.get("final_finish_event_sent_by_llm_reason") and not state.get("final_finish_event_sent_flag_for_cleanup", False):
        final_reason = state.get("final_finish_reason_from_llm", "stream_end")
        yield orjson_dumps_bytes_wrapper(AppStreamEventPy(type="finish", reason=final_reason, timestamp=get_current_time_iso()).model_dump(by_alias=True, exclude_none=True))