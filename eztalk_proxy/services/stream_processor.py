import logging
import asyncio
import httpx
import re
from typing import Dict, Any, AsyncGenerator, Tuple, Optional

from ..models.api_models import AppStreamEventPy, ChatRequestModel
from ..utils.helpers import (
    get_current_time_iso,
    orjson_dumps_bytes_wrapper
)
logger = logging.getLogger("EzTalkProxy.StreamProcessors")

MIN_REASONING_FLUSH_CHUNK_SIZE = 1
MIN_CONTENT_FLUSH_CHUNK_SIZE = 5  # 降低阈值，确保更小的内容块也能及时发送

def preprocess_ai_output_content(content: str, request_id: str) -> str:
    """
    直接返回原生AI输出内容，不做任何处理
    """
    log_prefix = f"RID-{request_id}"
    logger.debug(f"{log_prefix}: Returning native AI output without preprocessing")
    return content

def postprocess_ai_output_chunk(chunk: str, accumulated_content: str, request_id: str) -> str:
    """
    直接返回原生AI输出块，不做任何后处理
    """
    log_prefix = f"RID-{request_id}"
    logger.debug(f"{log_prefix}: Returning native AI output chunk without postprocessing")
    return chunk

def extract_think_tags(text: str) -> Tuple[str, str]:
    """
    从文本中提取<think>标签内容
    
    Args:
        text: 原始文本
        
    Returns:
        Tuple[思考内容, 剩余内容]
    """
    if not text or '<think>' not in text:
        return "", text
    
    # 匹配所有<think>...</think>标签（支持多个标签和换行）
    think_pattern = r'<think>(.*?)</think>'
    matches = re.findall(think_pattern, text, re.DOTALL | re.IGNORECASE)
    
    if not matches:
        return "", text
    
    # 提取所有思考内容
    thinking_content = "\n\n".join(match.strip() for match in matches if match.strip())
    
    # 移除所有<think>标签及其内容，得到剩余内容
    remaining_content = re.sub(think_pattern, '', text, flags=re.DOTALL | re.IGNORECASE)
    remaining_content = remaining_content.strip()
    
    return thinking_content, remaining_content

def should_extract_think_tags_from_content(request_data, request_id: str) -> bool:
    """
    判断是否应该从content中提取<think>标签
    主要针对DeepSeek等在content中包含<think>标签的模型
    """
    log_prefix = f"RID-{request_id}"
    
    if not hasattr(request_data, 'model') or not request_data.model:
        return False
        
    model_lower = request_data.model.lower()
    
    # DeepSeek模型经常在content中使用<think>标签
    if "deepseek" in model_lower:
        logger.info(f"{log_prefix}: Enabling <think> tag extraction for DeepSeek model: {request_data.model}")
        return True
    
    # MKE提供商也可能使用这种格式
    if hasattr(request_data, 'provider') and request_data.provider and request_data.provider.lower() == "mke":
        logger.info(f"{log_prefix}: Enabling <think> tag extraction for MKE provider")
        return True
    
    # 其他可能使用<think>标签的模型可以在这里添加
    think_tag_models = ['qwen', 'claude', 'gpt']  # 一些可能使用思考标签的模型
    for model_keyword in think_tag_models:
        if model_keyword in model_lower:
            logger.info(f"{log_prefix}: Enabling <think> tag extraction for model containing '{model_keyword}': {request_data.model}")
            return True
    
    return False

def is_excessive_whitespace(text: str) -> bool:
    """
    智能检查文本是否只包含过多的空白字符
    针对OpenAI兼容接口经常产生大量空白段落的问题
    """
    if not text:
        return True

    # 计算有效内容比例
    total_chars = len(text)
    whitespace_chars = text.count(' ') + text.count('\t') + text.count('\n') + text.count('\r')
    content_chars = total_chars - whitespace_chars

    # 如果有效内容少于10%，认为是过多空白
    if total_chars > 0 and content_chars / total_chars < 0.1:
        return True

    # 如果只包含单个空格或制表符，跳过
    if text.strip() == '':
        return True

    # 如果只包含换行符，跳过
    if text.replace('\n', '').replace('\r', '') == '':
        return True

    # 如果包含过多连续换行符（超过4个），且没有其他内容，跳过
    if text.count('\n') > 4 and len(text.replace('\n', '').replace(' ', '').replace('\t', '')) == 0:
        return True

    # 检查是否只包含重复的空白模式
    stripped = text.strip()
    if stripped and len(set(stripped)) <= 2 and all(c in ' \t\n\r' for c in stripped):
        return True

    return False

def should_skip_content_chunk(content_str: str, accumulated_content: str) -> bool:
    """
    直接传递所有AI输出内容，不做任何过滤
    """
    return False

def is_meaningful_content_chunk(content_str: str, min_length: int = 1) -> bool:
    """
    所有内容都被认为是有意义的，直接传递原生AI输出
    """
    return True

def optimize_content_chunking(content_str: str, accumulated_content: str) -> str:
    """
    直接返回原生AI输出内容，不做任何优化
    """
    return content_str

async def process_openai_like_sse_stream(
    parsed_sse_data: Dict[str, Any],
    current_processing_state: Dict[str, Any],
    request_id: str,
    request_data = None
) -> AsyncGenerator[Dict[str, Any], None]:
    log_prefix = f"RID-{request_id}"
    
    state = current_processing_state
    state.setdefault("had_any_reasoning", False)
    state.setdefault("reasoning_finish_event_sent", False)
    state.setdefault("final_finish_event_sent_by_llm_reason", False)
    state.setdefault("accumulated_content", "")
    state.setdefault("accumulated_thinking", "")  # 累积的思考内容
    state.setdefault("total_chunks_processed", 0)
    state.setdefault("content_chunks_received", 0)
    state.setdefault("reasoning_chunks_received", 0)
    state.setdefault("thinking_chunks_received", 0)  # 从<think>标签提取的思考块数量
    state.setdefault("ai_raw_output_log", [])  # 记录AI原始输出
    
    # 检查是否需要从content中提取<think>标签
    extract_think_tags_enabled = request_data and should_extract_think_tags_from_content(request_data, request_id)
    if extract_think_tags_enabled:
        logger.info(f"{log_prefix}: <think> tag extraction enabled for model: {request_data.model}")

    choices_count = len(parsed_sse_data.get('choices', []))
    if choices_count > 0:
        logger.info(f"{log_prefix}: process_openai_like_sse_stream: Processing SSE data with {choices_count} choices")

    for choice in parsed_sse_data.get('choices', []):
        delta = choice.get('delta', {})
        finish_reason = choice.get("finish_reason")

        reasoning_chunk = delta.get("reasoning_content")
        content_chunk = delta.get("content")
        tool_calls_chunk = delta.get("tool_calls")
        
        state["total_chunks_processed"] += 1

        # Send reasoning chunk immediately if it exists
        if reasoning_chunk:
            state["reasoning_chunks_received"] += 1
            logger.info(f"{log_prefix}: process_openai_like_sse_stream: Received reasoning chunk #{state['reasoning_chunks_received']} ({len(str(reasoning_chunk))} chars)")
            if reasoning_chunk:
                yield {"type": "reasoning", "text": str(reasoning_chunk), "timestamp": get_current_time_iso()}
                state["had_any_reasoning"] = True

        # Accumulate content and flush when it reaches a certain size
        if content_chunk is not None:
            # Convert to string and check if it's not just None or empty
            content_str = str(content_chunk)
            state["content_chunks_received"] += 1
            
            # 记录AI原始输出用于调试空白问题
            raw_output_record = {
                "chunk_id": state["content_chunks_received"],
                "raw_content": content_str,
                "length": len(content_str),
                "repr": repr(content_str),  # 显示转义字符
                "whitespace_analysis": {
                    "newlines": content_str.count('\n'),
                    "spaces": content_str.count(' '),
                    "tabs": content_str.count('\t'),
                    "is_whitespace_only": content_str.strip() == '',
                    "consecutive_newlines": len([m for m in re.finditer(r'\n{2,}', content_str)])
                }
            }
            state["ai_raw_output_log"].append(raw_output_record)
            
            logger.info(f"{log_prefix}: AI_RAW_OUTPUT chunk #{state['content_chunks_received']}: length={len(content_str)}, content={repr(content_str[:100])}")
            logger.info(f"{log_prefix}: AI_RAW_OUTPUT whitespace: newlines={raw_output_record['whitespace_analysis']['newlines']}, spaces={raw_output_record['whitespace_analysis']['spaces']}, consecutive_newlines={raw_output_record['whitespace_analysis']['consecutive_newlines']}")
            
            # 检查是否需要从content中提取<think>标签
            if extract_think_tags_enabled and content_str:
                thinking_content = ""
                actual_content = content_str
                
                # 尝试从当前chunk中提取<think>标签
                if '<think>' in content_str:
                    extracted_thinking, remaining_content = extract_think_tags(content_str)
                    if extracted_thinking:
                        thinking_content = extracted_thinking
                        actual_content = remaining_content
                        logger.info(f"{log_prefix}: THINK_TAG_EXTRACTION: Found thinking content in chunk #{state['content_chunks_received']}: {len(thinking_content)} chars")
                
                # 如果有思考内容，立即发送
                if thinking_content:
                    state["thinking_chunks_received"] += 1
                    state["accumulated_thinking"] += thinking_content
                    logger.info(f"{log_prefix}: REASONING_FROM_THINK_TAG: Sending thinking chunk #{state['thinking_chunks_received']} ({len(thinking_content)} chars)")
                    yield {"type": "reasoning", "text": thinking_content, "timestamp": get_current_time_iso()}
                    state["had_any_reasoning"] = True
                
                # 使用提取后的实际内容进行后续处理
                content_str = actual_content
            
            # 直接处理原生AI输出内容
            if content_str:
                # 记录变化前的状态
                old_length = len(state["accumulated_content"])
                
                # 直接使用原生内容，不做任何处理
                state["accumulated_content"] += content_str
                new_length = len(state["accumulated_content"])
                
                # 记录日志
                logger.info(f"{log_prefix}: NATIVE_CONTENT_ACCUMULATION: {old_length} -> {new_length} chars")
                logger.debug(f"{log_prefix}: Using native AI output without processing")
                
                # If we receive content and haven't sent reasoning_finish, send it now.
                if state["had_any_reasoning"] and not state["reasoning_finish_event_sent"]:
                    logger.info(f"{log_prefix}: process_openai_like_sse_stream: Sending reasoning_finish before content")
                    yield {"type": "reasoning_finish", "timestamp": get_current_time_iso()}
                    state["reasoning_finish_event_sent"] = True
                
                # 降低刷新条件：内容达到阈值或包含完整句子时立即刷新
                should_flush = (
                    len(state["accumulated_content"]) >= MIN_CONTENT_FLUSH_CHUNK_SIZE or
                    # 包含句子结束符号时立即刷新
                    any(punct in state["accumulated_content"] for punct in ['.', '!', '?', '。', '！', '？']) or
                    # 包含代码块结束时立即刷新
                    '```' in state["accumulated_content"] or
                    # 包含数学公式结束时立即刷新
                    ('$' in state["accumulated_content"] and state["accumulated_content"].count('$') % 2 == 0)
                )
                
                if should_flush:
                    # 直接使用原生AI输出，不做任何清理
                    native_content = state["accumulated_content"]
                    
                    logger.info(f"{log_prefix}: NATIVE_CONTENT_FLUSH: Sending native AI output ({len(native_content)} chars)")
                    logger.debug(f"{log_prefix}: Native content preview: {repr(native_content[:200])}")
                    
                    yield {"type": "content", "text": native_content, "timestamp": get_current_time_iso()}
                    state["accumulated_content"] = ""
            else:
                if not content_str:
                    logger.info(f"{log_prefix}: SKIPPED_CONTENT: Empty content chunk")
                else:
                    logger.info(f"{log_prefix}: SKIPPED_CONTENT: Excessive whitespace chunk - {repr(content_str[:100])}")

        # Handle tool calls and finish reason
        if tool_calls_chunk or finish_reason:
            logger.info(f"{log_prefix}: process_openai_like_sse_stream: Handling finish/tool_calls - tool_calls: {bool(tool_calls_chunk)}, finish_reason: {finish_reason}")
            
            # Before handling finish or tool calls, flush any remaining content.
            if state["accumulated_content"]:
                logger.info(f"{log_prefix}: FINAL_CONTENT_FLUSH: Flushing remaining content ({len(state['accumulated_content'])} chars)")
                logger.info(f"{log_prefix}: FINAL_CONTENT_PREVIEW: {repr(state['accumulated_content'][:200])}")
                
                # 直接使用原生AI输出，不做最终处理
                native_final_content = state["accumulated_content"]
                logger.info(f"{log_prefix}: FINAL_NATIVE_CONTENT_FLUSH: Sending final native AI output ({len(native_final_content)} chars)")
                yield {"type": "content", "text": native_final_content, "timestamp": get_current_time_iso()}
                state["accumulated_content"] = ""

            # If there was any reasoning, ensure the finish event is sent before the final block.
            if state["had_any_reasoning"] and not state["reasoning_finish_event_sent"]:
                logger.info(f"{log_prefix}: process_openai_like_sse_stream: Sending final reasoning_finish")
                yield {"type": "reasoning_finish", "timestamp": get_current_time_iso()}
                state["reasoning_finish_event_sent"] = True
            if tool_calls_chunk:
                logger.info(f"{log_prefix}: process_openai_like_sse_stream: Processing tool_calls_chunk")
                yield {"type": "tool_calls_chunk", "data": tool_calls_chunk, "timestamp": get_current_time_iso()}
            
            if finish_reason:
                logger.info(f"{log_prefix}: OpenAI-like choice finish_reason: {finish_reason}.")
                logger.info(f"{log_prefix}: process_openai_like_sse_stream: Stream summary - total chunks: {state['total_chunks_processed']}, content chunks: {state['content_chunks_received']}, reasoning chunks: {state['reasoning_chunks_received']}")
                
                # 输出AI原始输出汇总分析
                if state["ai_raw_output_log"]:
                    total_raw_length = sum(record["length"] for record in state["ai_raw_output_log"])
                    total_newlines = sum(record["whitespace_analysis"]["newlines"] for record in state["ai_raw_output_log"])
                    total_consecutive_newlines = sum(record["whitespace_analysis"]["consecutive_newlines"] for record in state["ai_raw_output_log"])
                    whitespace_only_chunks = sum(1 for record in state["ai_raw_output_log"] if record["whitespace_analysis"]["is_whitespace_only"])
                    
                    logger.info(f"{log_prefix}: AI_OUTPUT_SUMMARY: total_raw_length={total_raw_length}, total_newlines={total_newlines}, consecutive_newlines_groups={total_consecutive_newlines}, whitespace_only_chunks={whitespace_only_chunks}")
                    
                    # 显示前几个包含大量空白的chunk
                    problematic_chunks = [record for record in state["ai_raw_output_log"] if record["whitespace_analysis"]["consecutive_newlines"] > 0 or record["whitespace_analysis"]["is_whitespace_only"]]
                    if problematic_chunks:
                        logger.info(f"{log_prefix}: PROBLEMATIC_CHUNKS found {len(problematic_chunks)} chunks with excessive whitespace:")
                        for i, record in enumerate(problematic_chunks[:5]):  # 只显示前5个问题chunk
                            logger.info(f"{log_prefix}: PROBLEMATIC_CHUNK #{record['chunk_id']}: {record['repr']}")
                    else:
                        logger.info(f"{log_prefix}: AI_OUTPUT_ANALYSIS: No problematic chunks detected - AI output appears clean")
                
                yield {"type": "finish", "reason": finish_reason, "timestamp": get_current_time_iso()}
                state["final_finish_event_sent_by_llm_reason"] = True

def should_apply_custom_separator_logic(
    request_data: ChatRequestModel,
    request_id: str,
    is_google_like_path: bool,
    is_native_thinking_active: bool
) -> bool:
    log_prefix = f"RID-{request_id}"
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
        # 直接使用原生AI输出，不做最终预处理
        native_remaining_content = accumulated_content
        logger.info(f"{log_prefix}: Cleanup: Flushing remaining native content: '{native_remaining_content[:100]}...'")
        yield orjson_dumps_bytes_wrapper(AppStreamEventPy(type="content", text=native_remaining_content, timestamp=get_current_time_iso()).model_dump(by_alias=True, exclude_none=True))

    # Send the final finish event if it hasn't been sent already by a finish_reason from the LLM.
    if not state.get("final_finish_event_sent_by_llm_reason"):
        final_reason = "stream_end"
        if not upstream_ok:
            final_reason = "upstream_error_or_connection_failed"
        
        logger.info(f"{log_prefix}: Cleanup: Sending final finish event with reason '{final_reason}'.")
        yield orjson_dumps_bytes_wrapper(AppStreamEventPy(type="finish", reason=final_reason, timestamp=get_current_time_iso()).model_dump(by_alias=True, exclude_none=True))