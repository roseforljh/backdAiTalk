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
from .format_repair import format_repair_service
logger = logging.getLogger("EzTalkProxy.StreamProcessors")

MIN_REASONING_FLUSH_CHUNK_SIZE = 1
MIN_CONTENT_FLUSH_CHUNK_SIZE = 1  # 🎯 紧急修复：降低到最小值，避免内容丢失

# 简单判断当前是否处于未闭合的代码围栏中（``` 计数为奇数）
def _inside_code_fence(text: str) -> bool:
    if not text:
        return False
    return text.count('```') % 2 == 1

# Markdown 块完整性检测器
class MarkdownBlockDetector:
    """检测 Markdown 块的完整性，优化流式输出时机"""
    
    @staticmethod
    def is_complete_code_block(text: str) -> bool:
        """检测代码块是否完整"""
        if not text.strip():
            return False
        lines = text.split('\n')
        fence_count = 0
        for line in lines:
            if line.strip().startswith('```'):
                fence_count += 1
        return fence_count >= 2 and fence_count % 2 == 0
    
    @staticmethod
    def is_complete_math_block(text: str) -> bool:
        """检测数学块是否完整"""
        if not text.strip():
            return False
        # 检测 $$ 围栏
        double_dollar_count = text.count('$$')
        if double_dollar_count >= 2 and double_dollar_count % 2 == 0:
            return True
        # 检测 \[ \] 围栏
        if text.count('\\[') > 0 and text.count('\\]') > 0:
            return text.count('\\[') == text.count('\\]')
        return False
    
    @staticmethod
    def is_complete_table_block(text: str) -> bool:
        """检测表格是否完整"""
        if not text.strip():
            return False
        lines = [line.strip() for line in text.split('\n') if line.strip()]
        if len(lines) < 2:
            return False
        # 检查是否有表格分隔行
        has_separator = any('---' in line and '|' in line for line in lines)
        # 检查是否所有行都包含 |
        all_have_pipes = all('|' in line for line in lines)
        return has_separator and all_have_pipes
    
    @staticmethod
    def is_complete_list_item(text: str) -> bool:
        """检测列表项是否完整"""
        if not text.strip():
            return False
        # 简单检测：以列表标记开始且包含完整内容
        stripped = text.strip()
        list_markers = ['* ', '- ', '+ '] + [f'{i}. ' for i in range(1, 10)]
        starts_with_marker = any(stripped.startswith(marker) for marker in list_markers)
        # 如果以换行符结束或包含句号，认为是完整的
        ends_properly = text.endswith('\n') or text.endswith('.') or text.endswith('。')
        return starts_with_marker and (ends_properly or len(stripped) > 20)
    
    @staticmethod
    def is_safe_flush_point(accumulated_content: str, new_chunk: str) -> bool:
        """判断是否是安全的刷新点"""
        # 🎯 紧急修复：简化逻辑，减少过度过滤
        full_content = accumulated_content + new_chunk
        
        # 只检查最关键的情况：代码围栏内部不安全
        if _inside_code_fence(full_content):
            return False
        
        # 其他情况一律认为安全，确保内容不丢失
        return True
    
    @staticmethod
    def detect_block_type(text: str) -> str:
        """检测块类型"""
        if not text.strip():
            return "text"
        
        text_lower = text.lower().strip()
        
        if text_lower.startswith('```'):
            return "code_block"
        elif '$$' in text or '\\[' in text:
            return "math_block"
        elif '|' in text and ('---' in text or text.count('|') > 2):
            return "table"
        elif text_lower.startswith(('* ', '- ', '+ ')) or any(text_lower.startswith(f'{i}. ') for i in range(1, 10)):
            return "list"
        elif text_lower.startswith('#'):
            return "heading"
        else:
            return "text"

# 初始化块检测器
block_detector = MarkdownBlockDetector()

def preprocess_ai_output_content(content: str, request_id: str) -> str:
    """
    对AI输出内容进行格式修复预处理
    """
    log_prefix = f"RID-{request_id}"
    
    if not content or not content.strip():
        return content
    
    try:
        # 应用格式修复
        repaired_content = format_repair_service.repair_ai_output(content, "general")
        
        if repaired_content != content:
            logger.debug(f"{log_prefix}: Applied format repair to content ({len(content)} -> {len(repaired_content)} chars)")
        
        return repaired_content
    except Exception as e:
        logger.warning(f"{log_prefix}: Format repair failed: {e}, returning original content")
        return content

def postprocess_ai_output_chunk(chunk: str, accumulated_content: str, request_id: str) -> str:
    """
    对AI输出块进行格式修复后处理
    """
    log_prefix = f"RID-{request_id}"
    
    if not chunk or not chunk.strip():
        return chunk
    
    try:
        # 对于代码块相关的内容，应用格式修复
        if '```' in chunk or '`' in chunk:
            repaired_chunk = format_repair_service.repair_ai_output(chunk, "code")
            
            if repaired_chunk != chunk:
                logger.debug(f"{log_prefix}: Applied code format repair to chunk")
            
            return repaired_chunk
        
        # 对于其他内容，应用通用格式修复
        repaired_chunk = format_repair_service.repair_ai_output(chunk, "general")
        return repaired_chunk
        
    except Exception as e:
        logger.warning(f"{log_prefix}: Chunk format repair failed: {e}, returning original chunk")
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
            
            # 处理AI输出内容，并进行更保守的流式处理
            if content_str:
                # 记录变化前的状态
                old_length = len(state["accumulated_content"])
                
                # 累积内容
                state["accumulated_content"] += content_str
                new_length = len(state["accumulated_content"])
                
                # 记录日志
                logger.info(f"{log_prefix}: CONTENT_ACCUMULATION: {old_length} -> {new_length} chars")
                
                # If we receive content and haven't sent reasoning_finish, send it now.
                if state["had_any_reasoning"] and not state["reasoning_finish_event_sent"]:
                    logger.info(f"{log_prefix}: process_openai_like_sse_stream: Sending reasoning_finish before content")
                    yield {"type": "reasoning_finish", "timestamp": get_current_time_iso()}
                    state["reasoning_finish_event_sent"] = True
                
                # 增强的刷新条件：使用 Markdown 块完整性检测
                accumulated_content = state["accumulated_content"]
                
                # 🎯 紧急修复：简化刷新条件，确保所有内容都被发送
                # 直接刷新任何非空内容，不进行复杂的完整性检查
                should_flush = len(accumulated_content) > 0
                
                # 检测当前块类型
                current_block_type = block_detector.detect_block_type(accumulated_content)
                
                logger.debug(f"{log_prefix}: FLUSH_DECISION: should_flush={should_flush}, block_type={current_block_type}, accumulated_len={len(accumulated_content)}")

                if should_flush:
                    # 根据配置决定是否在流式阶段进行修复
                    if getattr(format_repair_service, 'config', None) and not format_repair_service.config.enable_realtime_repair:
                        content_to_send = accumulated_content
                    else:
                        content_to_send = format_repair_service.repair_ai_output(accumulated_content, "general")
                    
                    logger.info(f"{log_prefix}: CONTENT_FLUSH: Sending {'raw' if content_to_send is accumulated_content else 'repaired'} content ({len(content_to_send)} chars, block_type={current_block_type})")
                    logger.debug(f"{log_prefix}: Content preview: {repr(content_to_send[:200])}")
                    
                    output_type = current_block_type if current_block_type != "text" else format_repair_service.detect_output_type(content_to_send)
                    yield {"type": "content", "text": content_to_send, "output_type": output_type, "block_type": current_block_type, "timestamp": get_current_time_iso()}
                    state["accumulated_content"] = ""
                    content_str = "" # 标记为已处理
            else:
                if not content_str:
                    logger.info(f"{log_prefix}: SKIPPED_CONTENT: Empty content chunk")
                else:
                    logger.info(f"{log_prefix}: SKIPPED_CONTENT: Excessive whitespace chunk - {repr(content_str[:100])}")

        # Handle tool calls and finish reason
        if tool_calls_chunk or finish_reason:
            logger.info(f"{log_prefix}: process_openai_like_sse_stream: Handling finish/tool_calls - tool_calls: {bool(tool_calls_chunk)}, finish_reason: {finish_reason}")
            
            # Before handling finish or tool calls, flush any remaining content (only if not already sent).
            if state["accumulated_content"] and not state.get("final_content_sent", False):
                logger.info(f"{log_prefix}: FINAL_CONTENT_FLUSH: Flushing remaining content ({len(state['accumulated_content'])} chars)")
                logger.info(f"{log_prefix}: FINAL_CONTENT_PREVIEW: {repr(state['accumulated_content'][:200])}")
                
                # 对最后剩余的内容进行格式修复（最终阶段允许完整修复）
                final_content_to_send = format_repair_service.repair_ai_output(state["accumulated_content"], "general")
                logger.info(f"{log_prefix}: FINAL_CONTENT_FLUSH: Sending final repaired content ({len(final_content_to_send)} chars)")
                output_type = format_repair_service.detect_output_type(final_content_to_send)
                # 🎯 修复：使用 content_final 类型发送最终内容块
                yield {"type": "content_final", "text": final_content_to_send, "output_type": output_type, "timestamp": get_current_time_iso()}
                state["accumulated_content"] = ""
                state["final_content_sent"] = True  # 标记最终内容已发送
            elif state["accumulated_content"]:
                logger.info(f"{log_prefix}: FINAL_CONTENT_SKIP: Content exists but final_content_sent=True, clearing to prevent cleanup duplication")
                state["accumulated_content"] = ""  # 清空以避免cleanup时重复

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
    
    if isinstance(error, httpx.TimeoutException): 
        error_message = "Request to LLM API timed out."
    elif isinstance(error, httpx.HTTPStatusError):
        # 处理HTTP状态码错误，提供友好的错误信息
        status_code = error.response.status_code
        if status_code == 429:
            error_message = "请求频率过高 (429 Too Many Requests)，请稍后重试。服务器暂时限制了请求频率。"
        elif status_code == 401:
            error_message = "身份验证失败 (401 Unauthorized)，请检查API密钥配置。"
        elif status_code == 403:
            error_message = "访问被拒绝 (403 Forbidden)，请检查权限设置或API配额。"
        elif status_code == 404:
            error_message = "服务端点未找到 (404 Not Found)，请检查API地址配置。"
        elif status_code == 500:
            error_message = "服务器内部错误 (500 Internal Server Error)，请稍后重试。"
        elif status_code == 502:
            error_message = "网关错误 (502 Bad Gateway)，服务器可能暂时不可用。"
        elif status_code == 503:
            error_message = "服务不可用 (503 Service Unavailable)，服务器正在维护中。"
        else:
            error_message = f"HTTP错误 {status_code}: {error.response.reason_phrase or 'Unknown error'}"
    elif isinstance(error, httpx.RequestError): 
        error_message = f"Network error: {error}"
    elif isinstance(error, asyncio.CancelledError): 
        logger.info(f"{log_prefix}: Stream cancelled."); return
    else: 
        error_message = f"Unexpected error: {str(error)[:200]}"
    
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

    # 🎯 修复：确保在清理阶段发送任何剩余的累积内容，防止内容丢失
    if accumulated_content:
        logger.info(f"{log_prefix}: Cleanup: Flushing remaining content ({len(accumulated_content)} chars)")
        logger.debug(f"{log_prefix}: Cleanup: Content preview: '{accumulated_content[:100]}...'")
        
        final_content_to_send = format_repair_service.repair_ai_output(accumulated_content, "general")
        output_type = format_repair_service.detect_output_type(final_content_to_send)

        # 🎯 使用 'content_final' 类型发送最后的内容块，以便客户端知道这是流的结尾
        yield orjson_dumps_bytes_wrapper(AppStreamEventPy(
            type="content_final",
            text=final_content_to_send,
            output_type=output_type,
            timestamp=get_current_time_iso()
        ).model_dump(by_alias=True, exclude_none=True))
        state["accumulated_content"] = ""
    else:
        logger.info(f"{log_prefix}: Cleanup: No remaining content to flush")

    # Send the final finish event if it hasn't been sent already by a finish_reason from the LLM.
    if not state.get("final_finish_event_sent_by_llm_reason"):
        final_reason = "stream_end"
        if not upstream_ok:
            final_reason = "upstream_error_or_connection_failed"
        
        logger.info(f"{log_prefix}: Cleanup: Sending final finish event with reason '{final_reason}'.")
        yield orjson_dumps_bytes_wrapper(AppStreamEventPy(type="finish", reason=final_reason, timestamp=get_current_time_iso()).model_dump(by_alias=True, exclude_none=True))