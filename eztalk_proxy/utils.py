# eztalk_proxy/utils.py

import orjson
import re
import logging
import datetime # 在 get_current_time_iso 中使用 datetime.datetime
from typing import Any, Dict, List, Tuple, Optional

from fastapi.responses import JSONResponse

# 假设您的 config.py 文件与 utils.py 在同一目录下或可通过相对路径访问
# 并定义了 COMMON_HEADERS 和 MAX_SSE_LINE_LENGTH
from .config import COMMON_HEADERS, MAX_SSE_LINE_LENGTH

logger = logging.getLogger("EzTalkProxy.Utils")


def orjson_dumps_bytes_wrapper(data: Any) -> bytes:
    """
    使用 orjson 将数据序列化为字节串。
    选项:
    - OPT_NON_STR_KEYS: 允许非字符串键（如果您的数据结构需要）。
    - OPT_PASSTHROUGH_DATETIME: datetime 对象将按原样传递（通常与 OPT_NAIVE_UTC 或 OPT_UTC_Z 结合使用，
                                但这里可能依赖 orjson 的默认行为或后续处理）。
    - OPT_APPEND_NEWLINE: 在末尾附加换行符，适用于 SSE。
    """
    return orjson.dumps(
        data,
        option=orjson.OPT_NON_STR_KEYS | orjson.OPT_PASSTHROUGH_DATETIME | orjson.OPT_APPEND_NEWLINE
    )

def error_response(
    code: int,
    msg: str,
    request_id: Optional[str] = None,
    headers: Optional[Dict[str, str]] = None
) -> JSONResponse:
    """
    创建并记录一个标准的错误 JSON响应。
    """
    log_msg = f"错误 {code}: {msg}"
    if request_id:
        log_msg = f"RID-{request_id}: {log_msg}"
    logger.warning(log_msg)
    
    final_headers = {**COMMON_HEADERS, **(headers or {})}
    
    return JSONResponse(
        status_code=code,
        content={"error": {"message": msg, "code": code, "type": "proxy_error"}},
        headers=final_headers
    )

def strip_potentially_harmful_html_and_normalize_newlines(text: str) -> str:
    """
    清理文本：移除潜在有害的HTML标签（script, style），转换<br>, <p>标签为换行，
    并规范化连续的换行符和行首尾空格。
    """
    if not isinstance(text, str):
        # 如果输入不是字符串（例如 None），返回空字符串以避免后续处理错误
        return ""
    
    current_logger = logging.getLogger("EzTalkProxy.SPHASANN") # Specific logger for this function
    current_logger.debug(f"Input (first 200 chars): '{text[:200]}'")

    # 步骤 1: 移除 <script> 和 <style> 标签及其内容
    text_before_script_style_strip = text
    text = re.sub(r"<script[^>]*>.*?</script>|<style[^>]*>.*?</style>", "", text, flags=re.IGNORECASE | re.DOTALL)
    if text != text_before_script_style_strip:
        current_logger.debug(f"SPHASANN Step 1 (script/style strip): Applied. Text (first 200 chars): '{text[:200]}'")
    
    # 步骤 2: HTML <br> 和 <p> 标签规范化为换行符
    text_before_html_br_p_norm = text
    text = re.sub(r"<br\s*/?>", "\n", text, flags=re.IGNORECASE)      # <br> 替换为 \n
    text = re.sub(r"</p\s*>", "\n\n", text, flags=re.IGNORECASE)    # </p> 替换为 \n\n
    text = re.sub(r"<p[^>]*>", "", text, flags=re.IGNORECASE)        # 移除 <p>起始标签
    if text != text_before_html_br_p_norm:
        current_logger.debug(f"SPHASANN Step 2 (HTML br/p norm): Applied. Text (first 200 chars): '{text[:200]}'")

    # 步骤 3: 处理特定分隔符前缀，确保其前后有适当的换行
    separator_prefix_pattern_regex = r"\s*(---###)" # 匹配可选的前导空格和 "---###"
    text_before_prefix_sep_processing = text
    # 在 "---###" 前强制添加两个换行符，移除其前导空格
    text = re.sub(separator_prefix_pattern_regex, r"\n\n\1", text) 
    if text != text_before_prefix_sep_processing:
        current_logger.debug(f"SPHASANN Step 3 (---### prefix normalization): Applied. Text (first 200 chars): '{text[:200]}'")

    # 步骤 4: 合并多个连续换行符为一个或两个换行符
    text_before_collapse_newlines = text
    text = re.sub(r"\n{3,}", "\n\n", text) # 3个或更多 \n 替换为 \n\n
    if text != text_before_collapse_newlines:
        current_logger.debug(f"SPHASANN Step 4 (collapse \\n{{3,}} to \\n\\n): Applied. Text (first 200 chars): '{text[:200]}'")

    # 步骤 5: 去除每行的首尾空格
    lines = text.split('\n')
    stripped_lines = [line.strip() for line in lines]
    text_after_line_stripping = "\n".join(stripped_lines)
    if text != text_after_line_stripping: 
        current_logger.debug(f"SPHASANN Step 5 (line stripping & rejoin): Applied. Text (first 200 chars): '{text_after_line_stripping[:200]}'")
    text = text_after_line_stripping
    
    # 最终文本
    final_text = text 
    current_logger.debug(f"SPHASANN Final output (first 200 chars): '{final_text[:200]}'")
    return final_text


def extract_sse_lines(buffer: bytearray) -> Tuple[List[bytes], bytearray]:
    """
    从字节缓冲区中提取所有完整的SSE（Server-Sent Events）行。
    处理 '\n' 分隔的行，并移除可选的 '\r'。
    如果行长度超过 MAX_SSE_LINE_LENGTH，则跳过该行并记录警告。
    """
    lines: List[bytes] = []
    start_index: int = 0
    buffer_len = len(buffer)

    while start_index < buffer_len:
        newline_index = buffer.find(b'\n', start_index)
        
        if newline_index == -1:
            # 没有找到换行符，剩余部分是不完整的行
            break
        
        # 提取行，不包括换行符
        line = buffer[start_index:newline_index]
        
        # 移除行尾可能存在的回车符 '\r'
        if line.endswith(b'\r'):
            line = line[:-1]
            
        if len(line) > MAX_SSE_LINE_LENGTH:
            logger.warning(
                f"SSE line too long ({len(line)} bytes), exceeded MAX_SSE_LINE_LENGTH ({MAX_SSE_LINE_LENGTH}). Line skipped. "
                f"Content start: {line[:100]!r}"
            )
        else:
            lines.append(line)
            
        start_index = newline_index + 1 # 移动到下一行的开始
        
    return lines, buffer[start_index:] # 返回提取的行列表和缓冲区中剩余的部分


def get_current_time_iso() -> str:
    """
    获取当前 UTC 时间的 ISO 8601 格式字符串，并以 'Z' 结尾表示 UTC。
    """
    return datetime.datetime.utcnow().isoformat() + "Z"

def is_gemini_2_5_model(model_name: str) -> bool:
    """
    简单判断模型名称是否包含 "gemini-2.5"，用于识别 Gemini 2.5 系列模型。
    不区分大小写。
    """
    if not isinstance(model_name, str): # 防御性检查，确保 model_name 是字符串
        return False
    return "gemini-2.5" in model_name.lower()