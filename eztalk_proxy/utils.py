# eztalk_proxy/utils.py

import orjson
import re
import logging
import datetime
from typing import Any, Dict, List, Tuple, Optional
import os # 新增导入
import shutil # 新增导入

from fastapi.responses import JSONResponse

from .config import ( # 确保从 config 导入新加的配置
    COMMON_HEADERS,
    MAX_SSE_LINE_LENGTH,
    TEMP_UPLOAD_DIR, # 新增
    SUPPORTED_DOCUMENT_MIME_TYPES_FOR_TEXT_EXTRACTION, # 新增
    MAX_DOCUMENT_CONTENT_CHARS_FOR_PROMPT # 新增
)

# 尝试导入文档处理库，如果失败则记录警告，相关功能将受限
try:
    import PyPDF2 # 用于 PDF
except ImportError:
    PyPDF2 = None
    logging.warning("PyPDF2 library not found. PDF text extraction will not be available.")

try:
    import docx # python-docx, 用于 DOCX
except ImportError:
    docx = None
    logging.warning("python-docx library not found. DOCX text extraction will not be available.")

# try:
#     import magic # python-magic, 用于更可靠的MIME类型检测 (可选)
# except ImportError:
#     magic = None
#     logging.warning("python-magic library not found. Advanced MIME type detection will rely on browser-provided types.")


logger = logging.getLogger("EzTalkProxy.Utils")


def orjson_dumps_bytes_wrapper(data: Any) -> bytes:
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
    if not isinstance(text, str):
        return ""
    current_logger = logging.getLogger("EzTalkProxy.SPHASANN")
    current_logger.debug(f"Input (first 200 chars): '{text[:200]}'")
    text_before_script_style_strip = text
    text = re.sub(r"<script[^>]*>.*?</script>|<style[^>]*>.*?</style>", "", text, flags=re.IGNORECASE | re.DOTALL)
    if text != text_before_script_style_strip:
        current_logger.debug(f"SPHASANN Step 1 (script/style strip): Applied. Text (first 200 chars): '{text[:200]}'")
    text_before_html_br_p_norm = text
    text = re.sub(r"<br\s*/?>", "\n", text, flags=re.IGNORECASE)
    text = re.sub(r"</p\s*>", "\n\n", text, flags=re.IGNORECASE)
    text = re.sub(r"<p[^>]*>", "", text, flags=re.IGNORECASE)
    if text != text_before_html_br_p_norm:
        current_logger.debug(f"SPHASANN Step 2 (HTML br/p norm): Applied. Text (first 200 chars): '{text[:200]}'")
    separator_prefix_pattern_regex = r"\s*(---###)"
    text_before_prefix_sep_processing = text
    text = re.sub(separator_prefix_pattern_regex, r"\n\n\1", text) 
    if text != text_before_prefix_sep_processing:
        current_logger.debug(f"SPHASANN Step 3 (---### prefix normalization): Applied. Text (first 200 chars): '{text[:200]}'")
    text_before_collapse_newlines = text
    text = re.sub(r"\n{3,}", "\n\n", text)
    if text != text_before_collapse_newlines:
        current_logger.debug(f"SPHASANN Step 4 (collapse \\n{{3,}} to \\n\\n): Applied. Text (first 200 chars): '{text[:200]}'")
    lines = text.split('\n')
    stripped_lines = [line.strip() for line in lines]
    text_after_line_stripping = "\n".join(stripped_lines)
    if text != text_after_line_stripping: 
        current_logger.debug(f"SPHASANN Step 5 (line stripping & rejoin): Applied. Text (first 200 chars): '{text_after_line_stripping[:200]}'")
    text = text_after_line_stripping
    final_text = text 
    current_logger.debug(f"SPHASANN Final output (first 200 chars): '{final_text[:200]}'")
    return final_text

def extract_sse_lines(buffer: bytearray) -> Tuple[List[bytes], bytearray]:
    lines: List[bytes] = []
    start_index: int = 0
    buffer_len = len(buffer)
    while start_index < buffer_len:
        newline_index = buffer.find(b'\n', start_index)
        if newline_index == -1:
            break
        line = buffer[start_index:newline_index]
        if line.endswith(b'\r'):
            line = line[:-1]
        if len(line) > MAX_SSE_LINE_LENGTH:
            logger.warning(
                f"SSE line too long ({len(line)} bytes), exceeded MAX_SSE_LINE_LENGTH ({MAX_SSE_LINE_LENGTH}). Line skipped. "
                f"Content start: {line[:100]!r}"
            )
        else:
            lines.append(line)
        start_index = newline_index + 1
    return lines, buffer[start_index:]

def get_current_time_iso() -> str:
    return datetime.datetime.utcnow().isoformat() + "Z"

def is_gemini_2_5_model(model_name: str) -> bool:
    if not isinstance(model_name, str):
        return False
    return "gemini-2.5" in model_name.lower()

# --- 新增：文档处理辅助函数 ---

def _extract_text_from_pdf_pypdf2(file_path: str) -> Optional[str]:
    """使用 PyPDF2 从 PDF 文件中提取文本。"""
    if not PyPDF2:
        logger.warning("Attempted to extract PDF text, but PyPDF2 library is not available.")
        return None
    text_content = ""
    try:
        with open(file_path, "rb") as f:
            reader = PyPDF2.PdfReader(f)
            if reader.is_encrypted:
                try:
                    # 尝试用空密码解密，对某些保护性PDF可能有效
                    if reader.decrypt("") == PyPDF2.PasswordType.OWNER_PASSWORD or \
                       reader.decrypt("") == PyPDF2.PasswordType.USER_PASSWORD :
                        logger.info(f"Successfully decrypted PDF (with empty password): {file_path}")
                    else:
                        logger.warning(f"PDF file is encrypted and could not be decrypted with an empty password: {file_path}")
                        return None # Or some indicator of encryption
                except Exception as decrypt_err:
                    logger.warning(f"Failed to decrypt PDF {file_path}: {decrypt_err}")
                    return None

            for page in reader.pages:
                try:
                    text_content += page.extract_text() or "" # Add "or """ to handle None from extract_text
                except Exception as page_extract_err:
                    logger.warning(f"Error extracting text from a page in {file_path}: {page_extract_err}")
                    continue # 继续处理下一页
        return text_content.strip()
    except FileNotFoundError:
        logger.error(f"PDF file not found for extraction: {file_path}")
        return None
    except Exception as e:
        logger.error(f"Error extracting text from PDF {file_path} using PyPDF2: {e}", exc_info=True)
        return None

def _extract_text_from_docx_python_docx(file_path: str) -> Optional[str]:
    """使用 python-docx 从 DOCX 文件中提取文本。"""
    if not docx:
        logger.warning("Attempted to extract DOCX text, but python-docx library is not available.")
        return None
    try:
        doc_obj = docx.Document(file_path)
        full_text = [para.text for para in doc_obj.paragraphs]
        return "\n".join(full_text).strip()
    except FileNotFoundError:
        logger.error(f"DOCX file not found for extraction: {file_path}")
        return None
    except Exception as e:
        logger.error(f"Error extracting text from DOCX {file_path} using python-docx: {e}", exc_info=True)
        return None

def _extract_text_from_plain_text(file_path: str) -> Optional[str]:
    """从纯文本文件中提取文本 (尝试多种编码)。"""
    common_encodings = ['utf-8', 'gbk', 'gb2312', 'latin-1', 'iso-8859-1'] # 可以根据需要调整
    try:
        for encoding in common_encodings:
            try:
                with open(file_path, "r", encoding=encoding) as f:
                    return f.read().strip()
            except UnicodeDecodeError:
                logger.debug(f"Failed to decode plain text file {file_path} with encoding {encoding}")
                continue
            except FileNotFoundError:
                logger.error(f"Plain text file not found for extraction: {file_path}")
                return None
        logger.warning(f"Could not decode plain text file {file_path} with common encodings.")
        return None # 如果所有尝试都失败
    except Exception as e:
        logger.error(f"Error extracting text from plain text file {file_path}: {e}", exc_info=True)
        return None

async def extract_text_from_uploaded_document(
    uploaded_file_path: str,
    mime_type: Optional[str],
    original_filename: str
) -> Optional[str]:
    """
    根据MIME类型从上传的文档（已保存到临时路径）中提取文本。
    返回提取的文本或None（如果不支持或提取失败）。
    """
    logger.info(f"Attempting to extract text from '{original_filename}' (path: {uploaded_file_path}, mime: {mime_type})")
    
    # 优先使用传入的 MIME 类型，如果它在支持列表中
    effective_mime_type = mime_type.lower() if mime_type else None

    # （可选）如果 mime_type 不可靠或未知，可以使用 python-magic 进行更准确的检测
    # if magic and (not effective_mime_type or effective_mime_type == "application/octet-stream"):
    #     try:
    #         detected_mime = magic.from_file(uploaded_file_path, mime=True)
    #         if detected_mime and detected_mime != effective_mime_type:
    #             logger.info(f"MIME type detected by python-magic for '{original_filename}': {detected_mime} (original: {mime_type})")
    #             effective_mime_type = detected_mime.lower()
    #     except Exception as e_magic:
    #         logger.warning(f"Error using python-magic for '{original_filename}': {e_magic}")

    if not effective_mime_type:
        logger.warning(f"No effective MIME type for '{original_filename}', cannot determine extraction method.")
        return None

    extracted_text: Optional[str] = None

    if effective_mime_type in SUPPORTED_DOCUMENT_MIME_TYPES_FOR_TEXT_EXTRACTION:
        if effective_mime_type == "application/pdf":
            extracted_text = _extract_text_from_pdf_pypdf2(uploaded_file_path)
        elif effective_mime_type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
            extracted_text = _extract_text_from_docx_python_docx(uploaded_file_path)
        elif effective_mime_type == "application/msword": # .doc 文件
            # .doc 文件提取比较麻烦，通常需要 unoconv (依赖 LibreOffice) 或其他特定库
            # 这里暂时返回一个提示，或尝试简单的文本提取（如果文件内容是纯文本）
            logger.warning(f"Basic text extraction for .doc ('{original_filename}') is not robust. Full content might not be extracted.")
            extracted_text = _extract_text_from_plain_text(uploaded_file_path) # 尝试作为纯文本读取
            if not extracted_text:
                 extracted_text = "[后端提示：.doc 文件内容提取可能不完整或失败]"
        elif effective_mime_type.startswith("text/"): # 包括 text/plain, text/markdown, text/csv 等
            extracted_text = _extract_text_from_plain_text(uploaded_file_path)
        # 你可以在这里添加对其他 SUPPORTED_DOCUMENT_MIME_TYPES_FOR_TEXT_EXTRACTION 中类型的处理
        # 例如: CSV 或 Markdown 可能需要不同的处理方式或直接使用文本
        else:
            logger.info(f"MIME type '{effective_mime_type}' for '{original_filename}' is in supported list but no specific extractor implemented, attempting plain text.")
            extracted_text = _extract_text_from_plain_text(uploaded_file_path)
    else:
        logger.warning(f"Unsupported MIME type for text extraction: '{effective_mime_type}' for file '{original_filename}'.")
        return None

    if extracted_text:
        if len(extracted_text) > MAX_DOCUMENT_CONTENT_CHARS_FOR_PROMPT:
            logger.info(f"Extracted text from '{original_filename}' truncated from {len(extracted_text)} to {MAX_DOCUMENT_CONTENT_CHARS_FOR_PROMPT} characters.")
            extracted_text = extracted_text[:MAX_DOCUMENT_CONTENT_CHARS_FOR_PROMPT] + \
                             f"\n[内容已截断，原始长度超过 {MAX_DOCUMENT_CONTENT_CHARS_FOR_PROMPT} 字符]"
        logger.info(f"Successfully extracted text (len: {len(extracted_text)}) from '{original_filename}'.")
        return extracted_text.strip()
    else:
        logger.warning(f"Failed to extract text from '{original_filename}' (mime: {effective_mime_type}).")
        return None

# --- 文档处理辅助函数结束 ---