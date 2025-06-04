# eztalk_proxy/utils.py

import orjson
import re
import logging
import datetime
from typing import Any, Dict, List, Tuple, Optional
import os
import shutil
import uuid # 确保 uuid 已导入，如果 upload_to_gcs 中要用随机blob名

from fastapi.responses import JSONResponse
from fastapi import UploadFile # 导入 UploadFile 以便类型提示

from .config import ( 
    COMMON_HEADERS,
    MAX_SSE_LINE_LENGTH,
    TEMP_UPLOAD_DIR, 
    SUPPORTED_DOCUMENT_MIME_TYPES_FOR_TEXT_EXTRACTION, 
    MAX_DOCUMENT_CONTENT_CHARS_FOR_PROMPT,
    # GCS 相关配置，虽然在这里不直接使用，但导入以表明 utils 中可能有 GCS 相关功能
    # GEMINI_ENABLE_GCS_UPLOAD, # 这些在调用函数时传入，不在 utils 中直接使用
    # GCS_BUCKET_NAME,
    # GCS_PROJECT_ID
)

# 尝试导入文档处理库和 GCS 库
try:
    from google.cloud import storage
    from google.auth.exceptions import DefaultCredentialsError
except ImportError:
    storage = None # type: ignore
    DefaultCredentialsError = None # type: ignore
    logging.warning(
        "google-cloud-storage or google-auth library not found. "
        "GCS upload functionality for large files (video/audio) for Gemini will not be available."
    )

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

# --- 文档处理辅助函数 ---

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
                    if reader.decrypt("") == PyPDF2.PasswordType.OWNER_PASSWORD or \
                       reader.decrypt("") == PyPDF2.PasswordType.USER_PASSWORD :
                        logger.info(f"Successfully decrypted PDF (with empty password): {file_path}")
                    else:
                        logger.warning(f"PDF file is encrypted and could not be decrypted with an empty password: {file_path}")
                        return None 
                except Exception as decrypt_err:
                    logger.warning(f"Failed to decrypt PDF {file_path}: {decrypt_err}")
                    return None

            for page in reader.pages:
                try:
                    text_content += page.extract_text() or "" 
                except Exception as page_extract_err:
                    logger.warning(f"Error extracting text from a page in {file_path}: {page_extract_err}")
                    continue 
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
    common_encodings = ['utf-8', 'gbk', 'gb2312', 'latin-1', 'iso-8859-1'] 
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
        return None 
    except Exception as e:
        logger.error(f"Error extracting text from plain text file {file_path}: {e}", exc_info=True)
        return None

async def extract_text_from_uploaded_document(
    uploaded_file_path: str,
    mime_type: Optional[str],
    original_filename: str
) -> Optional[str]:
    logger.info(f"Attempting to extract text from '{original_filename}' (path: {uploaded_file_path}, mime: {mime_type})")
    effective_mime_type = mime_type.lower() if mime_type else None

    if not effective_mime_type:
        logger.warning(f"No effective MIME type for '{original_filename}', cannot determine extraction method.")
        return None

    extracted_text: Optional[str] = None

    if effective_mime_type in SUPPORTED_DOCUMENT_MIME_TYPES_FOR_TEXT_EXTRACTION:
        if effective_mime_type == "application/pdf":
            extracted_text = _extract_text_from_pdf_pypdf2(uploaded_file_path)
        elif effective_mime_type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
            extracted_text = _extract_text_from_docx_python_docx(uploaded_file_path)
        elif effective_mime_type == "application/msword": 
            logger.warning(f"Basic text extraction for .doc ('{original_filename}') is not robust. Full content might not be extracted.")
            extracted_text = _extract_text_from_plain_text(uploaded_file_path) 
            if not extracted_text:
                 extracted_text = "[后端提示：.doc 文件内容提取可能不完整或失败]"
        elif effective_mime_type.startswith("text/"): 
            extracted_text = _extract_text_from_plain_text(uploaded_file_path)
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

# --- GCS 上传辅助函数 ---
async def upload_to_gcs(
    file_obj: Any, # Can be UploadFile.file (SpooledTemporaryFile) or a file path string
    original_filename: str, # Used for generating a unique blob name
    bucket_name: str,
    project_id: Optional[str] = None,
    content_type: Optional[str] = None,
    request_id: Optional[str] = None
) -> Optional[str]:
    """
    Uploads a file object or file from a path to Google Cloud Storage.
    Returns the GCS URI (gs://bucket_name/destination_blob_name) if successful, else None.
    """
    log_prefix = f"RID-{request_id}" if request_id else "[GCS_UPLOAD]"
    
    if not storage:
        logger.error(f"{log_prefix} GCS upload skipped: google-cloud-storage library not available.")
        return None
    if not bucket_name:
        logger.error(f"{log_prefix} GCS upload skipped: GCS_BUCKET_NAME is not configured.")
        return None

    # Generate a unique blob name to avoid collisions
    _, file_extension = os.path.splitext(original_filename)
    # Sanitize original_filename part for blob name
    safe_original_filename_part = "".join(c if c.isalnum() or c in ['.', '_', '-'] else '_' for c in original_filename.rsplit('.', 1)[0])[:50]
    destination_blob_name = f"uploads/{request_id or 'unknown_req'}/{safe_original_filename_part}_{uuid.uuid4().hex[:8]}{file_extension}"

    logger.info(f"{log_prefix} Attempting to upload to GCS: bucket='{bucket_name}', blob='{destination_blob_name}'")

    try:
        if project_id:
            storage_client = storage.Client(project=project_id)
        else:
            # GOOGLE_APPLICATION_CREDENTIALS env var should be set for this to work
            storage_client = storage.Client() 
            
        bucket = storage_client.bucket(bucket_name)
        blob = bucket.blob(destination_blob_name)

        if isinstance(file_obj, UploadFile): # FastAPI UploadFile
            await file_obj.seek(0) # Ensure reading from the beginning
            blob.upload_from_file(file_obj.file, content_type=content_type or file_obj.content_type)
        elif hasattr(file_obj, 'read') and hasattr(file_obj, 'seek'): # Generic file-like object
            file_obj.seek(0) 
            blob.upload_from_file(file_obj, content_type=content_type)
        elif isinstance(file_obj, str) and os.path.exists(file_obj): # File path
            blob.upload_from_filename(file_obj, content_type=content_type)
        else:
            logger.error(f"{log_prefix} GCS upload failed for '{original_filename}': Invalid file_obj type or file path does not exist.")
            return None

        gcs_uri = f"gs://{bucket_name}/{destination_blob_name}"
        logger.info(f"{log_prefix} Successfully uploaded '{original_filename}' to GCS: {gcs_uri}")
        return gcs_uri
    except DefaultCredentialsError:
        logger.error(
            f"{log_prefix} GCS upload failed for '{original_filename}': Google Cloud Default Credentials not found. "
            "Ensure GOOGLE_APPLICATION_CREDENTIALS environment variable is set correctly "
            "or the runtime environment has appropriate GCS permissions.",
            exc_info=True 
        )
        return None
    except Exception as e:
        logger.error(f"{log_prefix} GCS upload failed for '{original_filename}' (blob '{destination_blob_name}'): {e}", exc_info=True)
        return None

# --- GCS 上传辅助函数结束 ---