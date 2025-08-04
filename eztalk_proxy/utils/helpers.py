import orjson
import re
import logging
import datetime
from typing import Any, Dict, List, Tuple, Optional
import os
import uuid

from fastapi.responses import JSONResponse
from fastapi import UploadFile

from ..core.config import (
    COMMON_HEADERS,
    MAX_SSE_LINE_LENGTH,
    SUPPORTED_DOCUMENT_MIME_TYPES_FOR_TEXT_EXTRACTION,
    MAX_DOCUMENT_CONTENT_CHARS_FOR_PROMPT,
)

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
    import PyPDF2
except ImportError:
    PyPDF2 = None
    logging.warning("1PyPDF2 library not found. PDF text extraction will not be available.")

try:
    import docx
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

def _extract_text_from_pdf_pypdf2(file_path: str) -> Optional[str]:
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
        # Treat HTML as a plain text file for extraction, which is a common and effective approach.
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

async def upload_to_gcs(
    file_obj: Any,
    original_filename: str,
    bucket_name: str,
    project_id: Optional[str] = None,
    content_type: Optional[str] = None,
    request_id: Optional[str] = None
) -> Optional[str]:
    log_prefix = f"RID-{request_id}" if request_id else "[GCS_UPLOAD]"
    
    if not storage:
        logger.error(f"{log_prefix} GCS upload skipped: google-cloud-storage library not available.")
        return None
    if not bucket_name:
        logger.error(f"{log_prefix} GCS upload skipped: GCS_BUCKET_NAME is not configured.")
        return None

    _, file_extension = os.path.splitext(original_filename)
    safe_original_filename_part = "".join(c if c.isalnum() or c in ['.', '_', '-'] else '_' for c in original_filename.rsplit('.', 1)[0])[:50]
    destination_blob_name = f"uploads/{request_id or 'unknown_req'}/{safe_original_filename_part}_{uuid.uuid4().hex[:8]}{file_extension}"

    logger.info(f"{log_prefix} Attempting to upload to GCS: bucket='{bucket_name}', blob='{destination_blob_name}'")

    try:
        if project_id:
            storage_client = storage.Client(project=project_id)
        else:
            storage_client = storage.Client()
            
        bucket = storage_client.bucket(bucket_name)
        blob = bucket.blob(destination_blob_name)

        if isinstance(file_obj, UploadFile):
            await file_obj.seek(0)
            blob.upload_from_file(file_obj.file, content_type=content_type or file_obj.content_type)
        elif hasattr(file_obj, 'read') and hasattr(file_obj, 'seek'):
            file_obj.seek(0)
            blob.upload_from_file(file_obj, content_type=content_type)
        elif isinstance(file_obj, str) and os.path.exists(file_obj):
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