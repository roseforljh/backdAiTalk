import os
from dotenv import load_dotenv

load_dotenv()

APP_VERSION = os.getenv("APP_VERSION", "1.9.9.77-gcs-support")

LOG_LEVEL_FROM_ENV = os.getenv("LOG_LEVEL", "INFO").upper()

DEFAULT_OPENAI_API_BASE_URL = os.getenv("DEFAULT_OPENAI_API_BASE_URL", "https://api.openai.com")
GOOGLE_API_BASE_URL = os.getenv("GOOGLE_API_BASE_URL", "https://generativelanguage.googleapis.com")
OPENAI_COMPATIBLE_PATH = os.getenv("OPENAI_COMPATIBLE_PATH", "/v1/chat/completions")

GOOGLE_API_KEY_ENV = os.getenv("GOOGLE_API_KEY")
GOOGLE_CSE_ID = os.getenv("GOOGLE_CSE_ID")

API_TIMEOUT = int(os.getenv("API_TIMEOUT", "600"))
READ_TIMEOUT = float(os.getenv("READ_TIMEOUT", "60.0"))
MAX_CONNECTIONS = int(os.getenv("MAX_CONNECTIONS", "200"))
MAX_SSE_LINE_LENGTH = int(os.getenv("MAX_SSE_LINE_LENGTH", f"{1024 * 1024}"))

SEARCH_RESULT_COUNT = int(os.getenv("SEARCH_RESULT_COUNT", "5"))
SEARCH_SNIPPET_MAX_LENGTH = int(os.getenv("SEARCH_SNIPPET_MAX_LENGTH", "200"))

THINKING_PROCESS_SEPARATOR = os.getenv("THINKING_PROCESS_SEPARATOR", "--- FINAL ANSWER ---")
MIN_FLUSH_LENGTH_HEURISTIC = int(os.getenv("MIN_FLUSH_LENGTH_HEURISTIC", "80"))

COMMON_HEADERS = {"X-Accel-Buffering": "no"}

TEMP_UPLOAD_DIR = os.getenv("TEMP_UPLOAD_DIR", "/tmp/temp_document_uploads")
MAX_DOCUMENT_UPLOAD_SIZE_MB = int(os.getenv("MAX_DOCUMENT_UPLOAD_SIZE_MB", "20"))
MAX_DOCUMENT_CONTENT_CHARS_FOR_PROMPT = int(os.getenv("MAX_DOCUMENT_CONTENT_CHARS_FOR_PROMPT", "15000"))

SUPPORTED_DOCUMENT_MIME_TYPES_FOR_TEXT_EXTRACTION = [
    # Plain Text & Data Formats
    "text/plain",
    "text/html",
    "text/csv",
    "text/markdown",
    "application/json",
    "text/xml",
    "text/rtf",
    
    # Document Formats
    "application/pdf",
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document", # .docx
    "application/msword", # .doc

    # Audio Formats
    "audio/flac",
    "audio/wav",
    "audio/x-wav",

    # Common Code Formats (treated as plain text)
    "application/x-javascript",
    "text/javascript",
    "text/css",
    "application/x-python",
    "text/x-python",
]

GEMINI_SUPPORTED_UPLOAD_MIMETYPES = [
    "image/png",
    "image/jpeg",
    "image/webp",
    "image/heic",
    "image/heif",
    "video/mp4", "application/mp4",
    "video/mpeg",
    "video/quicktime",
    "video/x-msvideo",
    "video/x-flv",
    "video/x-matroska",
    "video/webm",
    "video/x-ms-wmv",
    "video/3gpp",
    "video/x-m4v",
    "audio/wav", "audio/x-wav",
    "audio/mpeg",
    "audio/aac",
    "audio/ogg",
    "audio/opus",
    "audio/flac",
    "audio/midi",
    "audio/amr",
    "audio/aiff",
    "audio/x-m4a",
    "text/plain",
    "application/pdf",
]

GEMINI_ENABLE_GCS_UPLOAD = os.getenv("GEMINI_ENABLE_GCS_UPLOAD", "False").lower() == "true"
GCS_BUCKET_NAME = os.getenv("GCS_BUCKET_NAME")
GCS_PROJECT_ID = os.getenv("GCS_PROJECT_ID", None)