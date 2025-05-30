# eztalk_proxy/config.py
import os
from dotenv import load_dotenv

load_dotenv()

# Application Info
APP_VERSION = os.getenv("APP_VERSION", "1.9.9.75-websearch-restored") # 你可以更新这个版本

# Logging
LOG_LEVEL_FROM_ENV = os.getenv("LOG_LEVEL", "DEBUG").upper()

# API Endpoints and Paths
DEFAULT_OPENAI_API_BASE_URL = os.getenv("DEFAULT_OPENAI_API_BASE_URL", "https://api.openai.com")
GOOGLE_API_BASE_URL = os.getenv("GOOGLE_API_BASE_URL", "https://generativelanguage.googleapis.com")
OPENAI_COMPATIBLE_PATH = os.getenv("OPENAI_COMPATIBLE_PATH", "/v1/chat/completions")

# API Keys (from frontend or env for specific services like web search)
GOOGLE_API_KEY_ENV = os.getenv("GOOGLE_API_KEY")
GOOGLE_CSE_ID = os.getenv("GOOGLE_CSE_ID")

# Timeouts and Limits
API_TIMEOUT = int(os.getenv("API_TIMEOUT", "300"))
READ_TIMEOUT = float(os.getenv("READ_TIMEOUT", "60.0"))
MAX_CONNECTIONS = int(os.getenv("MAX_CONNECTIONS", "200"))
MAX_SSE_LINE_LENGTH = int(os.getenv("MAX_SSE_LINE_LENGTH", f"{1024 * 1024}"))

# Web Search Configuration
SEARCH_RESULT_COUNT = int(os.getenv("SEARCH_RESULT_COUNT", "5"))
SEARCH_SNIPPET_MAX_LENGTH = int(os.getenv("SEARCH_SNIPPET_MAX_LENGTH", "200"))

# AI Behavior
THINKING_PROCESS_SEPARATOR = os.getenv("THINKING_PROCESS_SEPARATOR", "--- FINAL ANSWER ---")
MIN_FLUSH_LENGTH_HEURISTIC = int(os.getenv("MIN_FLUSH_LENGTH_HEURISTIC", "80"))

# HTTP
COMMON_HEADERS = {"X-Accel-Buffering": "no"}

# --- 文件处理配置 (新增) ---
TEMP_UPLOAD_DIR = os.getenv("TEMP_UPLOAD_DIR", "temp_document_uploads") # 临时文件存储目录
MAX_DOCUMENT_UPLOAD_SIZE_MB = int(os.getenv("MAX_DOCUMENT_UPLOAD_SIZE_MB", "20")) # 最大文档上传大小 (MB)
# 限制从文档中提取并送入prompt的文本长度 (字符数)
MAX_DOCUMENT_CONTENT_CHARS_FOR_PROMPT = int(os.getenv("MAX_DOCUMENT_CONTENT_CHARS_FOR_PROMPT", "15000"))

# 支持的文档MIME类型，用于文本提取
# (可以进一步细化，比如在代码中直接定义，或者保持在这里)
SUPPORTED_DOCUMENT_MIME_TYPES_FOR_TEXT_EXTRACTION = [
    "text/plain",
    "application/pdf",
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document", # docx
    "application/msword", # doc (提取可能不完美或需要特定库)
    # "text/markdown", # md
    # "text/csv", # csv (可能需要特殊处理，例如转换为文本描述)
    # "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", # xlsx
    # "application/vnd.ms-excel", # xls
]
# --- 文件处理配置结束 ---

# 确保临时上传目录存在
if not os.path.exists(TEMP_UPLOAD_DIR):
    try:
        os.makedirs(TEMP_UPLOAD_DIR)
        print(f"Created temporary upload directory: {TEMP_UPLOAD_DIR}")
    except OSError as e:
        print(f"Error creating temporary upload directory {TEMP_UPLOAD_DIR}: {e}")