# eztalk_proxy/config.py
import os
from dotenv import load_dotenv

load_dotenv() # 加载 .env 文件中的环境变量

# Application Info
APP_VERSION = os.getenv("APP_VERSION", "1.9.9.76-doc-support") # 更新版本号示例

# Logging
LOG_LEVEL_FROM_ENV = os.getenv("LOG_LEVEL", "DEBUG").upper()

# API Endpoints and Paths
DEFAULT_OPENAI_API_BASE_URL = os.getenv("DEFAULT_OPENAI_API_BASE_URL", "https://api.openai.com")
GOOGLE_API_BASE_URL = os.getenv("GOOGLE_API_BASE_URL", "https://generativelanguage.googleapis.com")
OPENAI_COMPATIBLE_PATH = os.getenv("OPENAI_COMPATIBLE_PATH", "/v1/chat/completions")

# API Keys (from frontend or env for specific services like web search)
# 这些主要用于那些不由前端直接提供 API Key 的服务，例如你可能在后端硬编码或通过环境变量配置的 Web 搜索 API Key
GOOGLE_API_KEY_ENV = os.getenv("GOOGLE_API_KEY") # 用于 Google Custom Search API 等
GOOGLE_CSE_ID = os.getenv("GOOGLE_CSE_ID")       # 用于 Google Custom Search Engine ID

# Timeouts and Limits
API_TIMEOUT = int(os.getenv("API_TIMEOUT", "300")) # 通用 API 请求超时 (秒)
READ_TIMEOUT = float(os.getenv("READ_TIMEOUT", "60.0")) # Ktor/httpx 流读取超时 (秒)，但对于 SSE 流通常会覆盖
MAX_CONNECTIONS = int(os.getenv("MAX_CONNECTIONS", "200")) # httpx 连接池大小
MAX_SSE_LINE_LENGTH = int(os.getenv("MAX_SSE_LINE_LENGTH", f"{1024 * 1024}")) # SSE 行最大长度

# Web Search Configuration
SEARCH_RESULT_COUNT = int(os.getenv("SEARCH_RESULT_COUNT", "5"))
SEARCH_SNIPPET_MAX_LENGTH = int(os.getenv("SEARCH_SNIPPET_MAX_LENGTH", "200"))

# AI Behavior (这些常量目前在代码中未直接使用，但保留以备将来可能)
THINKING_PROCESS_SEPARATOR = os.getenv("THINKING_PROCESS_SEPARATOR", "--- FINAL ANSWER ---")
MIN_FLUSH_LENGTH_HEURISTIC = int(os.getenv("MIN_FLUSH_LENGTH_HEURISTIC", "80"))

# HTTP Headers
COMMON_HEADERS = {"X-Accel-Buffering": "no"} # 用于禁用Nginx等代理的缓冲，确保SSE流式传输

# --- 文件处理配置 ---
# Dockerfile 中创建的目录名是 temp_document_uploads。
# 如果 WORKDIR 是 /app，这个相对路径会解析为 /app/temp_document_uploads
# 确保这个值与 Dockerfile 中 RUN mkdir -p ... 使用的目录名一致。
TEMP_UPLOAD_DIR = os.getenv("TEMP_UPLOAD_DIR", "temp_document_uploads")

MAX_DOCUMENT_UPLOAD_SIZE_MB = int(os.getenv("MAX_DOCUMENT_UPLOAD_SIZE_MB", "20"))
MAX_DOCUMENT_CONTENT_CHARS_FOR_PROMPT = int(os.getenv("MAX_DOCUMENT_CONTENT_CHARS_FOR_PROMPT", "15000"))

SUPPORTED_DOCUMENT_MIME_TYPES_FOR_TEXT_EXTRACTION = [
    "text/plain",
    "application/pdf",
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document", # docx
    "application/msword", # doc
    # "text/markdown",
    # "text/csv",
]
# --- 文件处理配置结束 ---