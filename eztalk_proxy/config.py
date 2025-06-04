import os
from dotenv import load_dotenv

load_dotenv() # 加载 .env 文件中的环境变量

# Application Info
APP_VERSION = os.getenv("APP_VERSION", "1.9.9.77-gcs-support") # 更新版本号示例

# Logging
LOG_LEVEL_FROM_ENV = os.getenv("LOG_LEVEL", "DEBUG").upper()

# API Endpoints and Paths
DEFAULT_OPENAI_API_BASE_URL = os.getenv("DEFAULT_OPENAI_API_BASE_URL", "https://api.openai.com")
GOOGLE_API_BASE_URL = os.getenv("GOOGLE_API_BASE_URL", "https://generativelanguage.googleapis.com")
OPENAI_COMPATIBLE_PATH = os.getenv("OPENAI_COMPATIBLE_PATH", "/v1/chat/completions")

# API Keys (from frontend or env for specific services like web search)
GOOGLE_API_KEY_ENV = os.getenv("GOOGLE_API_KEY") # 用于 Google Custom Search API 等
GOOGLE_CSE_ID = os.getenv("GOOGLE_CSE_ID")       # 用于 Google Custom Search Engine ID

# Timeouts and Limits
API_TIMEOUT = int(os.getenv("API_TIMEOUT", "300")) # 通用 API 请求超时 (秒)
READ_TIMEOUT = float(os.getenv("READ_TIMEOUT", "60.0")) # 流读取超时 (秒)
MAX_CONNECTIONS = int(os.getenv("MAX_CONNECTIONS", "200")) # httpx 连接池大小
MAX_SSE_LINE_LENGTH = int(os.getenv("MAX_SSE_LINE_LENGTH", f"{1024 * 1024}")) # SSE 行最大长度

# Web Search Configuration
SEARCH_RESULT_COUNT = int(os.getenv("SEARCH_RESULT_COUNT", "5"))
SEARCH_SNIPPET_MAX_LENGTH = int(os.getenv("SEARCH_SNIPPET_MAX_LENGTH", "200"))

# AI Behavior
THINKING_PROCESS_SEPARATOR = os.getenv("THINKING_PROCESS_SEPARATOR", "--- FINAL ANSWER ---")
MIN_FLUSH_LENGTH_HEURISTIC = int(os.getenv("MIN_FLUSH_LENGTH_HEURISTIC", "80"))

# HTTP Headers
COMMON_HEADERS = {"X-Accel-Buffering": "no"} # 用于禁用Nginx等代理的缓冲，确保SSE流式传输

# --- 文件处理配置 ---
TEMP_UPLOAD_DIR = os.getenv("TEMP_UPLOAD_DIR", "temp_document_uploads")
MAX_DOCUMENT_UPLOAD_SIZE_MB = int(os.getenv("MAX_DOCUMENT_UPLOAD_SIZE_MB", "20")) # 对于GCS上传，这个限制可能需要调整或单独处理
MAX_DOCUMENT_CONTENT_CHARS_FOR_PROMPT = int(os.getenv("MAX_DOCUMENT_CONTENT_CHARS_FOR_PROMPT", "15000"))

SUPPORTED_DOCUMENT_MIME_TYPES_FOR_TEXT_EXTRACTION = [
    "text/plain",
    "application/pdf",
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document", # docx
    "application/msword", # doc
]

# Gemini 支持的上传文件 MIME 类型 (包括通过 GCS 上传的视频/音频)
GEMINI_SUPPORTED_UPLOAD_MIMETYPES = [
    # 图片 (通常内联 Base64)
    "image/png",
    "image/jpeg",
    "image/webp",
    "image/heic",
    "image/heif",
    # 视频 (通过 GCS URI) - 参考 Google AI SDK 文档
    "video/mp4", "application/mp4",
    "video/mpeg",
    "video/quicktime", # .mov
    "video/x-msvideo", # .avi
    "video/x-flv",
    "video/x-matroska", # .mkv
    "video/webm",
    "video/x-ms-wmv", # .wmv
    "video/3gpp",     # .3gp
    "video/x-m4v",    # .m4v
    # 音频 (通过 GCS URI) - 参考 Google AI SDK 文档
    "audio/wav", "audio/x-wav",
    "audio/mpeg",     # .mp3
    "audio/aac",
    "audio/ogg",      # .ogg (Vorbis)
    "audio/opus",
    "audio/flac",
    "audio/midi",     # .mid, .midi
    "audio/amr",
    "audio/aiff",
    "audio/x-m4a",    # .m4a
    # 纯文本也可以通过 fileData 传递 (虽然通常用 text part)
    "text/plain", 
    # PDF 也可以通过 fileData (GCS) 传递给 Gemini，如果不想内联或提取文本
    "application/pdf",
]

# --- GCS 配置 (用于 Gemini 大文件上传，如视频/音频) ---
# 是否启用通过 GCS 上传大文件给 Gemini (True/False)
GEMINI_ENABLE_GCS_UPLOAD = os.getenv("GEMINI_ENABLE_GCS_UPLOAD", "False").lower() == "true"
# 你的 Google Cloud Storage Bucket 名称
GCS_BUCKET_NAME = os.getenv("GCS_BUCKET_NAME")
# Google Cloud 项目 ID (如果需要显式指定，通常 SDK 会从凭证中获取)
# 如果不设置，google-cloud-storage 客户端通常会尝试从环境中自动检测项目ID。
GCS_PROJECT_ID = os.getenv("GCS_PROJECT_ID", None) 
# GOOGLE_APPLICATION_CREDENTIALS 环境变量应指向服务账号密钥JSON文件路径。
# 这个通常在运行环境中设置，而不是在这里直接读取其值。代码会使用此环境变量。

# --- 文件处理配置结束 ---