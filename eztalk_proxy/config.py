# eztalk_proxy/config.py
import os
from dotenv import load_dotenv

load_dotenv()

# Application Info
APP_VERSION = os.getenv("APP_VERSION", "1.9.9.75-websearch-restored")

# Logging
LOG_LEVEL_FROM_ENV = os.getenv("LOG_LEVEL", "DEBUG").upper()

# API Endpoints and Paths
DEFAULT_OPENAI_API_BASE_URL = os.getenv("DEFAULT_OPENAI_API_BASE_URL", "https://api.openai.com")
# GOOGLE_API_BASE_URL 用于 Google Generative Language API (使用 API Key)
# 这个URL会被 routers/multimodal_chat.py 用来构建请求，如果前端不提供完整的API地址
GOOGLE_API_BASE_URL = os.getenv("GOOGLE_API_BASE_URL", "https://generativelanguage.googleapis.com")
OPENAI_COMPATIBLE_PATH = os.getenv("OPENAI_COMPATIBLE_PATH", "/v1/chat/completions")

# --- API Keys and IDs ---
# GOOGLE_GEMINI_API_KEY 现在将从前端请求的 api_key 字段获取，此处不再需要从环境变量读取
# GOOGLE_APPLICATION_CREDENTIALS_STRING (用于Vertex AI服务账户) 也不再需要
# GOOGLE_CLOUD_PROJECT 和 VERTEX_AI_REGION (用于Vertex AI) 也不再需要

# 保留可能用于Web搜索或其他非Gemini Google服务的API Key
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