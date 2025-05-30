import os
import logging
import httpx # 用于全局 HTTP 客户端
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from typing import Optional

# 导入您的配置
from .config import (
    APP_VERSION, API_TIMEOUT, READ_TIMEOUT, MAX_CONNECTIONS,
    LOG_LEVEL_FROM_ENV, # COMMON_HEADERS (如果在这里使用，否则可以移除)
    TEMP_UPLOAD_DIR # 确保这个已在 config.py 中定义
)
# 导入您的主聊天路由
from .routers import chat as chat_router
# multimodal_chat 模块中的逻辑将由 chat_router 内部根据模型名称调用

# --- 日志配置 ---
# (这部分与你之前提供的代码一致，保持不变)
numeric_log_level = getattr(logging, LOG_LEVEL_FROM_ENV.upper(), logging.INFO)
logging.basicConfig(
    level=numeric_log_level,
    format='%(asctime)s %(levelname)-8s [%(name)s:%(module)s:%(lineno)d] - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger("EzTalkProxy.Main")

# 设置其他模块的日志级别 (如果 SPHASANN 是你项目的一部分)
if hasattr(logging.getLogger("EzTalkProxy"), 'SPHASANN'): # 检查是否存在
    logging.getLogger("EzTalkProxy.SPHASANN").setLevel(LOG_LEVEL_FROM_ENV.upper())

for lib_logger_name in ["httpx", "httpcore", "googleapiclient.discovery_cache", "uvicorn.access", "watchfiles"]:
    logging.getLogger(lib_logger_name).setLevel(logging.WARNING)
logging.getLogger("uvicorn.error").setLevel(logging.INFO)
# --- 日志配置结束 ---


# --- 应用生命周期管理 (Lifespan) ---
@asynccontextmanager
async def lifespan(app_instance: FastAPI):
    logger.info("Lifespan: 应用启动，开始初始化...")
    client_local: Optional[httpx.AsyncClient] = None
    try:
        # 初始化 HTTP 客户端
        client_local = httpx.AsyncClient(
            timeout=httpx.Timeout(API_TIMEOUT, read=READ_TIMEOUT),
            limits=httpx.Limits(max_connections=MAX_CONNECTIONS),
            http2=True,
            follow_redirects=True,
            trust_env=False # 显式设置，避免依赖系统代理（除非需要）
        )
        app_instance.state.http_client = client_local
        logger.info(f"Lifespan: HTTP客户端初始化成功。Timeout Connect: {API_TIMEOUT}s, Read Timeout: {READ_TIMEOUT}s, Max Connections: {MAX_CONNECTIONS}")

        # --- 检查并创建临时上传目录 ---
        if not os.path.exists(TEMP_UPLOAD_DIR):
            try:
                os.makedirs(TEMP_UPLOAD_DIR, exist_ok=True) # exist_ok=True 避免并发创建时出错
                logger.info(f"Lifespan: 成功创建或已存在临时上传目录: {TEMP_UPLOAD_DIR}")
            except OSError as e_mkdir:
                logger.error(f"Lifespan: 创建临时上传目录 {TEMP_UPLOAD_DIR} 失败: {e_mkdir}", exc_info=True)
                # 关键目录创建失败，可以考虑是否要阻止应用启动
                # raise RuntimeError(f"无法创建关键的临时上传目录: {TEMP_UPLOAD_DIR}. 应用无法启动。")
        else:
            logger.info(f"Lifespan: 临时上传目录已存在: {TEMP_UPLOAD_DIR}")
        # --- 目录检查结束 ---

    except Exception as e:
        logger.error(f"Lifespan: HTTP客户端初始化过程中发生错误: {e}", exc_info=True)
        # 即使出错，也尝试设置一个 None，以便后续代码不会因 getattr 失败而崩溃
        app_instance.state.http_client = None
    
    yield # FastAPI 应用在此运行

    logger.info("Lifespan: 应用关闭，开始关闭HTTP客户端...")
    client_to_close = getattr(app_instance.state, "http_client", None)
    if client_to_close and isinstance(client_to_close, httpx.AsyncClient) and not client_to_close.is_closed:
        try:
            await client_to_close.aclose()
            logger.info("Lifespan: HTTP客户端成功关闭。")
        except Exception as e:
            logger.error(f"Lifespan: 关闭HTTP客户端时发生错误: {e}", exc_info=True)
    elif client_to_close and isinstance(client_to_close, httpx.AsyncClient) and client_to_close.is_closed:
        logger.info("Lifespan: HTTP客户端先前已经关闭。")
    else:
        logger.warning("Lifespan: HTTP客户端未找到、状态未知或类型不正确，可能无需关闭或已处理。")
    
    # 清理状态，避免内存泄漏或后续访问已关闭的客户端
    if hasattr(app_instance.state, "http_client"):
        delattr(app_instance.state, "http_client") # 或者 app_instance.state.http_client = None

    logger.info("Lifespan: 应用关闭流程完成。")


# --- FastAPI 应用实例 ---
app = FastAPI(
    title="EzTalk Proxy",
    description=f"代理服务，版本: {APP_VERSION}",
    version=APP_VERSION,
    lifespan=lifespan, # 使用 lifespan 上下文管理器
    docs_url="/docs",   # OpenAPI 文档路径
    redoc_url="/redoc"  # ReDoc 文档路径
)

# --- CORS 中间件配置 ---
# 为了安全，生产环境中应配置具体的来源列表
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # 在开发中可以设为 "*"，生产中应设为你的前端域名列表
    allow_credentials=True,
    allow_methods=["*"], # 或者更具体的方法如 ["GET", "POST"]
    allow_headers=["*"], # 或者具体的头部
    expose_headers=["*"] # 如果前端需要访问特定的响应头
)
logger.info(f"FastAPI EzTalk Proxy v{APP_VERSION} 初始化完成，已配置CORS。")

# --- 包含主路由 ---
# 假设你的 chat_router.router 在 eztalk_proxy/routers/chat.py 中定义
app.include_router(chat_router.router, prefix="/api/v1") # 示例：添加API版本前缀
logger.info("聊天路由已加载到路径 /api/v1/chat (或其他在chat_router中定义的路径)")


# --- 健康检查端点 ---
@app.get("/health", status_code=200, include_in_schema=False, tags=["Utilities"])
async def health_check(request: Request):
    logger.debug("Health check endpoint called.")
    client_from_state = getattr(request.app.state, "http_client", None)
    client_status = "ok"
    detail_message = "HTTP client initialized and seems operational."

    if client_from_state is None:
        client_status = "error" # 应该是 error，因为这是关键组件
        detail_message = "HTTP client not initialized in app.state."
    elif not isinstance(client_from_state, httpx.AsyncClient):
        client_status = "error"
        detail_message = f"Unexpected object type in app.state.http_client: {type(client_from_state)}"
    elif client_from_state.is_closed:
        client_status = "warning" # 或 error，取决于你的容忍度
        detail_message = "HTTP client in app.state is closed."


    response_data = {"status": client_status, "detail": detail_message, "app_version": APP_VERSION}
    return response_data


# --- Uvicorn 启动配置 (当直接运行此文件时) ---
if __name__ == "__main__":
    import uvicorn

    APP_HOST = os.getenv("HOST", "0.0.0.0")
    APP_PORT = int(os.getenv("PORT", 7860))
    # DEV_RELOAD 通常由 uvicorn 命令行参数 --reload 控制，而不是环境变量
    # 如果你想通过环境变量控制，你需要在 uvicorn.run 中使用它

    # Uvicorn 的日志配置可以简化，它默认会使用 logging 模块的配置
    # 但如果你想更精细地控制 Uvicorn 自身的日志，可以像之前那样配置 log_config
    # 为了简单起见，这里使用 Uvicorn 默认行为，它会尊重 logging.basicConfig
    
    logger.info(f"准备启动 Uvicorn 服务器: http://{APP_HOST}:{APP_PORT}")
    logger.info(f"应用日志级别 (EzTalkProxy.*): {LOG_LEVEL_FROM_ENV}")
    
    uvicorn.run(
        "eztalk_proxy.main:app", # 指向 FastAPI 应用实例
        host=APP_HOST,
        port=APP_PORT,
        log_level=LOG_LEVEL_FROM_ENV.lower(), # 将我们配置的日志级别传递给 Uvicorn
        reload=os.getenv("DEV_RELOAD", "false").lower() == "true" # 从环境变量读取 reload 设置
        # reload_dirs=["eztalk_proxy"], # 如果 reload=True, 可以指定监控的目录
    )