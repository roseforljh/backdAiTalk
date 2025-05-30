# eztalk_proxy/main.py
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
    LOG_LEVEL_FROM_ENV, COMMON_HEADERS,
    TEMP_UPLOAD_DIR # <--- 新增导入 TEMP_UPLOAD_DIR
)
# 导入您的主聊天路由
from .routers import chat as chat_router
# multimodal_chat 模块中的逻辑将由 chat_router 内部根据模型名称调用

# --- 日志配置 ---
numeric_log_level = getattr(logging, LOG_LEVEL_FROM_ENV.upper(), logging.INFO)
logging.basicConfig(
    level=numeric_log_level,
    format='%(asctime)s %(levelname)-8s [%(name)s:%(module)s:%(lineno)d] - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger("EzTalkProxy.Main")

# 设置其他模块的日志级别
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
            timeout=httpx.Timeout(API_TIMEOUT, read=READ_TIMEOUT), # 使用 config.py 中的 API_TIMEOUT
            limits=httpx.Limits(max_connections=MAX_CONNECTIONS), # 使用 config.py 中的 MAX_CONNECTIONS
            http2=True,
            follow_redirects=True,
            trust_env=False
        )
        app_instance.state.http_client = client_local
        logger.info(f"Lifespan: HTTP客户端初始化成功。Timeout Connect: {API_TIMEOUT}s, Read Timeout: {READ_TIMEOUT}s, Max Connections: {MAX_CONNECTIONS}")

        # --- 新增：检查并创建临时上传目录 ---
        if not os.path.exists(TEMP_UPLOAD_DIR):
            try:
                os.makedirs(TEMP_UPLOAD_DIR)
                logger.info(f"Lifespan: 成功创建临时上传目录: {TEMP_UPLOAD_DIR}")
            except OSError as e_mkdir:
                logger.error(f"Lifespan: 创建临时上传目录 {TEMP_UPLOAD_DIR} 失败: {e_mkdir}", exc_info=True)
                # 你可能想在这里决定如果目录创建失败是否要阻止应用启动
                # 例如: raise RuntimeError(f"Could not create temp upload directory: {TEMP_UPLOAD_DIR}")
        else:
            logger.info(f"Lifespan: 临时上传目录已存在: {TEMP_UPLOAD_DIR}")
        # --- 新增结束 ---

    except Exception as e:
        logger.error(f"Lifespan: HTTP客户端初始化过程中发生错误: {e}", exc_info=True)
        app_instance.state.http_client = None # 确保即使出错也设置
    
    yield # FastAPI 应用在此运行

    logger.info("Lifespan: 应用关闭，开始关闭HTTP客户端...")
    client_to_close = getattr(app_instance.state, "http_client", None)
    if client_to_close and hasattr(client_to_close, "is_closed") and not client_to_close.is_closed:
        try:
            await client_to_close.aclose()
            logger.info("Lifespan: HTTP客户端成功关闭。")
        except Exception as e:
            logger.error(f"Lifespan: 关闭HTTP客户端时发生错误: {e}", exc_info=True)
    elif client_to_close and hasattr(client_to_close, 'is_closed') and client_to_close.is_closed:
        logger.info("Lifespan: HTTP客户端先前已经关闭。")
    else:
        logger.warning("Lifespan: HTTP客户端未找到或状态未知，可能无需关闭或已处理。")
    
    app_instance.state.http_client = None
    logger.info("Lifespan: 应用关闭流程完成。")


# --- FastAPI 应用实例 ---
app = FastAPI(
    title="EzTalk Proxy",
    description=f"代理服务，版本: {APP_VERSION}",
    version=APP_VERSION,
    lifespan=lifespan, # 使用 lifespan 上下文管理器
    docs_url="/docs",
    redoc_url="/redoc"
)

# --- CORS 中间件配置 ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"]
)
logger.info(f"FastAPI EzTalk Proxy v{APP_VERSION} 初始化完成，已配置CORS。")

# --- 包含主路由 ---
# 假设你的 chat_router.router 在 routers/chat.py 中定义
app.include_router(chat_router.router) # 直接包含路由，前缀可以在路由内部定义或在这里定义

# --- 健康检查端点 ---
@app.get("/health", status_code=200, include_in_schema=False, tags=["Utilities"])
async def health_check(request: Request):
    logger.debug("Health check endpoint called.")
    client_from_state = getattr(request.app.state, "http_client", None)
    client_status = "ok"
    detail_message = "HTTP client initialized and open."

    if client_from_state is None:
        client_status = "warning"
        detail_message = "HTTP client not initialized in app.state."
    elif hasattr(client_from_state, 'is_closed') and client_from_state.is_closed:
        client_status = "warning"
        detail_message = "HTTP client in app.state is closed."
    elif not hasattr(client_from_state, 'is_closed'): # 进一步检查类型是否符合预期
        client_status = "error"
        detail_message = f"Unexpected object in app.state.http_client: {type(client_from_state)}"

    response_data = {"status": client_status, "detail": detail_message, "app_version": APP_VERSION}
    return response_data


# --- Uvicorn 启动配置 (当直接运行此文件时) ---
if __name__ == "__main__":
    import uvicorn

    APP_HOST = os.getenv("HOST", "0.0.0.0")
    APP_PORT = int(os.getenv("PORT", 7860))
    DEV_RELOAD = os.getenv("DEV_RELOAD", "false").lower() == "true"

    log_config = uvicorn.config.LOGGING_CONFIG.copy()
    log_config["formatters"].setdefault("default", {"fmt": "%(levelprefix)s %(asctime)s [%(name)s] - %(message)s", "datefmt": "%Y-%m-%d %H:%M:%S", "use_colors": None})
    log_config["formatters"]["default"]["fmt"] = "%(asctime)s %(levelname)-8s [%(name)s:%(module)s:%(lineno)d] - %(message)s"
    log_config["formatters"].setdefault("access", {"fmt": "", "datefmt": "", "use_colors": None})
    log_config["formatters"]["access"]["fmt"] = '%(asctime)s %(levelname)-8s [%(name)s] - %(client_addr)s - "%(request_line)s" %(status_code)s'
    log_config["formatters"]["access"]["datefmt"] = "%Y-%m-%d %H:%M:%S"
    log_config["handlers"].setdefault("default", {"formatter": "default", "class": "logging.StreamHandler", "stream": "ext://sys.stderr"})
    log_config["handlers"].setdefault("access", {"formatter": "access", "class": "logging.StreamHandler", "stream": "ext://sys.stdout"})
    log_config.setdefault("loggers", {})
    log_config["loggers"]["uvicorn"] = {"handlers": ["default"], "level": LOG_LEVEL_FROM_ENV.upper(), "propagate": False}
    log_config["loggers"]["uvicorn.error"] = {"handlers": ["default"], "level": "INFO", "propagate": False} # Uvicorn 自身的错误
    log_config["loggers"]["uvicorn.access"] = {"handlers": ["access"], "level": "WARNING", "propagate": False} # Uvicorn 访问日志

    logger.info(f"准备启动 Uvicorn 服务器: http://{APP_HOST}:{APP_PORT}")
    logger.info(f"开发模式自动重载 (通过命令行 --reload 控制): {DEV_RELOAD}")
    logger.info(f"应用日志级别 (EzTalkProxy.*): {LOG_LEVEL_FROM_ENV}")
    
    uvicorn.run(
        "eztalk_proxy.main:app",
        host=APP_HOST,
        port=APP_PORT,
        log_config=log_config,
        reload=DEV_RELOAD # uvicorn reload 参数
    )