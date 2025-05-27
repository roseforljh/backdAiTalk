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
    LOG_LEVEL_FROM_ENV, COMMON_HEADERS
)
# 导入您的主聊天路由
# /chat 端点将由 routers/chat.py 文件中的 router 对象定义和处理
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

# 设置其他模块的日志级别 (如果需要)
logging.getLogger("EzTalkProxy.SPHASANN").setLevel(LOG_LEVEL_FROM_ENV.upper())
for lib_logger_name in ["httpx", "httpcore", "googleapiclient.discovery_cache", "uvicorn.access", "watchfiles"]: # 添加 watchfiles
    logging.getLogger(lib_logger_name).setLevel(logging.WARNING)
logging.getLogger("uvicorn.error").setLevel(logging.INFO) # Uvicorn 自身的错误日志级别
# --- 日志配置结束 ---


# --- 应用生命周期管理 (Lifespan) ---
@asynccontextmanager
async def lifespan(app_instance: FastAPI):
    logger.info("Lifespan: 应用启动，开始初始化HTTP客户端...")
    client_local: Optional[httpx.AsyncClient] = None
    try:
        client_local = httpx.AsyncClient(
            timeout=httpx.Timeout(API_TIMEOUT, read=READ_TIMEOUT),
            limits=httpx.Limits(max_connections=MAX_CONNECTIONS),
            http2=True,
            follow_redirects=True,
            trust_env=False # 明确设置，除非您特意需要系统代理
        )
        app_instance.state.http_client = client_local # 将客户端实例存储在 app.state 中
        logger.info(f"Lifespan: HTTP客户端初始化成功。Timeout: {API_TIMEOUT}s, Read Timeout: {READ_TIMEOUT}s, Max Connections: {MAX_CONNECTIONS}")
    except Exception as e:
        logger.error(f"Lifespan: HTTP客户端初始化失败: {e}", exc_info=True)
        app_instance.state.http_client = None
    
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
    
    app_instance.state.http_client = None # 清理状态
    logger.info("Lifespan: 应用关闭流程完成。")


# --- FastAPI 应用实例 ---
app = FastAPI(
    title="EzTalk Proxy",
    description=f"代理服务，版本: {APP_VERSION}",
    version=APP_VERSION,
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc"
)

# --- CORS 中间件配置 ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # 生产环境建议指定具体来源
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"] # 按需配置需要暴露的头部
)
logger.info(f"FastAPI EzTalk Proxy v{APP_VERSION} 初始化完成，已配置CORS。")

# --- 包含主路由 ---
# routers/chat.py 将处理 /chat 路径的请求，并在内部根据模型名称分发逻辑
app.include_router(chat_router.router)


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
    elif not hasattr(client_from_state, 'is_closed'):
        client_status = "error"
        detail_message = f"Unexpected object in app.state.http_client: {type(client_from_state)}"

    response_data = {"status": client_status, "detail": detail_message, "app_version": APP_VERSION}
    return response_data


# --- Uvicorn 启动配置 (当直接运行此文件时) ---
if __name__ == "__main__":
    import uvicorn

    APP_HOST = os.getenv("HOST", "0.0.0.0")
    APP_PORT = int(os.getenv("PORT", 7860)) # 与您日志中的端口保持一致
    # RELOAD_DELAY = float(os.getenv("RELOAD_DELAY", "1.0")) # (可选) uvicorn --reload-delay
    # WORKERS = int(os.getenv("WORKERS", "1")) # (可选) uvicorn --workers

    # 开发模式自动重载，从环境变量读取，默认为False
    # 注意：Uvicorn 的 --reload 标志通常在命令行传递，而不是在代码中配置 reload=True
    # 如果要通过代码控制，可能需要不同的启动方式或针对 uvicorn.Server 的更底层配置
    # 这里我们假设 --reload 是通过命令行参数传递给 uvicorn 的
    DEV_RELOAD = os.getenv("DEV_RELOAD", "false").lower() == "true"


    # Uvicorn 日志配置 (与您提供的基本一致)
    log_config = uvicorn.config.LOGGING_CONFIG.copy()
    log_config["formatters"].setdefault("default", {"fmt": "%(levelprefix)s %(asctime)s [%(name)s] - %(message)s", "datefmt": "%Y-%m-%d %H:%M:%S", "use_colors": None})
    log_config["formatters"]["default"]["fmt"] = "%(asctime)s %(levelname)-8s [%(name)s:%(module)s:%(lineno)d] - %(message)s"
    
    log_config["formatters"].setdefault("access", {"fmt": "", "datefmt": "", "use_colors": None})
    log_config["formatters"]["access"]["fmt"] = '%(asctime)s %(levelname)-8s [%(name)s] - %(client_addr)s - "%(request_line)s" %(status_code)s' # noqa: E501
    log_config["formatters"]["access"]["datefmt"] = "%Y-%m-%d %H:%M:%S"
    
    log_config["handlers"].setdefault("default", {"formatter": "default", "class": "logging.StreamHandler", "stream": "ext://sys.stderr"}) # noqa: E501
    log_config["handlers"].setdefault("access", {"formatter": "access", "class": "logging.StreamHandler", "stream": "ext://sys.stdout"}) # noqa: E501
    
    log_config.setdefault("loggers", {})
    log_config["loggers"]["uvicorn"] = {"handlers": ["default"], "level": LOG_LEVEL_FROM_ENV.upper(), "propagate": False}
    log_config["loggers"]["uvicorn.error"] = {"handlers": ["default"], "level": "INFO", "propagate": False}
    log_config["loggers"]["uvicorn.access"] = {"handlers": ["access"], "level": "WARNING", "propagate": False}


    logger.info(f"准备启动 Uvicorn 服务器: http://{APP_HOST}:{APP_PORT}")
    logger.info(f"开发模式自动重载 (通过命令行 --reload 控制): {DEV_RELOAD}") # 提示DEV_RELOAD的来源
    logger.info(f"应用日志级别 (EzTalkProxy.*): {LOG_LEVEL_FROM_ENV}")
    
    uvicorn.run(
        "eztalk_proxy.main:app", # 指向 FastAPI app 实例的正确路径
        host=APP_HOST,
        port=APP_PORT,
        log_config=log_config,
        reload=DEV_RELOAD # 如果要通过代码控制reload，Uvicorn的这个参数可能不直接生效于其主进程
                          # 通常 --reload 是 uvicorn 命令行的参数
    )