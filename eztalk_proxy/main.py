# eztalk_proxy/main.py
import os
import logging
import httpx
from fastapi import FastAPI, Request # Request 已在您的版本中正确导入
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from typing import Optional

# 导入您的配置
from .config import (
    APP_VERSION, API_TIMEOUT, READ_TIMEOUT, MAX_CONNECTIONS,
    LOG_LEVEL_FROM_ENV, COMMON_HEADERS # 假设 COMMON_HEADERS 在 config.py 中定义
)
# 导入您的路由
# 主要的 /chat 端点在 chat_router 中定义
from .routers import chat as chat_router
# 我们稍后会在 chat_router 内部调用 multimodal_chat 中的逻辑
# from .routers import multimodal_chat as multimodal_chat_router # 暂时不在main.py中直接include

# --- 日志配置 (与您提供的保持一致) ---
numeric_level = getattr(logging, LOG_LEVEL_FROM_ENV.upper(), logging.INFO) # 确保大写
logging.basicConfig(
    level=numeric_level,
    format='%(asctime)s %(levelname)-8s [%(name)s:%(module)s:%(lineno)d] - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger("EzTalkProxy.Main")

logging.getLogger("EzTalkProxy.SPHASANN").setLevel(LOG_LEVEL_FROM_ENV.upper()) # 确保大写

for lib_logger_name in ["httpx", "httpcore", "googleapiclient.discovery_cache", "uvicorn.access"]:
    logging.getLogger(lib_logger_name).setLevel(logging.WARNING)
logging.getLogger("uvicorn.error").setLevel(logging.INFO)
# --- 日志配置结束 ---

# http_client 的全局声明和 lifespan 管理与您提供的保持一致
# global http_client 声明不是必须的，因为我们通过 app.state 传递
# http_client: Optional[httpx.AsyncClient] = None

@asynccontextmanager
async def lifespan(app_instance: FastAPI):
    logger.info("Lifespan: 初始化HTTP客户端...")
    client_local: Optional[httpx.AsyncClient] = None
    try:
        client_local = httpx.AsyncClient(
            timeout=httpx.Timeout(API_TIMEOUT, read=READ_TIMEOUT),
            limits=httpx.Limits(max_connections=MAX_CONNECTIONS),
            http2=True,
            follow_redirects=True,
            trust_env=False # 明确设置不信任环境变量中的代理设置，除非您特意需要
        )
        app_instance.state.http_client = client_local
        # http_client = client_local # 全局变量赋值可以去掉，推荐使用 app.state
        logger.info("Lifespan: HTTP客户端初始化成功。")
    except Exception as e:
        logger.error(f"Lifespan: HTTP客户端初始化失败: {e}", exc_info=True)
        app_instance.state.http_client = None # 确保出错时 state 中的也是 None
        # http_client = None
    
    yield # FastAPI 应用在此运行

    logger.info("Lifespan: 关闭HTTP客户端...")
    # 从 app.state 获取客户端进行关闭
    client_to_close = getattr(app_instance.state, "http_client", None)
    if client_to_close and hasattr(client_to_close, "is_closed") and not client_to_close.is_closed:
        try:
            await client_to_close.aclose()
            logger.info("Lifespan: HTTP客户端成功关闭。")
        except Exception as e:
            logger.error(f"Lifespan: 关闭HTTP客户端错误: {e}", exc_info=True)
    elif client_to_close and hasattr(client_to_close, 'is_closed') and client_to_close.is_closed:
        logger.info("Lifespan: HTTP客户端已经关闭。")
    else:
        logger.warning("Lifespan: HTTP客户端未找到或无法关闭。")
    
    app_instance.state.http_client = None # 清理状态
    # http_client = None
    logger.info("Lifespan: 关闭流程完成。")


app = FastAPI(
    title="EzTalk Proxy",
    description=f"代理服务，版本: {APP_VERSION}",
    version=APP_VERSION,
    lifespan=lifespan, # 应用生命周期管理
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS 中间件 (与您提供的保持一致)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # 生产环境建议指定具体来源
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"] # 谨慎暴露头部
)
logger.info(f"FastAPI EzTalk Proxy v{APP_VERSION} 初始化完成，已配置CORS。")

# 包含主要的聊天路由
# /chat 端点由 routers/chat.py 文件中的 router 对象定义和处理
# chat.py 内部将负责判断模型类型并分发给 Gemini 或非 Gemini 的处理逻辑
app.include_router(chat_router.router)


@app.get("/health", status_code=200, include_in_schema=False, tags=["Utilities"])
async def health_check(request: Request):
    logger.info("Health check endpoint called.")
    client_from_state = getattr(request.app.state, "http_client", None)
    client_status = "ok"
    detail_message = "HTTP client initialized and open."

    if client_from_state is None:
        client_status = "warning"
        detail_message = "HTTP client not initialized."
    elif hasattr(client_from_state, 'is_closed') and client_from_state.is_closed:
        client_status = "warning"
        detail_message = "HTTP client is closed."
    
    response_data = {"status": client_status, "detail": detail_message, "app_version": APP_VERSION}
    logger.info(f"Health check response: {response_data}")
    return response_data


if __name__ == "__main__":
    import uvicorn

    APP_HOST = os.getenv("HOST", "0.0.0.0")
    APP_PORT = int(os.getenv("PORT", 7860)) # 您日志中是7860，但您代码中是8000，统一一下
    DEV_RELOAD = os.getenv("DEV_RELOAD", "false").lower() == "true"

    # Uvicorn 日志配置 (与您提供的保持一致)
    log_config = uvicorn.config.LOGGING_CONFIG.copy()
    log_config["formatters"].setdefault("default", {"fmt": "", "datefmt": "", "use_colors": None})
    log_config["formatters"]["default"]["fmt"] = "%(asctime)s %(levelname)-8s [%(name)s] - %(message)s"
    log_config["formatters"]["default"]["datefmt"] = "%Y-%m-%d %H:%M:%S"
    log_config["formatters"].setdefault("access", {"fmt": "", "datefmt": "", "use_colors": None})
    log_config["formatters"]["access"]["fmt"] = '%(asctime)s %(levelname)-8s [%(name)s] - %(client_addr)s - "%(request_line)s" %(status_code)s' # noqa: E501
    log_config["formatters"]["access"]["datefmt"] = "%Y-%m-%d %H:%M:%S"
    log_config["handlers"].setdefault("default", {"formatter": "default", "class": "logging.StreamHandler", "stream": "ext://sys.stderr"}) # noqa: E501
    log_config["handlers"].setdefault("access", {"formatter": "access", "class": "logging.StreamHandler", "stream": "ext://sys.stdout"}) # noqa: E501
    log_config.setdefault("loggers", {})
    log_config["loggers"]["uvicorn.error"] = {"handlers": ["default"], "level": "INFO", "propagate": False}
    log_config["loggers"]["uvicorn.access"] = {"handlers": ["access"], "level": "WARNING", "propagate": False}


    logger.info(f"Starting Uvicorn server: http://{APP_HOST}:{APP_PORT}")
    logger.info(f"Development reload: {DEV_RELOAD}")
    logger.info(f"Application Log Level (EzTalkProxy.*): {LOG_LEVEL_FROM_ENV}")
    uvicorn.run(
        "eztalk_proxy.main:app", # 指向 FastAPI app 实例
        host=APP_HOST,
        port=APP_PORT,
        log_config=log_config,
        reload=DEV_RELOAD
    )