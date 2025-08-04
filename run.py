import uvicorn
import os
import logging

if __name__ == "__main__":
    # This setup allows running the app directly from the project root
    
    LOG_LEVEL_FROM_ENV = os.getenv("LOG_LEVEL", "INFO").upper()
    
    # Basic logging config for the runner script itself
    logging.basicConfig(
        level=getattr(logging, LOG_LEVEL_FROM_ENV, logging.INFO),
        format='%(asctime)s %(levelname)-8s [RUNNER] - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    APP_HOST = os.getenv("HOST", "0.0.0.0")
    APP_PORT = int(os.getenv("PORT", 7860))
    
    logging.info(f"Starting EzTalk Proxy server...")
    logging.info(f"Host: {APP_HOST}, Port: {APP_PORT}")
    logging.info(f"Log Level: {LOG_LEVEL_FROM_ENV}")
    
    try:
        uvicorn.run(
            "eztalk_proxy.main:app",
            host=APP_HOST,
            port=APP_PORT,
            log_level=LOG_LEVEL_FROM_ENV.lower(),
            reload=False,  # 生产环境不使用reload
            workers=1,     # 单worker避免资源竞争
            access_log=True,
            server_header=False,
            date_header=False
        )
    except Exception as e:
        logging.error(f"Failed to start server: {e}")
        raise