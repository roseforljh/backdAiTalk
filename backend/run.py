import uvicorn
import os
import logging

if __name__ == "__main__":
    # This setup allows running the app directly from the project root
    # using `python -m backdAiTalk.run`
    
    LOG_LEVEL_FROM_ENV = os.getenv("LOG_LEVEL", "DEBUG").upper()
    
    # Basic logging config for the runner script itself
    logging.basicConfig(
        level=LOG_LEVEL_FROM_ENV,
        format='%(asctime)s %(levelname)-8s [RUNNER] - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    APP_HOST = os.getenv("HOST", "0.0.0.0")
    APP_PORT = int(os.getenv("PORT", 7860))
    
    logging.info(f"Preparing to start Uvicorn server for the application.")
    logging.info(f"Host: {APP_HOST}, Port: {APP_PORT}")
    logging.info(f"Log Level: {LOG_LEVEL_FROM_ENV}")
    
    uvicorn.run(
        "eztalk_proxy.main:app",
        host=APP_HOST,
        port=APP_PORT,
        log_level=LOG_LEVEL_FROM_ENV.lower(),
        reload=os.getenv("DEV_RELOAD", "false").lower() == "true"
    )