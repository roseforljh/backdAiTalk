import os
import sys
import logging
import time

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)-8s [%(name)s:%(module)s:%(lineno)d] - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    stream=sys.stdout # Ensure logs go to stdout for Cloud Run
)
logger = logging.getLogger("TestStartupScript")

logger.info("--- PYTHON STARTUP TEST SCRIPT RUNNING ---")
logger.info(f"Python version: {sys.version}")
logger.info("All Environment Variables (sensitive values redacted for logging):")
for k, v in os.environ.items():
    # Basic redaction for common sensitive patterns
    if "KEY" in k.upper() or "SECRET" in k.upper() or "TOKEN" in k.upper() or "PASSWORD" in k.upper():
        logger.info(f"  {k}=******")
    else:
        logger.info(f"  {k}={v}")

logger.info(f"PORT env var from os.getenv('PORT'): {os.getenv('PORT')}")
logger.info(f"GOOGLE_API_KEY env var exists in os.environ: {bool(os.getenv('GOOGLE_API_KEY'))}")
logger.info(f"GOOGLE_CSE_ID env var exists in os.environ: {bool(os.getenv('GOOGLE_CSE_ID'))}")

# Attempt to import config to see if it crashes at import time
try:
    logger.info("Attempting to import eztalk_proxy.config...")
    from eztalk_proxy import config
    logger.info("Successfully imported eztalk_proxy.config.")
    if hasattr(config, 'GOOGLE_API_KEY_ENV'):
        logger.info(f"  config.GOOGLE_API_KEY_ENV is set (via os.getenv('GOOGLE_API_KEY')): {bool(config.GOOGLE_API_KEY_ENV)}")
    else:
        logger.warning("  config.GOOGLE_API_KEY_ENV attribute not found.")
    if hasattr(config, 'GOOGLE_CSE_ID'):
        logger.info(f"  config.GOOGLE_CSE_ID is set: {bool(config.GOOGLE_CSE_ID)}")
    else:
        logger.warning("  config.GOOGLE_CSE_ID attribute not found.")
    if hasattr(config, 'API_TIMEOUT'):
        logger.info(f"  config.API_TIMEOUT (type: {type(config.API_TIMEOUT)}): {config.API_TIMEOUT}")
    else:
        logger.warning("  config.API_TIMEOUT attribute not found.")
    if hasattr(config, 'LOG_LEVEL_FROM_ENV'):
        logger.info(f"  config.LOG_LEVEL_FROM_ENV: {config.LOG_LEVEL_FROM_ENV}")
    else:
        logger.warning("  config.LOG_LEVEL_FROM_ENV attribute not found.")

except Exception as e_config:
    logger.error("Failed to import or access attributes from eztalk_proxy.config:", exc_info=True)

# Attempt to import main module (but not run app)
try:
    logger.info("Attempting to import eztalk_proxy.main (as main_module)...")
    # Ensure relative imports work correctly if this script is run directly
    # This might require adjusting PYTHONPATH or how the script is called if run outside Docker context
    # For Docker CMD ["python", "eztalk_proxy/test_startup.py"], APP_HOME is /app, 
    # and eztalk_proxy is a package inside it, so direct import should work.
    from eztalk_proxy import main as main_module 
    logger.info("Successfully imported eztalk_proxy.main.")
    if hasattr(main_module, 'app'):
        logger.info("eztalk_proxy.main has an 'app' FastAPI attribute.")
    else:
        logger.warning("eztalk_proxy.main does NOT have an 'app' FastAPI attribute.")

except Exception as e_main:
    logger.error("Failed to import eztalk_proxy.main:", exc_info=True)
    
logger.info("--- PYTHON STARTUP TEST SCRIPT FINISHED ---")

# Note: This script will exit after printing logs. 
# Cloud Run's TCP health check will likely fail because no server is started on PORT.
# The primary goal is to see if this script *runs at all* and what it logs,
# especially during the import attempts. If we see these logs, Python itself is working.
# If we see tracebacks during imports, we've found a problem.
# If we see *no logs at all* from this script, the issue is very early (e.g. Python interpreter itself, or file not found).