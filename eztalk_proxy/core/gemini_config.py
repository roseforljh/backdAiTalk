"""
Gemini-specific configuration for enhanced markdown and math processing
"""
import os

# Gemini enhancement features
GEMINI_MATH_ENHANCEMENT_ENABLED = os.getenv("GEMINI_MATH_ENHANCEMENT_ENABLED", "true").lower() == "true"
GEMINI_TABLE_ENHANCEMENT_ENABLED = os.getenv("GEMINI_TABLE_ENHANCEMENT_ENABLED", "true").lower() == "true"
GEMINI_CODE_ENHANCEMENT_ENABLED = os.getenv("GEMINI_CODE_ENHANCEMENT_ENABLED", "true").lower() == "true"
GEMINI_ENHANCED_PROMPTS_ENABLED = os.getenv("GEMINI_ENHANCED_PROMPTS_ENABLED", "true").lower() == "true"

# Processing parameters
GEMINI_BUFFER_SIZE = int(os.getenv("GEMINI_BUFFER_SIZE", "500"))
GEMINI_MATH_DELIMITER_CHECK = os.getenv("GEMINI_MATH_DELIMITER_CHECK", "true").lower() == "true"
GEMINI_TABLE_AUTO_SEPARATOR = os.getenv("GEMINI_TABLE_AUTO_SEPARATOR", "true").lower() == "true"

# Performance settings
GEMINI_CACHE_PROCESSED_CONTENT = os.getenv("GEMINI_CACHE_PROCESSED_CONTENT", "true").lower() == "true"
GEMINI_MAX_CACHE_SIZE = int(os.getenv("GEMINI_MAX_CACHE_SIZE", "1000"))

# Debug settings
GEMINI_DEBUG_PROCESSING = os.getenv("GEMINI_DEBUG_PROCESSING", "false").lower() == "true"
GEMINI_LOG_FIXES = os.getenv("GEMINI_LOG_FIXES", "false").lower() == "true"