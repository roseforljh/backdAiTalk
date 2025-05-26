# multimodal_models.py
# (可以放在 config 目录或与 api_helpers 同级)

KNOWN_OPENAI_MULTIMODAL_MODELS = {
    "gpt-4-vision-preview", "gpt-4o", "gpt-4-turbo",
}
KNOWN_GOOGLE_MULTIMODAL_MODELS = {
    "gemini-1.0-pro-vision-latest", "gemini-pro-vision",
    "gemini-1.5-pro-latest", "gemini-1.5-flash-latest",
}

def is_model_multimodal(provider: str, model_name: str) -> bool:
    model_lower = model_name.lower()
    if provider == "openai":
        return model_lower in KNOWN_OPENAI_MULTIMODAL_MODELS
    elif provider == "google":
        # 你的 is_gemini_2_5_model 也可以在这里被调用或集成
        # from ..utils import is_gemini_2_5_model # 假设可以导入
        # if is_gemini_2_5_model(model_name): return True
        return model_lower in KNOWN_GOOGLE_MULTIMODAL_MODELS
    return False