# eztalk_proxy/multimodal_api_helpers.py
import logging
from typing import List, Dict, Any, Optional, Union, Tuple

# Pydantic 模型
from eztalk_proxy.models import ( # 使用绝对导入
    ChatRequestModel,
    PartsApiMessagePy,
    # AppStreamEventPy # Not used for request preparation
)
from eztalk_proxy.multimodal_models import ( # 使用绝对导入
    PyTextContentPart,
    PyFileUriContentPart,
    PyInlineDataContentPart
)
from eztalk_proxy.config import GOOGLE_API_BASE_URL # 使用绝对导入
from eztalk_proxy.utils import is_gemini_2_5_model # 使用绝对导入

logger = logging.getLogger("EzTalkProxy.MultimodalAPIHelpers")


def convert_parts_messages_to_rest_api_contents(
    messages: List[PartsApiMessagePy],
    request_id: str
) -> List[Dict[str, Any]]:
    log_prefix = f"RID-{request_id}"
    rest_api_contents: List[Dict[str, Any]] = []

    for i, msg in enumerate(messages):
        if not isinstance(msg, PartsApiMessagePy):
            logger.warning(f"{log_prefix}: Expected PartsApiMessagePy at index {i}, got {type(msg)}. Skipping.")
            continue

        rest_parts: List[Dict[str, Any]] = []

        for actual_part in msg.parts:
            try:
                if isinstance(actual_part, PyTextContentPart):
                    rest_parts.append({"text": actual_part.text})
                elif isinstance(actual_part, PyInlineDataContentPart):
                    rest_parts.append({
                        "inlineData": {
                            "mimeType": actual_part.mime_type,
                            "data": actual_part.base64_data
                        }
                    })
                elif isinstance(actual_part, PyFileUriContentPart):
                    if actual_part.uri.startswith("gs://"):
                        rest_parts.append({
                            "fileData": {
                                "mimeType": actual_part.mime_type,
                                "fileUri": actual_part.uri
                            }
                        })
                    else:
                        logger.warning(f"{log_prefix}: HTTP/S URI '{actual_part.uri}' for FileUriPart. REST API support varies. Skipping for now.")
                else:
                    logger.warning(f"{log_prefix}: Unknown actual part type: {type(actual_part)}. Skipping part.")
            except Exception as e_part:
                logger.error(f"{log_prefix}: Error processing message part for REST API: {actual_part}, Error: {e_part}", exc_info=True)
        
        if rest_parts:
            role_for_api = msg.role
            if msg.role == "assistant":
                role_for_api = "model"
            elif msg.role == "tool": # For REST API, tool responses are often role "function" or "tool" (for function_response part)
                role_for_api = "function" # Assuming "function" role for tool call results
                if not msg.name and not any("functionResponse" in part for part in rest_parts if isinstance(part, dict)):
                    logger.warning(f"{log_prefix}: Message with role 'tool' (mapped to 'function') might be missing 'name' or 'functionResponse' structure.")


            if role_for_api not in ["user", "model", "function"]: # Common REST API roles for contents
                 logger.warning(f"{log_prefix}: Invalid role '{msg.role}' for Gemini REST API, mapping to 'user'.")
                 role_for_api = "user"
            rest_api_contents.append({"role": role_for_api, "parts": rest_parts})
        else:
            logger.warning(f"{log_prefix}: Message from role {msg.role} at index {i} resulted in no valid parts for REST API. Skipping.")
    return rest_api_contents

def prepare_gemini_rest_api_request( # Renamed function
    chat_input: ChatRequestModel,
    request_id: str
) -> Tuple[str, Dict[str, str], Dict[str, Any]]:
    log_prefix = f"RID-{request_id}"

    model_name = chat_input.model
    base_api_url = GOOGLE_API_BASE_URL.rstrip('/')
    # Use v1beta for streamGenerateContent, as it often has newer features like thinking
    target_url = f"{base_api_url}/v1beta/models/{model_name}:streamGenerateContent?key={chat_input.api_key}"
    target_url += "&alt=sse" # Request Server-Sent Events

    headers = {"Content-Type": "application/json"}

    json_payload: Dict[str, Any] = {}

    # Convert messages to REST API 'contents' structure
    parts_api_messages = [msg for msg in chat_input.messages if isinstance(msg, PartsApiMessagePy)]
    json_payload["contents"] = convert_parts_messages_to_rest_api_contents(parts_api_messages, request_id)

    # --- Generation Config & Thinking Config for REST API ---
    generation_config_rest: Dict[str, Any] = {}
    
    # Populate from chat_input.generation_config (GenerationConfigPy) if present
    if chat_input.generation_config:
        gc_in = chat_input.generation_config
        if gc_in.temperature is not None: generation_config_rest["temperature"] = gc_in.temperature
        if gc_in.top_p is not None: generation_config_rest["topP"] = gc_in.top_p # camelCase for REST
        if gc_in.max_output_tokens is not None: generation_config_rest["maxOutputTokens"] = gc_in.max_output_tokens
        # candidateCount, stopSequences can be added here

        if gc_in.thinking_config:
            tc_in = gc_in.thinking_config
           
            thinking_config_for_gen_config: Dict[str, Any] = {}
            if tc_in.include_thoughts is not None:
                thinking_config_for_gen_config["includeThoughts"] = tc_in.include_thoughts
                logger.info(f"{log_prefix}: REST API: Setting includeThoughts={tc_in.include_thoughts} in thinkingConfig.")
            if tc_in.thinking_budget is not None:
                if "flash" in model_name.lower() or is_gemini_2_5_model(model_name): # Gemini 2.5 Flash supports budget
                    thinking_config_for_gen_config["thinkingBudget"] = tc_in.thinking_budget
                    logger.info(f"{log_prefix}: REST API: Setting thinkingBudget={tc_in.thinking_budget} in thinkingConfig.")
            
            if thinking_config_for_gen_config: # If any thinking params were set
                generation_config_rest["thinkingConfig"] = thinking_config_for_gen_config

    if "temperature" not in generation_config_rest and chat_input.temperature is not None:
        generation_config_rest["temperature"] = chat_input.temperature
    if "topP" not in generation_config_rest and chat_input.top_p is not None:
        generation_config_rest["topP"] = chat_input.top_p
    if "maxOutputTokens" not in generation_config_rest and chat_input.max_tokens is not None:
        generation_config_rest["maxOutputTokens"] = chat_input.max_tokens

    # Add generation_config_rest to payload if it has any content
    if generation_config_rest:
        json_payload["generationConfig"] = generation_config_rest


    if chat_input.tools:
        
        logger.warning(f"{log_prefix}: REST API: Tool configuration needs to be implemented based on Gemini REST API specs.")
       

    logger.info(
        f"{log_prefix}: Prepared Gemini REST API request. URL: {target_url.split('?key=')[0]}... "
        f"Payload keys: {list(json_payload.keys())}"
    )
    if "generationConfig" in json_payload:
        logger.info(f"{log_prefix}: generationConfig in REST payload: {json_payload['generationConfig']}")

    return target_url, headers, json_payload