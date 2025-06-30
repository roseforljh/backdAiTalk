import orjson
import logging
from typing import List, Dict, Any, Optional, Union, Tuple
from urllib.parse import urljoin

from ..models.api_models import (
    ChatRequestModel,
    SimpleTextApiMessagePy,
    PartsApiMessagePy,
    PyTextContentPart,
    PyFileUriContentPart,
    PyInlineDataContentPart
)
from ..core.config import (
    DEFAULT_OPENAI_API_BASE_URL,
    OPENAI_COMPATIBLE_PATH,
    GOOGLE_API_BASE_URL
)
# Assuming prompts will be moved and consolidated
from ..prompts.katex import KATEX_FORMATTING_INSTRUCTION, DEEPSEEK_KATEX_FORMATTING_INSTRUCTION, QWEN_KATEX_FORMATTING_INSTRUCTION
from ..prompts.supreme_intelligence_advisor import SUPREME_INTELLIGENCE_ADVISOR_PROMPT
from ..utils.helpers import is_gemini_2_5_model

logger = logging.getLogger("EzTalkProxy.Services.RequestBuilder")

def prepare_openai_request(
    request_data: ChatRequestModel,
    processed_messages: List[Dict[str, Any]],
    request_id: str
) -> Tuple[str, Dict[str, str], Dict[str, Any]]:
    base_url = (request_data.api_address or DEFAULT_OPENAI_API_BASE_URL).strip().rstrip('/')
    target_url = urljoin(f"{base_url}/", OPENAI_COMPATIBLE_PATH.lstrip('/'))

    headers = {
        "Authorization": f"Bearer {request_data.api_key}",
        "Content-Type": "application/json",
        "Accept": "text/event-stream"
    }

    final_messages = list(processed_messages)
    model_name_lower = request_data.model.lower()
    instruction = ""
    if "qwen" in model_name_lower:
        instruction = QWEN_KATEX_FORMATTING_INSTRUCTION
    elif "deepseek" in model_name_lower:
        instruction = DEEPSEEK_KATEX_FORMATTING_INSTRUCTION
    else:
        instruction = KATEX_FORMATTING_INSTRUCTION

    system_message_index = -1
    for i, msg in enumerate(final_messages):
        if msg.get("role") == "system":
            system_message_index = i
            break
    
    if system_message_index != -1:
        content = final_messages[system_message_index].get("content", "")
        if isinstance(content, str) and instruction not in content:
            final_messages[system_message_index]["content"] = f"{content}\n\n{instruction}".strip()
    else:
        final_messages.insert(0, {"role": "system", "content": instruction})

    payload: Dict[str, Any] = {
        "model": request_data.model,
        "messages": final_messages,
        "stream": True,
    }

    gen_conf = request_data.generation_config
    if gen_conf:
        payload.update({
            "temperature": gen_conf.temperature,
            "top_p": gen_conf.top_p,
            "max_tokens": gen_conf.max_output_tokens,
        })

    payload.update({
        "temperature": payload.get("temperature") or request_data.temperature,
        "top_p": payload.get("top_p") or request_data.top_p,
        "max_tokens": payload.get("max_tokens") or request_data.max_tokens,
        "tools": request_data.tools,
        "tool_choice": request_data.tool_choice,
    })
    
    payload = {k: v for k, v in payload.items() if v is not None}

    if "qwen" in model_name_lower and isinstance(request_data.qwen_enable_search, bool):
        payload["enable_search"] = request_data.qwen_enable_search

    if request_data.custom_model_parameters:
        for key, value in request_data.custom_model_parameters.items():
            if key not in payload:
                payload[key] = value
    
    if request_data.custom_extra_body:
        payload.update(request_data.custom_extra_body)
        
    return target_url, headers, payload

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
                    logger.warning(f"{log_prefix}: Unknown actual part type during conversion: {type(actual_part)}. Content: {str(actual_part)[:100]}. Skipping part.")
            except Exception as e_part:
                logger.error(f"{log_prefix}: Error processing message part for REST API: {actual_part}, Error: {e_part}", exc_info=True)
        
        if rest_parts:
            role_for_api = msg.role
            if msg.role == "assistant":
                role_for_api = "model"
            elif msg.role == "tool":
                role_for_api = "function"
            
            if role_for_api not in ["user", "model", "function"]:
                 logger.warning(f"{log_prefix}: Mapping role '{msg.role}' to 'user' for Gemini REST API contents (current role_for_api: {role_for_api}).")
                 role_for_api = "user"
            
            content_to_add = {"role": role_for_api, "parts": rest_parts}
            rest_api_contents.append(content_to_add)
        else:
            logger.warning(f"{log_prefix}: Message from role {msg.role} at index {i} resulted in no valid parts for REST API. Skipping.")
    
    return rest_api_contents

def prepare_gemini_rest_api_request(
    chat_input: ChatRequestModel,
    request_id: str
) -> Tuple[str, Dict[str, str], Dict[str, Any]]:
    log_prefix = f"RID-{request_id}"
    logger.info(f"{log_prefix}: Preparing Gemini REST API request for model {chat_input.model}.")

    model_name = chat_input.model
    base_api_url = GOOGLE_API_BASE_URL.rstrip('/')
    target_url = f"{base_api_url}/v1beta/models/{model_name}:streamGenerateContent?key={chat_input.api_key}&alt=sse"

    headers = {"Content-Type": "application/json"}
    json_payload: Dict[str, Any] = {}
    
    messages_to_convert_or_use: List[PartsApiMessagePy] = []
    for msg_abstract in chat_input.messages:
        if isinstance(msg_abstract, PartsApiMessagePy):
            messages_to_convert_or_use.append(msg_abstract)
        elif isinstance(msg_abstract, SimpleTextApiMessagePy):
            text_part = PyTextContentPart(type="text_content", text=msg_abstract.content or "")
            parts_message_equivalent = PartsApiMessagePy(
                role=msg_abstract.role,
                message_type="parts_message",
                parts=[text_part],
                name=msg_abstract.name,
                tool_calls=msg_abstract.tool_calls,
                tool_call_id=msg_abstract.tool_call_id
            )
            messages_to_convert_or_use.append(parts_message_equivalent)
        else:
            logger.warning(f"{log_prefix}: Encountered unknown message type {type(msg_abstract)} in chat_input.messages during Gemini REST prep. Skipping.")

    if not messages_to_convert_or_use:
        logger.error(f"{log_prefix}: No processable messages found for Gemini REST request.")
        json_payload["contents"] = []
    else:
        json_payload["contents"] = convert_parts_messages_to_rest_api_contents(messages_to_convert_or_use, request_id)

    json_payload["system_instruction"] = {
        "parts": [{"text": SUPREME_INTELLIGENCE_ADVISOR_PROMPT}]
    }

    generation_config_rest: Dict[str, Any] = {}
    if chat_input.generation_config:
        gc_in = chat_input.generation_config
        if gc_in.temperature is not None: generation_config_rest["temperature"] = gc_in.temperature
        if gc_in.top_p is not None: generation_config_rest["topP"] = gc_in.top_p
        if gc_in.max_output_tokens is not None: generation_config_rest["maxOutputTokens"] = gc_in.max_output_tokens
        if gc_in.thinking_config:
            tc_in = gc_in.thinking_config
            thinking_config_for_gen_config: Dict[str, Any] = {}
            if tc_in.include_thoughts is not None:
                thinking_config_for_gen_config["includeThoughts"] = tc_in.include_thoughts
            if tc_in.thinking_budget is not None and ("flash" in model_name.lower() or is_gemini_2_5_model(model_name)):
                thinking_config_for_gen_config["thinkingBudget"] = tc_in.thinking_budget
            if thinking_config_for_gen_config:
                generation_config_rest["thinkingConfig"] = thinking_config_for_gen_config
    
    if "temperature" not in generation_config_rest and chat_input.temperature is not None:
        generation_config_rest["temperature"] = chat_input.temperature
    if "topP" not in generation_config_rest and chat_input.top_p is not None:
        generation_config_rest["topP"] = chat_input.top_p
    if "maxOutputTokens" not in generation_config_rest and chat_input.max_tokens is not None:
        generation_config_rest["maxOutputTokens"] = chat_input.max_tokens
    
    if generation_config_rest:
        json_payload["generationConfig"] = generation_config_rest

    if chat_input.tools:
        gemini_tools_payload = []
        converted_declarations = []
        for tool_entry in chat_input.tools:
            if tool_entry.get("type") == "function" and "function" in tool_entry:
                func_data = tool_entry["function"]
                declaration = {
                    "name": func_data.get("name"),
                    "description": func_data.get("description"),
                    "parameters": func_data.get("parameters")
                }
                declaration = {k: v for k, v in declaration.items() if v is not None}
                if "name" in declaration and "description" in declaration :
                    converted_declarations.append(declaration)
        
        if converted_declarations:
            gemini_tools_payload.append({"functionDeclarations": converted_declarations})
            json_payload["tools"] = gemini_tools_payload

            if chat_input.tool_choice:
                tool_config_payload: Dict[str, Any] = {}
                if isinstance(chat_input.tool_choice, str):
                    choice_str = chat_input.tool_choice.upper()
                    if choice_str in ["AUTO", "ANY", "NONE"]:
                        tool_config_payload = {"mode": choice_str}
                    elif choice_str == "REQUIRED":
                        tool_config_payload = {"mode": "ANY"}
                elif isinstance(chat_input.tool_choice, dict) and chat_input.tool_choice.get("type") == "function":
                    func_choice = chat_input.tool_choice.get("function", {})
                    func_name = func_choice.get("name")
                    if func_name:
                        tool_config_payload = {"mode": "ANY", "allowedFunctionNames": [func_name]}
                
                if tool_config_payload:
                    if "generationConfig" not in json_payload:
                        json_payload["generationConfig"] = {}
                    json_payload["generationConfig"]["toolConfig"] = {"functionCallingConfig": tool_config_payload}

    logger.info(f"{log_prefix}: Prepared Gemini REST API request. URL: {target_url.split('?key=')[0]}... Payload keys: {list(json_payload.keys())}")
    if "generationConfig" in json_payload: logger.info(f"{log_prefix}: generationConfig in REST payload: {json_payload['generationConfig']}")

    return target_url, headers, json_payload