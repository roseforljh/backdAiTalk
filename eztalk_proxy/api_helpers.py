# eztalk_proxy/api_helpers.py
import logging
import orjson # orjson is used by _convert_simple_text_messages_to_google_contents
from typing import List, Dict, Any, Optional, Union, Tuple

# 使用绝对导入
from eztalk_proxy.models import ChatRequestModel, SimpleTextApiMessagePy
from eztalk_proxy.config import DEFAULT_OPENAI_API_BASE_URL, OPENAI_COMPATIBLE_PATH
from eztalk_proxy.katex_prompt import KATEX_FORMATTING_INSTRUCTION
from eztalk_proxy.utils import is_gemini_2_5_model # is_gemini_2_5_model is used in prepare_google_request_payload_structure

logger = logging.getLogger("EzTalkProxy.APIHelpers")

def prepare_openai_request(
    request_data: ChatRequestModel,
    processed_messages: List[Dict[str, Any]], 
    request_id: str
) -> Tuple[str, Dict[str, str], Dict[str, Any]]:
    log_prefix = f"RID-{request_id}"

    # Determine the base URL
    raw_base_url = request_data.api_address.strip() if request_data.api_address and request_data.api_address.strip() else DEFAULT_OPENAI_API_BASE_URL
    base_url = raw_base_url.rstrip('/')

    # Determine the path to append, OPENAI_COMPATIBLE_PATH is typically "/v1/chat/completions"
    path_to_use = OPENAI_COMPATIBLE_PATH # Keep leading slash if present
    
    target_url: str

    # More generic URL construction:
    # If base_url already ends with the full desired path (e.g. .../v1/chat/completions), use base_url as is.
    if base_url.endswith(path_to_use):
        target_url = base_url
        logger.info(f"{log_prefix}: Base URL '{base_url}' already ends with full path '{path_to_use}'. Target URL: '{target_url}'")
    # If base_url ends with a part of the path (e.g. base_url ends with /v1, path_to_use is /v1/chat/completions)
    # then append the remaining part of path_to_use.
    elif path_to_use.startswith(base_url[base_url.rfind('/'):]) and base_url.count('/') > 2 : # Heuristic: base_url is more than just https://domain.com
        # This case is tricky. Example: base_url = "https://api.example.com/v1", path_to_use = "/v1/chat/completions"
        # We want "https://api.example.com/v1/chat/completions"
        # A simple way: if base_url ends with the first segment of path_to_use (e.g. /v1), append the rest.
        first_path_segment = path_to_use.split('/')[1] # e.g., "v1" from "/v1/chat/completions"
        if base_url.endswith(f"/{first_path_segment}"):
            remaining_path = '/'.join(path_to_use.split('/')[2:]) # e.g., "chat/completions"
            target_url = f"{base_url}/{remaining_path}"
            logger.info(f"{log_prefix}: Base URL '{base_url}' ends with first segment of path. Appending remaining path '{remaining_path}'. Target URL: '{target_url}'")
        else:
            target_url = f"{base_url}{path_to_use}" # Standard append if no smart logic matched
            logger.info(f"{log_prefix}: Appending full path. Base: '{base_url}', Path: '{path_to_use}', Target URL: '{target_url}'")
    else:
        # Default: append the full path, ensuring only one slash between base_url and path_to_use
        target_url = f"{base_url}{path_to_use}"
        logger.info(f"{log_prefix}: Standard concatenation. Base: '{base_url}', Path: '{path_to_use}', Target URL: '{target_url}'")


    headers = {
        "Authorization": f"Bearer {request_data.api_key}",
        "Content-Type": "application/json",
        "Accept": "text/event-stream"
    }

    final_openai_payload_msgs = []
    system_message_found_and_updated = False
    for m_dict in processed_messages:
        current_msg_payload = m_dict.copy() 

        if current_msg_payload.get("role") == "system":
            original_content = current_msg_payload.get("content", "")
            if isinstance(original_content, str):
                final_system_content = original_content
                if KATEX_FORMATTING_INSTRUCTION not in final_system_content:
                    final_system_content = (final_system_content + "\n\n" + KATEX_FORMATTING_INSTRUCTION).strip()
                current_msg_payload["content"] = final_system_content
            else:
                logger.warning(f"{log_prefix}: System message content is not a string, KaTeX not injected. Content: {original_content}")
            
            final_openai_payload_msgs.append(current_msg_payload)
            system_message_found_and_updated = True
        elif current_msg_payload.get("role") == "user" and isinstance(current_msg_payload.get("content"), list):
            logger.debug(f"{log_prefix}: User message with list content (likely vision) being passed.") 
            final_openai_payload_msgs.append(current_msg_payload)
        else:
            final_openai_payload_msgs.append(current_msg_payload)

    if not system_message_found_and_updated:
        final_openai_payload_msgs.insert(0, {"role": "system", "content": KATEX_FORMATTING_INSTRUCTION})
        logger.info(f"{log_prefix}: OpenAI Req: No system message found, prepended KaTeX instruction.")
    
    payload: Dict[str, Any] = {
        "model": request_data.model,
        "messages": final_openai_payload_msgs,
        "stream": True,
    }

    gen_conf = request_data.generation_config
    if gen_conf:
        if gen_conf.temperature is not None: payload["temperature"] = gen_conf.temperature
        if gen_conf.top_p is not None: payload["top_p"] = gen_conf.top_p
        if gen_conf.max_output_tokens is not None: payload["max_tokens"] = gen_conf.max_output_tokens
    
    if payload.get("temperature") is None and request_data.temperature is not None:
        payload["temperature"] = request_data.temperature
    if payload.get("top_p") is None and request_data.top_p is not None:
        payload["top_p"] = request_data.top_p
    if payload.get("max_tokens") is None and request_data.max_tokens is not None:
        payload["max_tokens"] = request_data.max_tokens

    if request_data.tools: payload["tools"] = request_data.tools
    if request_data.tool_choice: payload["tool_choice"] = request_data.tool_choice
    
    model_name_lower = request_data.model.lower()
    if hasattr(request_data, 'qwen_enable_search') and \
       request_data.qwen_enable_search is not None and \
       "qwen" in model_name_lower:
        if isinstance(request_data.qwen_enable_search, bool):
            payload["enable_search"] = request_data.qwen_enable_search
            logger.info(f"{log_prefix}: OpenAI Req: Applied 'enable_search={request_data.qwen_enable_search}' for Qwen model from 'qwen_enable_search' field.")
        else:
            logger.warning(f"{log_prefix}: OpenAI Req: 'qwen_enable_search' field was not a boolean ('{request_data.qwen_enable_search}'). Not applying.")

    if request_data.custom_model_parameters:
        logger.info(f"{log_prefix}: OpenAI Req: Processing custom_model_parameters: {request_data.custom_model_parameters}")
        for key, value in request_data.custom_model_parameters.items():
            if key == "enable_search" and "qwen" in model_name_lower and \
               hasattr(request_data, 'qwen_enable_search') and request_data.qwen_enable_search is not None:
                logger.warning(f"{log_prefix}: OpenAI Req: 'enable_search' for Qwen already handled by 'qwen_enable_search' field. Skipping from custom_model_parameters.")
                continue
            if key not in payload: 
                payload[key] = value
                logger.info(f"{log_prefix}: OpenAI Req: Applied custom parameter from map '{key}={value}'.")
            else:
                logger.warning(f"{log_prefix}: OpenAI Req: Custom parameter from map '{key}' conflicts with standard/qwen-specific payload key. NOT applied.")
    
    if request_data.custom_extra_body:
        logger.info(f"{log_prefix}: OpenAI Req: Applying custom_extra_body: {list(request_data.custom_extra_body.keys())}")
        for key, value in request_data.custom_extra_body.items():
            if key in payload and payload[key] != value : 
                logger.warning(f"{log_prefix}: OpenAI Req: custom_extra_body key '{key}' overwrites existing payload key from '{payload[key]}' to '{value}'.")
            elif key in payload and payload[key] == value:
                 logger.debug(f"{log_prefix}: OpenAI Req: custom_extra_body key '{key}' has same value as existing payload key. No change.")
            payload[key] = value
        
    logger.debug(f"{log_prefix}: Final OpenAI Request Payload (keys): {list(payload.keys())}")
    for m_idx, m_val in enumerate(payload.get("messages", [])):
        content_item = m_val.get("content")
        role_item = m_val.get("role")
        content_preview_str: str
        if isinstance(content_item, list):
            part_previews_list = []
            for p_idx, p_item in enumerate(content_item):
                if isinstance(p_item, dict):
                    part_type = p_item.get("type", "unknown_type")
                    if part_type == "text":
                        part_previews_list.append(f"TextPart[{p_idx}]: '{str(p_item.get('text',''))[:30]}...'")
                    elif part_type == "image_url":
                        url_data = p_item.get("image_url", {}).get("url", "")
                        preview_url = url_data[:70] + "..." if len(url_data) > 70 else url_data
                        part_previews_list.append(f"ImagePart[{p_idx}]: url='{preview_url}'")
                    else:
                         part_previews_list.append(f"Part[{p_idx}] Type='{part_type}'")
                else:
                    part_previews_list.append(f"UnknownPartData[{p_idx}]")
            content_preview_str = f"MultiPart: [{', '.join(part_previews_list)}]"
        elif isinstance(content_item, str):
            content_preview_str = content_item[:70] + "..." if len(content_item) > 70 else content_item
        else:
            content_preview_str = "UnknownOrNonSerializableContent"
        logger.debug(f"{log_prefix} - Message[{m_idx}] Role: {role_item}, Content Preview: {content_preview_str}")

    return target_url, headers, payload

# --- Google 相关辅助函数 ---
# (The rest of the file: _convert_simple_text_messages_to_google_contents, 
#  prepare_google_request_payload_structure, 
#  _convert_openai_tools_to_gemini_declarations, 
#  _convert_openai_tool_choice_to_gemini_tool_config 
#  remain the same as your last provided version of these functions.)

def _convert_simple_text_messages_to_google_contents(
    messages: List[SimpleTextApiMessagePy], request_id: str
) -> List[Dict[str, Any]]:
    log_prefix = f"RID-{request_id}"; google_contents: List[Dict[str, Any]] = []
    for msg in messages:
        if not isinstance(msg, SimpleTextApiMessagePy): continue
        role = "model" if msg.role == "assistant" else msg.role
        if role not in ["user", "model", "function"]: role = "user"
        parts = []
        if msg.content and msg.content.strip(): parts.append({"text": msg.content})
        
        if hasattr(msg, 'tool_calls') and msg.tool_calls and role == "model":
            for tc in msg.tool_calls:
                if tc.type == "function" and tc.function.name and tc.function.arguments is not None:
                    try: args_obj = orjson.loads(tc.function.arguments)
                    except orjson.JSONDecodeError: args_obj = {"error": "invalid json from assistant tool_call", "raw": tc.function.arguments}
                    parts.append({"functionCall": {"name": tc.function.name, "args": args_obj}})
        
        if msg.role == "tool" and msg.name and hasattr(msg, 'tool_call_id') and msg.tool_call_id and msg.content is not None:
            try: response_obj = orjson.loads(msg.content)
            except orjson.JSONDecodeError: response_obj = {"raw_response": msg.content, "detail": "Content not valid JSON for tool response."}
            role = "function" 
            parts.append({"functionResponse": {"name": msg.name, "response": response_obj}})

        if parts: google_contents.append({"role": role, "parts": parts})
    return google_contents

def prepare_google_request_payload_structure(
    rd: ChatRequestModel, 
    api_messages: List[SimpleTextApiMessagePy], 
    request_id: str
) -> Tuple[Dict[str, Any], bool]: 
    log_prefix = f"RID-{request_id}"
    logger.info(f"{log_prefix}: Preparing Google request payload (TEXT-ONLY/NON-GEMINI REST path) for model {rd.model}")
    
    generation_config_updates: Dict[str, Any] = {}
    is_native_gemini_thinking_active = False 
    
    system_instruction_parts = []
    user_facing_messages_simple: List[SimpleTextApiMessagePy] = []
    has_client_system_message = False

    for m_obj_dict in api_messages: # api_messages is List[Dict], convert to SimpleTextApiMessagePy if needed
        m_obj = SimpleTextApiMessagePy(**m_obj_dict) if isinstance(m_obj_dict, dict) else m_obj_dict
        if not isinstance(m_obj, SimpleTextApiMessagePy): # Ensure it's the correct type
            logger.warning(f"{log_prefix}: Skipping non-SimpleTextApiMessagePy object in prepare_google_request_payload_structure: {type(m_obj)}")
            continue

        if m_obj.role == "system" and m_obj.content and m_obj.content.strip(): 
            has_client_system_message = True
            system_content_with_katex = f"{m_obj.content.strip()}\n\n{KATEX_FORMATTING_INSTRUCTION}"
            system_instruction_parts.append(system_content_with_katex)
        else:
            user_facing_messages_simple.append(m_obj)

    if not has_client_system_message: 
        if not any(part.lower().strip() == KATEX_FORMATTING_INSTRUCTION.lower().strip() for part in system_instruction_parts):
            system_instruction_parts.append(KATEX_FORMATTING_INSTRUCTION)
    
    final_system_instruction_content = "\n\n".join(system_instruction_parts).strip() if system_instruction_parts else None
    
    google_api_contents = _convert_simple_text_messages_to_google_contents(user_facing_messages_simple, request_id)
    payload: Dict[str, Any] = {"contents": google_api_contents}

    if final_system_instruction_content:
        payload["systemInstruction"] = {"parts": [{"text": final_system_instruction_content}]}

    gen_conf_input = rd.generation_config
    if gen_conf_input:
        if gen_conf_input.temperature is not None: generation_config_updates["temperature"] = gen_conf_input.temperature
        if gen_conf_input.top_p is not None: generation_config_updates["topP"] = gen_conf_input.top_p
        if gen_conf_input.max_output_tokens is not None: generation_config_updates["maxOutputTokens"] = gen_conf_input.max_output_tokens
        if gen_conf_input.thinking_config and gen_conf_input.thinking_config.include_thoughts is not None:
            thinking_payload = {}
            if gen_conf_input.thinking_config.include_thoughts is not None:
                thinking_payload["includeThoughts"] = gen_conf_input.thinking_config.include_thoughts
            if gen_conf_input.thinking_config.thinking_budget is not None and is_gemini_2_5_model(rd.model):
                thinking_payload["thinkingBudget"] = gen_conf_input.thinking_config.thinking_budget
            if thinking_payload:
                generation_config_updates["thinkingConfig"] = thinking_payload
                is_native_gemini_thinking_active = bool(thinking_payload.get("includeThoughts"))
    else: 
        if rd.temperature is not None: generation_config_updates["temperature"] = rd.temperature
        if rd.top_p is not None: generation_config_updates["topP"] = rd.top_p
        if rd.max_tokens is not None: generation_config_updates["maxOutputTokens"] = rd.max_tokens
        
    if rd.tools:
        gemini_declarations = _convert_openai_tools_to_gemini_declarations(rd.tools, request_id)
        if gemini_declarations:
            payload["tools"] = [{"functionDeclarations": gemini_declarations}]
            if rd.tool_choice: 
                tool_config_converted = _convert_openai_tool_choice_to_gemini_tool_config(rd.tool_choice, gemini_declarations, request_id)
                if tool_config_converted:
                    generation_config_updates.setdefault("toolConfig", {}).update(tool_config_converted) 
        
    if generation_config_updates: 
        payload["generationConfig"] = generation_config_updates
        
    logger.debug(f"{log_prefix}: Google (non-Gemini REST) Request Payload: {str(payload)[:1000]}")
    return payload, is_native_gemini_thinking_active


def _convert_openai_tools_to_gemini_declarations(openai_tools: List[Dict[str, Any]], request_id: str) -> List[Dict[str, Any]]:
    log_prefix = f"RID-{request_id}"; declarations = []
    if not openai_tools: return []
    for tool_def in openai_tools:
        if tool_def.get("type") == "function" and "function" in tool_def:
            func_spec = tool_def["function"]
            declaration = {k: v for k, v in {"name": func_spec.get("name"),"description": func_spec.get("description"),"parameters": func_spec.get("parameters")}.items() if v is not None}
            if declaration.get("name") and declaration.get("description") is not None: declarations.append(declaration)
            else: logger.warning(f"{log_prefix}: Google tool conversion: Func def missing name/desc: {func_spec}")
    return declarations

def _convert_openai_tool_choice_to_gemini_tool_config(openai_tool_choice: Union[str, Dict[str, Any]], gemini_declarations: List[Dict[str, Any]], request_id: str) -> Optional[Dict[str, Any]]:
    log_prefix = f"RID-{request_id}"; mode = "AUTO"; allowed_function_names: Optional[List[str]] = None
    if not openai_tool_choice: return None
    if isinstance(openai_tool_choice, str):
        choice_lower = openai_tool_choice.lower()
        if choice_lower == "none": mode = "NONE"
        elif choice_lower == "auto": mode = "AUTO"
        elif choice_lower == "required": mode = "ANY" if gemini_declarations else "AUTO"
        else: logger.warning(f"{log_prefix}: Google tool_choice: Unsupported str value '{openai_tool_choice}', defaulting to AUTO."); mode = "AUTO"
    elif isinstance(openai_tool_choice, dict) and openai_tool_choice.get("type") == "function":
        func_name = openai_tool_choice.get("function", {}).get("name")
        if func_name:
            if any(decl.get("name") == func_name for decl in gemini_declarations):
                mode = "ANY"; allowed_function_names = [func_name]
            else: logger.warning(f"{log_prefix}: Google tool_choice: Specified func '{func_name}' not in declared tools. Defaulting to AUTO."); mode = "AUTO"
        else: mode = "ANY" if gemini_declarations else "AUTO"
    else: logger.warning(f"{log_prefix}: Google tool_choice: Invalid format {openai_tool_choice}. Defaulting to AUTO."); mode = "AUTO"
    
    function_calling_config: Dict[str, Any] = {"mode": mode}
    if mode == "ANY" and allowed_function_names:
        function_calling_config["allowed_function_names"] = allowed_function_names
    
    if mode == "NONE" or (gemini_declarations and (mode == "ANY" or mode == "AUTO")):
        return {"function_calling_config": function_calling_config}
    return None