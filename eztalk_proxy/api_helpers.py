import orjson
from typing import List, Dict, Any, Optional, Union, Tuple
from urllib.parse import urljoin

from eztalk_proxy.models import ChatRequestModel, SimpleTextApiMessagePy
from eztalk_proxy.config import DEFAULT_OPENAI_API_BASE_URL, OPENAI_COMPATIBLE_PATH
from eztalk_proxy.katex_prompt import KATEX_FORMATTING_INSTRUCTION
from eztalk_proxy.deepseek_katex_prompt import DEEPSEEK_KATEX_FORMATTING_INSTRUCTION
from eztalk_proxy.qwen_katex_prompt import QWEN_KATEX_FORMATTING_INSTRUCTION
from eztalk_proxy.utils import is_gemini_2_5_model

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

def _convert_simple_text_messages_to_google_contents(
    messages: List[SimpleTextApiMessagePy]
) -> List[Dict[str, Any]]:
    google_contents: List[Dict[str, Any]] = []
    for msg in messages:
        if not isinstance(msg, SimpleTextApiMessagePy): continue
        
        role = "model" if msg.role == "assistant" else msg.role
        if role not in ["user", "model", "function"]: role = "user"
        
        parts = []
        if msg.content and msg.content.strip():
            parts.append({"text": msg.content})
        
        if msg.tool_calls and role == "model":
            for tc in msg.tool_calls:
                if tc.type == "function" and tc.function.name and tc.function.arguments is not None:
                    try:
                        args_obj = orjson.loads(tc.function.arguments)
                    except orjson.JSONDecodeError:
                        args_obj = {"error": "invalid json from assistant tool_call", "raw": tc.function.arguments}
                    parts.append({"functionCall": {"name": tc.function.name, "args": args_obj}})
        
        if msg.role == "tool" and msg.name and msg.tool_call_id and msg.content is not None:
            try:
                response_obj = orjson.loads(msg.content)
            except orjson.JSONDecodeError:
                response_obj = {"raw_response": msg.content}
            role = "function"
            parts.append({"functionResponse": {"name": msg.name, "response": response_obj}})

        if parts:
            google_contents.append({"role": role, "parts": parts})
    return google_contents

def prepare_google_request_payload_structure(
    rd: ChatRequestModel,
    api_messages: List[SimpleTextApiMessagePy],
    request_id: str
) -> Tuple[Dict[str, Any], bool]:
    system_instruction_parts = []
    user_facing_messages = []
    
    for m_dict in api_messages:
        m_obj = SimpleTextApiMessagePy(**m_dict) if isinstance(m_dict, dict) else m_dict
        if not isinstance(m_obj, SimpleTextApiMessagePy): continue
        if m_obj.role == "system" and m_obj.content and m_obj.content.strip():
            system_instruction_parts.append(m_obj.content.strip())
        else:
            user_facing_messages.append(m_obj)

    if not any(KATEX_FORMATTING_INSTRUCTION in part for part in system_instruction_parts):
        system_instruction_parts.append(KATEX_FORMATTING_INSTRUCTION)
    
    final_system_instruction = "\n\n".join(system_instruction_parts).strip()
    
    payload: Dict[str, Any] = {
        "contents": _convert_simple_text_messages_to_google_contents(user_facing_messages)
    }
    if final_system_instruction:
        payload["systemInstruction"] = {"parts": [{"text": final_system_instruction}]}

    gen_conf_updates: Dict[str, Any] = {}
    is_native_gemini_thinking_active = False
    
    gen_conf_input = rd.generation_config
    if gen_conf_input:
        gen_conf_updates = {
            "temperature": gen_conf_input.temperature,
            "topP": gen_conf_input.top_p,
            "maxOutputTokens": gen_conf_input.max_output_tokens,
        }
        if gen_conf_input.thinking_config and gen_conf_input.thinking_config.include_thoughts is not None:
            thinking_payload = {"includeThoughts": gen_conf_input.thinking_config.include_thoughts}
            if gen_conf_input.thinking_config.thinking_budget is not None and is_gemini_2_5_model(rd.model):
                thinking_payload["thinkingBudget"] = gen_conf_input.thinking_config.thinking_budget
            gen_conf_updates["thinkingConfig"] = thinking_payload
            is_native_gemini_thinking_active = bool(thinking_payload.get("includeThoughts"))
    else:
        gen_conf_updates = {
            "temperature": rd.temperature, "topP": rd.top_p, "maxOutputTokens": rd.max_tokens
        }
        
    if rd.tools:
        gemini_declarations = _convert_openai_tools_to_gemini_declarations(rd.tools)
        if gemini_declarations:
            payload["tools"] = [{"functionDeclarations": gemini_declarations}]
            if rd.tool_choice:
                tool_config = _convert_openai_tool_choice_to_gemini_tool_config(rd.tool_choice, gemini_declarations)
                if tool_config:
                    gen_conf_updates.setdefault("toolConfig", {}).update(tool_config)
        
    payload["generationConfig"] = {k: v for k, v in gen_conf_updates.items() if v is not None}
    if not payload["generationConfig"]:
        del payload["generationConfig"]
        
    return payload, is_native_gemini_thinking_active

def _convert_openai_tools_to_gemini_declarations(openai_tools: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    declarations = []
    for tool_def in openai_tools:
        if tool_def.get("type") == "function" and "function" in tool_def:
            func_spec = tool_def["function"]
            if func_spec.get("name") and func_spec.get("description") is not None:
                declarations.append({
                    "name": func_spec.get("name"),
                    "description": func_spec.get("description"),
                    "parameters": func_spec.get("parameters")
                })
    return [d for d in declarations if d.get("parameters") is not None]

def _convert_openai_tool_choice_to_gemini_tool_config(
    openai_tool_choice: Union[str, Dict[str, Any]],
    gemini_declarations: List[Dict[str, Any]]
) -> Optional[Dict[str, Any]]:
    mode = "AUTO"
    allowed_function_names: Optional[List[str]] = None

    if isinstance(openai_tool_choice, str):
        choice_lower = openai_tool_choice.lower()
        if choice_lower == "none": mode = "NONE"
        elif choice_lower == "required": mode = "ANY" if gemini_declarations else "AUTO"
    elif isinstance(openai_tool_choice, dict) and openai_tool_choice.get("type") == "function":
        func_name = openai_tool_choice.get("function", {}).get("name")
        if func_name and any(d.get("name") == func_name for d in gemini_declarations):
            mode = "ANY"
            allowed_function_names = [func_name]
        else:
            mode = "ANY" if gemini_declarations else "AUTO"
    
    config: Dict[str, Any] = {"mode": mode}
    if mode == "ANY" and allowed_function_names:
        config["allowed_function_names"] = allowed_function_names
    
    if mode == "NONE" or (gemini_declarations and mode in ["ANY", "AUTO"]):
        return {"function_calling_config": config}
    return None