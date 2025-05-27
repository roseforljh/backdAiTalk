# eztalk_proxy/api_helpers.py
import logging
import orjson
from typing import List, Dict, Any, Optional, Union, Tuple

from .models import ChatRequestModel, SimpleTextApiMessagePy
from .config import DEFAULT_OPENAI_API_BASE_URL, OPENAI_COMPATIBLE_PATH
from .katex_prompt import KATEX_FORMATTING_INSTRUCTION

logger = logging.getLogger("EzTalkProxy.APIHelpers")

def prepare_openai_request(
    request_data: ChatRequestModel,
    processed_messages: List[Dict[str, Any]],
    request_id: str
) -> Tuple[str, Dict[str, str], Dict[str, Any]]:
    log_prefix = f"RID-{request_id}"

    base_url = request_data.api_address.strip() if request_data.api_address else DEFAULT_OPENAI_API_BASE_URL
    target_url = f"{base_url.rstrip('/')}{OPENAI_COMPATIBLE_PATH}"

    headers = {
        "Authorization": f"Bearer {request_data.api_key}",
        "Content-Type": "application/json",
        "Accept": "text/event-stream"
    }

    final_openai_payload_msgs = []
    system_message_found_and_updated = False
    # ... (处理 system message 和 KaTeX 指令的逻辑保持不变) ...
    for m_dict in processed_messages:
        if m_dict.get("role") == "system":
            original_content = m_dict.get("content", "")
            final_system_content = original_content
            if KATEX_FORMATTING_INSTRUCTION not in final_system_content:
                final_system_content = (final_system_content + "\n\n" + KATEX_FORMATTING_INSTRUCTION).strip()
            final_openai_payload_msgs.append({"role": "system", "content": final_system_content})
            system_message_found_and_updated = True
        else:
            final_openai_payload_msgs.append(m_dict)
    if not system_message_found_and_updated:
        final_openai_payload_msgs.insert(0, {"role": "system", "content": KATEX_FORMATTING_INSTRUCTION})
        logger.info(f"{log_prefix}: OpenAI Req: No system message in input, prepended KaTeX instruction.")


    payload: Dict[str, Any] = {
        "model": request_data.model,
        "messages": final_openai_payload_msgs,
        "stream": True,
    }

    # --- 修改：直接从 request_data 的顶层属性获取生成参数 ---
    # 确保这些属性名与 ChatRequestModel 中定义的（考虑了 alias 后的 JSON 键名）一致
    if request_data.temperature is not None: # 假设 ChatRequestModel 中有 temperature 字段
        payload["temperature"] = request_data.temperature
    if request_data.top_p is not None: # 假设 ChatRequestModel 中有 top_p 字段 (对应JSON "topP")
        payload["top_p"] = request_data.top_p
    if request_data.max_tokens is not None: # 假设 ChatRequestModel 中有 max_tokens 字段 (对应JSON "maxTokens")
        payload["max_tokens"] = request_data.max_tokens
    # --- 修改结束 ---

    if request_data.tools: payload["tools"] = request_data.tools
    if request_data.tool_choice: payload["tool_choice"] = request_data.tool_choice
    
    if request_data.custom_model_parameters:
        logger.info(f"{log_prefix}: OpenAI Req: Applying custom_model_parameters: {list(request_data.custom_model_parameters.keys())}")
        for key, value in request_data.custom_model_parameters.items():
            if key not in payload: 
                payload[key] = value
            else:
                logger.warning(f"{log_prefix}: OpenAI Req: Custom parameter '{key}' conflicts, NOT applied.")
    
    if request_data.custom_extra_body:
        logger.info(f"{log_prefix}: OpenAI Req: Applying custom_extra_body for non-Gemini (if proxy supports): {list(request_data.custom_extra_body.keys())}")
        payload.setdefault("extra_body", {}).update(request_data.custom_extra_body)
        
    logger.debug(f"{log_prefix}: OpenAI Request - URL: {target_url}, Model: {request_data.model}")
    logger.debug(f"{log_prefix}: OpenAI Request Payload (first 500 of messages): {str(payload.get('messages',[]))[:500]}")
    return target_url, headers, payload

# --- Google 相关辅助函数 ( _convert_simple_text_messages_to_google_contents, prepare_google_request_payload_structure 等) ---
# 这些函数也需要确保它们从 request_data 正确获取生成参数（如果它们需要的话）
# 为简洁，我只修改了 prepare_openai_request，您需要类似地检查和修改其他函数。
# ... (其余函数定义保持不变或根据需要调整参数获取逻辑) ...
def _convert_simple_text_messages_to_google_contents(
    messages: List[SimpleTextApiMessagePy], request_id: str
) -> List[Dict[str, Any]]:
    # (保持与上一版本一致)
    log_prefix = f"RID-{request_id}"; google_contents: List[Dict[str, Any]] = []
    for msg in messages:
        if not isinstance(msg, SimpleTextApiMessagePy): continue
        role = "model" if msg.role == "assistant" else msg.role
        if role not in ["user", "model", "function"]: role = "user"
        parts = []
        if msg.content and msg.content.strip(): parts.append({"text": msg.content})
        if msg.tool_calls and role == "model":
            for tc in msg.tool_calls:
                if tc.type == "function" and tc.function.name and tc.function.arguments is not None:
                    try: args_obj = orjson.loads(tc.function.arguments)
                    except: args_obj = {"error": "invalid json", "raw": tc.function.arguments}
                    parts.append({"functionCall": {"name": tc.function.name, "args": args_obj}})
        if msg.role == "tool" and msg.name and msg.tool_call_id and msg.content is not None:
            try: response_obj = orjson.loads(msg.content)
            except: response_obj = {"raw_response": msg.content}
            role = "function"; parts.append({"functionResponse": {"name": msg.name, "response": response_obj}})
        if parts: google_contents.append({"role": role, "parts": parts})
    return google_contents

def prepare_google_request_payload_structure(
    rd: ChatRequestModel, api_messages: List[SimpleTextApiMessagePy], request_id: str
) -> Tuple[Dict[str, Any], bool]: 
    log_prefix = f"RID-{request_id}"; logger.info(f"{log_prefix}: Preparing Google request payload (TEXT-ONLY/NON-GEMINI path) for model {rd.model}")
    generation_config_updates: Dict[str, Any] = {}; is_native_gemini_thinking_active = False 
    system_instruction_parts = []; user_facing_messages_simple: List[SimpleTextApiMessagePy] = []; has_client_system_message = False
    for m_obj in api_messages:
        if m_obj.role == "system" and m_obj.content and m_obj.content.strip(): 
            has_client_system_message = True; system_content_with_katex = f"{m_obj.content.strip()}\n\n{KATEX_FORMATTING_INSTRUCTION}"; system_instruction_parts.append(system_content_with_katex)
        else: user_facing_messages_simple.append(m_obj)
    if not has_client_system_message : 
        if not any(part.lower().strip() == KATEX_FORMATTING_INSTRUCTION.lower().strip() for part in system_instruction_parts): system_instruction_parts.append(KATEX_FORMATTING_INSTRUCTION)
    final_system_instruction_content = "\n\n".join(system_instruction_parts).strip() if system_instruction_parts else None
    google_api_contents = _convert_simple_text_messages_to_google_contents(user_facing_messages_simple, request_id)
    payload: Dict[str, Any] = {"contents": google_api_contents}
    if final_system_instruction_content: payload["systemInstruction"] = {"parts": [{"text": final_system_instruction_content}]}
    
    # 直接从顶层读取（已通过alias映射JSON的temperature, topP, maxTokens）
    if rd.temperature is not None: generation_config_updates["temperature"] = rd.temperature
    if rd.top_p is not None: generation_config_updates["topP"] = rd.top_p
    if rd.max_tokens is not None: generation_config_updates["maxOutputTokens"] = rd.max_tokens
        
    if generation_config_updates: payload["generationConfig"] = generation_config_updates
    logger.debug(f"{log_prefix}: Google (non-Gemini) Request Payload (first 500 of contents): {str(payload.get('contents',[]))[:500]}")
    return payload, is_native_gemini_thinking_active

def _convert_openai_tools_to_gemini_declarations(openai_tools: List[Dict[str, Any]], request_id: str) -> List[Dict[str, Any]]:
    # ... (保持不变)
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
    # ... (保持不变)
    log_prefix = f"RID-{request_id}"; mode = "AUTO"; allowed_function_names = []
    if not openai_tool_choice: return None
    if isinstance(openai_tool_choice, str):
        if openai_tool_choice == "none": mode = "NONE"
        elif openai_tool_choice == "auto": mode = "AUTO"
        elif openai_tool_choice == "required": mode = "ANY" if gemini_declarations else "AUTO"
        else: logger.warning(f"{log_prefix}: Google tool_choice: Unsupported str value '{openai_tool_choice}', defaulting to AUTO."); mode = "AUTO"
    elif isinstance(openai_tool_choice, dict) and openai_tool_choice.get("type") == "function":
        func_name = openai_tool_choice.get("function", {}).get("name")
        if func_name:
            if any(decl.get("name") == func_name for decl in gemini_declarations): mode = "ANY"; allowed_function_names = [func_name]
            else: logger.warning(f"{log_prefix}: Google tool_choice: Specified func '{func_name}' not in declared tools. Defaulting to AUTO."); mode = "AUTO"
        else: mode = "ANY" if gemini_declarations else "AUTO"
    else: logger.warning(f"{log_prefix}: Google tool_choice: Invalid format {openai_tool_choice}. Defaulting to AUTO."); mode = "AUTO"
    function_calling_config: Dict[str, Any] = {"mode": mode}
    if mode == "ANY" and allowed_function_names: function_calling_config["allowed_function_names"] = allowed_function_names
    if gemini_declarations or mode == "NONE": return {"function_calling_config": function_calling_config}
    elif mode == "AUTO" and not gemini_declarations: return None
    if gemini_declarations and mode == "ANY": return {"function_calling_config": function_calling_config}
    return None