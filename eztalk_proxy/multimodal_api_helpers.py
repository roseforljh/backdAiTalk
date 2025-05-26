# multimodal_api_helpers.py
import logging
from typing import List, Dict, Any, Tuple, Optional

# 假设这些模型与 multimodal_api_helpers.py 在同一个 eztalk_proxy 包内
from .models import ChatRequest # 你的Pydantic模型
from .config import DEFAULT_OPENAI_API_BASE_URL, OPENAI_COMPATIBLE_PATH # OpenAI 相关配置

# ========================== IMPORT ERROR NOTICE ==========================
# 以下导入语句依赖于 eztalk_proxy/api_helpers.py 文件。
# 如果出现 "cannot import name..." 错误，意味着这些名称
# (例如 get_pure_base64_from_data_uri) 没有在 api_helpers.py 中正确定义或导出。
# 你需要检查和修改 api_helpers.py 来解决这些导入问题。
from .api_helpers import get_pure_base64_from_data_uri, get_mime_type_from_data_uri, KATEX_FORMATTING_INSTRUCTION
from .api_helpers import _convert_openai_tools_to_gemini_declarations, _convert_openai_tool_choice_to_gemini_tool_config
# ========================================================================

from .utils import is_gemini_2_5_model # 假设 utils.py 在同级或可访问路径


logger = logging.getLogger("EzTalkProxy.APIHelpers.Multimodal")

def _extract_text_from_dumped_content(content: Any) -> str:
    """
    Helper to extract plain text from various content structures after model_dump().
    Content can be a dict (TextContentIn or MultipartContentIn dump), a list of parts, or a string.
    """
    texts = []
    if isinstance(content, dict):
        content_type = content.get("type")
        if content_type == "text_content" and content.get("text"):
            texts.append(content["text"])
        elif content_type == "multipart_content" and content.get("parts"):
            for part in content["parts"]:
                if part.get("type") == "text" and part.get("text"):
                    texts.append(part["text"])
    elif isinstance(content, list): # Fallback if content is already a list of parts
        for part in content:
            if isinstance(part, dict) and part.get("type") == "text" and part.get("text"):
                texts.append(part["text"])
    elif isinstance(content, str):
        texts.append(content)
    return "\n".join(texts).strip()

def prepare_openai_multimodal_request(rd: ChatRequest, messages_from_proxy: List[Dict[str, Any]], request_id: str) -> Tuple[str, Dict[str, str], Dict[str, Any]]:
    logger.info(f"RID-{request_id}: Preparing OpenAI MULTIMODAL request for model: {rd.model}")
    base = rd.api_address.strip() if rd.api_address else DEFAULT_OPENAI_API_BASE_URL
    url = f"{base.rstrip('/')}{OPENAI_COMPATIBLE_PATH}"
    headers = {"Content-Type": "application/json", "Authorization": f"Bearer {rd.api_key}"}

    processed_messages_for_api = []
    system_message_content_parts_text = []

    for msg_dict in messages_from_proxy:
        role = msg_dict["role"]
        content_from_proxy = msg_dict.get("content") # This is after model_dump()

        if role == "system":
            system_text = _extract_text_from_dumped_content(content_from_proxy)
            if system_text:
                system_message_content_parts_text.append(system_text)
            continue # System messages are handled together later

        api_content_for_openai: Any = None
        if role == "user":
            if isinstance(content_from_proxy, dict):
                content_type = content_from_proxy.get("type")
                if content_type == "text_content" and content_from_proxy.get("text"):
                    # OpenAI expects user message content to be a list of parts
                    api_content_for_openai = [{"type": "text", "text": content_from_proxy["text"].strip()}]
                elif content_type == "multipart_content" and content_from_proxy.get("parts"):
                    # Assuming parts are already in OpenAI format from ApiContentPart.model_dump()
                    api_content_for_openai = content_from_proxy["parts"]
            elif isinstance(content_from_proxy, str) and content_from_proxy.strip(): # Plain string user message
                api_content_for_openai = [{"type": "text", "text": content_from_proxy.strip()}]
            # If content_from_proxy is already a list (e.g. direct pass-through of OpenAI format list), it's handled by the next condition
            elif isinstance(content_from_proxy, list): # Should ideally come from multipart_content.parts
                 api_content_for_openai = content_from_proxy


        elif role == "assistant":
            # Assistant responses are typically text strings for OpenAI
            # If content is structured (e.g. from a TextContentIn dump), extract text
            assistant_text = _extract_text_from_dumped_content(content_from_proxy)
            if assistant_text:
                api_content_for_openai = assistant_text
            # If assistant message had tool_calls, content might be None, handled by `if api_content_for_openai is not None`

        api_message_for_openai: Dict[str, Any] = {"role": role}
        if api_content_for_openai is not None:
            api_message_for_openai["content"] = api_content_for_openai
        
        # Handle tool calls (assuming structure from ApiMessage.model_dump() is correct for OpenAI)
        if "tool_calls" in msg_dict:
            api_message_for_openai["tool_calls"] = msg_dict["tool_calls"]
        
        if role == "tool": # 'tool' role messages have content as string (tool result)
            if "tool_call_id" in msg_dict: api_message_for_openai["tool_call_id"] = msg_dict["tool_call_id"]
            if "name" in msg_dict: api_message_for_openai["name"] = msg_dict["name"] # Function name
            # Content for role=tool should be a string (result of the tool call)
            # The _extract_text_from_dumped_content helper should handle this if content_from_proxy was string
            tool_content_str = _extract_text_from_dumped_content(content_from_proxy)
            if tool_content_str and "content" not in api_message_for_openai: # Ensure not to overwrite
                api_message_for_openai["content"] = tool_content_str


        # Add message if it has content or tool_calls, or if it's a tool role with tool_call_id
        if api_message_for_openai.get("content") is not None or \
           api_message_for_openai.get("tool_calls") is not None or \
           (role == "tool" and "tool_call_id" in api_message_for_openai):
            processed_messages_for_api.append(api_message_for_openai)

    # Consolidate and add system message
    final_system_content_str = "\n\n".join(filter(None,system_message_content_parts_text)).strip()
    # Ensure KATEX_FORMATTING_INSTRUCTION is not None before using it in string operations
    katex_instruction_str = KATEX_FORMATTING_INSTRUCTION if isinstance(KATEX_FORMATTING_INSTRUCTION, str) else ""

    if katex_instruction_str and katex_instruction_str not in final_system_content_str:
        final_system_content_str = (final_system_content_str + "\n\n" + katex_instruction_str).strip()
    
    if final_system_content_str:
        processed_messages_for_api.insert(0, {"role": "system", "content": final_system_content_str})
    elif not any(m["role"] == "system" for m in processed_messages_for_api) and katex_instruction_str:
        # Add default KaTeX instruction if no other system message exists
        processed_messages_for_api.insert(0, {"role": "system", "content": katex_instruction_str})


    payload: Dict[str, Any] = {"model": rd.model, "messages": processed_messages_for_api, "stream": True}
    if rd.temperature is not None: payload["temperature"] = rd.temperature
    if rd.top_p is not None: payload["top_p"] = rd.top_p
    
    has_any_image = any(
        isinstance(m.get("content"), list) and any(p.get("type") == "image_url" for p in m["content"])
        for m in processed_messages_for_api if m["role"] == "user"
    )
    if has_any_image:
        payload["max_tokens"] = rd.max_tokens or 4096 # Default for vision
    elif rd.max_tokens is not None:
        payload["max_tokens"] = rd.max_tokens

    if rd.tools: payload["tools"] = rd.tools
    if rd.tool_choice: payload["tool_choice"] = rd.tool_choice
    if rd.custom_model_parameters: payload.update(rd.custom_model_parameters)
    if rd.custom_extra_body: 
        # Ensure extra_body is merged correctly if it already exists from custom_model_parameters
        current_extra_body = payload.get("extra_body", {})
        current_extra_body.update(rd.custom_extra_body)
        payload["extra_body"] = current_extra_body

    logger.debug(f"RID-{request_id}: OpenAI MULTIMODAL Request Payload (first 2 messages, contents truncated): {str([{'role':m.get('role'), 'content':str(m.get('content'))[:100]+'...'} for m in payload.get('messages', [])[:2]])}")
    return url, headers, payload


def prepare_google_multimodal_request(rd: ChatRequest, messages_from_proxy: List[Dict[str, Any]], request_id: str) -> Tuple[Dict[str, Any], bool]:
    logger.info(f"RID-{request_id}: Preparing Google MULTIMODAL request for model: {rd.model}")
    
    generation_config_updates: Dict[str, Any] = {}
    is_native_gemini_thinking_active = False
    system_instruction_parts_text = []
    
    user_facing_gemini_contents: List[Dict[str, Any]] = []

    for m_dict in messages_from_proxy:
        role = m_dict["role"]
        content_from_proxy = m_dict.get("content") # This is after model_dump()

        if role == "system":
            system_text = _extract_text_from_dumped_content(content_from_proxy)
            if system_text:
                system_instruction_parts_text.append(system_text)
            continue

        # Gemini roles: "user" for user, "model" for assistant, "function" for tool response
        api_role_for_gemini = "user" if role == "user" else ("model" if role == "assistant" else "function" if role == "tool" else role)
        gemini_parts_for_current_message = []

        if isinstance(content_from_proxy, dict): # Content is TextContentIn or MultipartContentIn (dumped)
            content_type = content_from_proxy.get("type")
            if content_type == "text_content" and content_from_proxy.get("text"):
                gemini_parts_for_current_message.append({"text": content_from_proxy["text"].strip()})
            elif content_type == "multipart_content" and content_from_proxy.get("parts"):
                for part_dict in content_from_proxy["parts"]: # part_dict is from ApiContentPart.model_dump()
                    if part_dict.get("type") == "text" and part_dict.get("text"):
                        gemini_parts_for_current_message.append({"text": part_dict["text"]})
                    elif part_dict.get("type") == "image_url" and part_dict.get("image_url", {}).get("url"):
                        data_uri = part_dict["image_url"]["url"]
                        # These functions need to be working from api_helpers.py
                        mime_type = get_mime_type_from_data_uri(data_uri) if callable(get_mime_type_from_data_uri) else "image/jpeg"
                        base64_data = get_pure_base64_from_data_uri(data_uri) if callable(get_pure_base64_from_data_uri) else None
                        if base64_data:
                            gemini_parts_for_current_message.append({
                                "inline_data": {"mime_type": mime_type, "data": base64_data}
                            })
                        else:
                            logger.warning(f"RID-{request_id}: Could not process image data URI for Gemini MULTIMODAL: {data_uri[:50]}...")
        elif isinstance(content_from_proxy, str) and content_from_proxy.strip(): # Plain string content
            gemini_parts_for_current_message.append({"text": content_from_proxy.strip()})
        # Note: if content_from_proxy was an explicit list of parts (old format, less likely now), this logic might miss it
        # The primary paths are dict (from dumped TextContentIn/MultipartContentIn) or str.

        # Handle tool calls for Gemini
        if role == "assistant" and "tool_calls" in m_dict: # Assistant requests a tool call
            api_role_for_gemini = "model" # Still "model" role when making a functionCall
            openai_tool_calls = m_dict["tool_calls"]
            for tc_dict in openai_tool_calls:
                if tc_dict.get("type") == "function":
                    func_data = tc_dict.get("function", {})
                    func_name = func_data.get("name")
                    func_args_str = func_data.get("arguments")
                    if func_name and func_args_str is not None:
                        try:
                            args_obj = orjson.loads(func_args_str)
                        except orjson.JSONDecodeError:
                            args_obj = {"error": "Invalid JSON arguments", "raw_args": func_args_str}
                        gemini_parts_for_current_message.append({"functionCall": {"name": func_name, "args": args_obj}})
        elif role == "tool": # This is a tool's response back to the model
            api_role_for_gemini = "function" # Gemini specific role for tool responses
            tool_name = m_dict.get("name") # This should be the function name from the original tool_call_id
            tool_content_str = _extract_text_from_dumped_content(content_from_proxy) # Tool content is string (JSON result)
            
            if tool_name and tool_content_str:
                try:
                    response_obj = orjson.loads(tool_content_str)
                except orjson.JSONDecodeError:
                    response_obj = {"raw_response": tool_content_str} # Send as is if not valid JSON
                gemini_parts_for_current_message.append({"functionResponse": {"name": tool_name, "response": response_obj}})


        if gemini_parts_for_current_message:
            user_facing_gemini_contents.append({"role": api_role_for_gemini, "parts": gemini_parts_for_current_message})

    # System instruction processing
    final_system_instruction_content_str = "\n\n".join(filter(None, system_instruction_parts_text)).strip()
    # Ensure KATEX_FORMATTING_INSTRUCTION is not None before using it
    katex_instruction_str = KATEX_FORMATTING_INSTRUCTION if isinstance(KATEX_FORMATTING_INSTRUCTION, str) else ""

    if not final_system_instruction_content_str and ("gemini" in rd.model.lower()) and katex_instruction_str:
        final_system_instruction_content_str = katex_instruction_str
    elif final_system_instruction_content_str and katex_instruction_str and katex_instruction_str not in final_system_instruction_content_str:
        final_system_instruction_content_str = (final_system_instruction_content_str + "\n\n" + katex_instruction_str).strip()
    
    payload: Dict[str, Any] = {"contents": user_facing_gemini_contents}
    if final_system_instruction_content_str:
        payload["systemInstruction"] = {"parts": [{"text": final_system_instruction_content_str}]}

    if is_gemini_2_5_model(rd.model):
        is_native_gemini_thinking_active = True
        generation_config_updates.setdefault("thinkingConfig", {}).update({"includeThoughts": True})

    if rd.tools and callable(_convert_openai_tools_to_gemini_declarations) and callable(_convert_openai_tool_choice_to_gemini_tool_config):
        gemini_declarations = _convert_openai_tools_to_gemini_declarations(rd.tools, request_id)
        if gemini_declarations:
            payload["tools"] = [{"functionDeclarations": gemini_declarations}]
            if rd.tool_choice:
                tool_config = _convert_openai_tool_choice_to_gemini_tool_config(rd.tool_choice, gemini_declarations, request_id)
                if tool_config:
                    generation_config_updates.setdefault("toolConfig", {}).update(tool_config)
    
    if rd.temperature is not None: generation_config_updates["temperature"] = rd.temperature
    if rd.top_p is not None: generation_config_updates["topP"] = rd.top_p
    if rd.max_tokens is not None: generation_config_updates["maxOutputTokens"] = rd.max_tokens

    if generation_config_updates:
        payload["generationConfig"] = generation_config_updates
    
    logger.debug(f"RID-{request_id}: Google MULTIMODAL Request Payload Contents (first 2, parts truncated): {str([{'role':c.get('role'), 'parts': [str(p)[:70]+'...' for p in c.get('parts',[])]} for c in payload.get('contents', [])[:2]])}")
    return payload, is_native_gemini_thinking_active