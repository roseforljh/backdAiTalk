import orjson
import logging
import copy
from typing import List, Dict, Any, Optional, Union, Tuple
from urllib.parse import urljoin

from ..models.api_models import (
    ChatRequestModel,
    SimpleTextApiMessagePy,
    PartsApiMessagePy,
    PyTextContentPart,
    PyFileUriContentPart,
    PyInlineDataContentPart,
    PyInputAudioContentPart
)
from ..core.config import (
    DEFAULT_OPENAI_API_BASE_URL,
    OPENAI_COMPATIBLE_PATH,
    GOOGLE_API_BASE_URL,
    GOOGLE_API_KEY_ENV
)

from ..utils.helpers import is_gemini_2_5_model

logger = logging.getLogger("EzTalkProxy.Services.RequestBuilder")

# 格式化输出prompt - 用于所有大模型
MARKDOWN_FORMAT_SYSTEM_PROMPT = """你必须严格遵循以下格式规则来组织输出，确保前端能够正确解析和显示：

## 数学公式格式（严格要求）：
* **行内数学公式：** 必须使用单个美元符号包裹：`$E=mc^2$`
* **块级数学公式：** 必须使用双美元符号包裹：`$$E=mc^2$$`
* **LaTeX语法要求：**
  - 指数：使用 `$x^{2}$` 而不是 `x^2`
  - 分数：使用 `$\frac{a}{b}$` 而不是 `a/b`
  - 开方：使用 `$\sqrt{x}$` 而不是 `√x`
  - 下标：使用 `$a_{i}$` 而不是 `a_i`
  - 希腊字母：使用 `$\alpha$` 而不是 `α`
* **数学公式示例：**
  - 正确：`勾股定理：$a^{2} + b^{2} = c^{2}$`
  - 错误：`勾股定理：a^2 + b^2 = c^2`

## 代码格式（严格要求）：
* **行内代码：** 使用反引号包裹：`` `print()` ``
* **代码块：** 必须使用三个反引号并指定语言：
```python
print("Hello World")
```
* **支持的语言标识：** python, javascript, java, cpp, c, html, css, sql, bash, json, xml, yaml

## Markdown格式：
* **标题：** 使用 `#` 到 `######`，# 后必须有空格
* **段落：** 通过空行分隔段落
* **加粗：** 使用 `**重点内容**` 突出重要信息
* **列表：**
  - 无序列表：使用 `- 项目`（- 后必须有空格）
  - 有序列表：使用 `1. 项目`（数字后必须有空格）
* **链接：** 使用 `[显示文本](URL)` 格式
* **引用：** 使用 `> 引用内容`（> 后必须有空格）
* **表格：** 使用标准Markdown表格格式

## 特别注意：
* 数学公式必须完整包裹在美元符号内，不能有遗漏
* 代码块必须有明确的语言标识
* 所有格式标记后必须有适当的空格
* 保持内容结构清晰，便于前端解析"""

def add_system_prompt_if_needed(messages: List[Dict[str, Any]], request_id: str) -> List[Dict[str, Any]]:
    """
    为所有大模型添加格式化系统prompt
    """
    log_prefix = f"RID-{request_id}"
    
    # 检查是否已经存在系统消息
    has_system_message = any(msg.get("role") == "system" for msg in messages)
    
    if not has_system_message:
        # 添加格式化系统prompt到消息列表开头
        system_message = {
            "role": "system",
            "content": MARKDOWN_FORMAT_SYSTEM_PROMPT
        }
        messages.insert(0, system_message)
        logger.info(f"{log_prefix}: Added Markdown formatting system prompt for all models")
    else:
        logger.info(f"{log_prefix}: System message already exists, skipping prompt injection")
    
    return messages

def is_gemini_model_in_openai_format(model_name: str) -> bool:
    """检测是否为使用OpenAI兼容格式的Gemini模型"""
    if not model_name:
        return False
    return "gemini" in model_name.lower()

def prepare_openai_request(
    request_data: ChatRequestModel,
    processed_messages: List[Dict[str, Any]],
    request_id: str,
   system_prompt: Optional[str] = None
) -> Tuple[str, Dict[str, str], Dict[str, Any]]:
    base_url = (request_data.api_address or DEFAULT_OPENAI_API_BASE_URL).strip().rstrip('/')
    target_url = urljoin(f"{base_url}/", OPENAI_COMPATIBLE_PATH.lstrip('/'))

    headers = {
        "Authorization": f"Bearer {request_data.api_key}",
        "Content-Type": "application/json",
        "Accept": "text/event-stream"
    }

    # 添加格式化系统prompt
    final_messages = add_system_prompt_if_needed(copy.deepcopy(processed_messages), request_id)
    if system_prompt:
       final_messages.insert(0, {"role": "system", "content": system_prompt})
    model_name_lower = request_data.model.lower()

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
    
    # 为Gemini模型添加Google搜索工具支持
    if is_gemini_model_in_openai_format(request_data.model) and request_data.use_web_search:
        tools_list = list(payload.get("tools", []) or [])
        # 添加Google搜索工具（使用OpenAI兼容格式）
        google_search_tool = {
            "type": "function",
            "function": {
                "name": "google_search",
                "description": "Search the web using Google to find current information",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "The search query to execute"
                        }
                    },
                    "required": ["query"]
                }
            }
        }
        tools_list.append(google_search_tool)
        payload["tools"] = tools_list
        logger.info(f"RID-{request_id}: Added Google Search tool for Gemini model in OpenAI format")
    
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
                elif isinstance(actual_part, PyInputAudioContentPart):
                    # 为Gemini REST API格式处理音频内容
                    # 音频数据作为inlineData处理，mime type根据format推断
                    mime_type = f"audio/{actual_part.format}"
                    rest_parts.append({
                        "inlineData": {
                            "mimeType": mime_type,
                            "data": actual_part.data
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

def add_system_prompt_to_gemini_messages(messages: List[PartsApiMessagePy], request_id: str) -> List[PartsApiMessagePy]:
    """
    为Gemini添加格式化系统prompt
    """
    log_prefix = f"RID-{request_id}"
    
    # 检查是否已经存在系统消息
    has_system_message = any(msg.role == "system" for msg in messages)
    
    if not has_system_message:
        # 为Gemini创建系统消息 (使用parts格式)
        system_text_part = PyTextContentPart(type="text_content", text=MARKDOWN_FORMAT_SYSTEM_PROMPT)
        system_message = PartsApiMessagePy(
            role="system",
            message_type="parts_message",
            parts=[system_text_part]
        )
        messages.insert(0, system_message)
        logger.info(f"{log_prefix}: Added Markdown formatting system prompt for Gemini")
    else:
        logger.info(f"{log_prefix}: System message already exists for Gemini, skipping prompt injection")
    
    return messages

def prepare_gemini_rest_api_request(
    chat_input: ChatRequestModel,
    request_id: str,
   system_prompt: Optional[str] = None
) -> Tuple[str, Dict[str, str], Dict[str, Any]]:
    log_prefix = f"RID-{request_id}"
    logger.info(f"{log_prefix}: Preparing Gemini REST API request for model {chat_input.model}.")

    model_name = chat_input.model
    
    # Only use user-provided API key, no fallback to environment variable
    if not chat_input.api_key:
        raise ValueError("No user-provided API key for Gemini")
    
    # Initialize base_api_url to ensure it's always defined
    base_api_url = GOOGLE_API_BASE_URL.rstrip('/')
    
    # Use user-provided API address if available, otherwise use Google official
    if chat_input.api_address:
        # Check if the user provided a complete URL or just a base URL
        if "/v1beta/models/" in chat_input.api_address:
            # User provided a complete URL, use it as is but add streaming parameters
            base_url = chat_input.api_address.rstrip('/')
            base_api_url = base_url.split('/v1beta/models/')[0]  # Extract base URL for logging
            if ":generateContent" in base_url:
                # Replace generateContent with streamGenerateContent
                target_url = base_url.replace(":generateContent", ":streamGenerateContent")
            else:
                target_url = base_url
            target_url = f"{target_url}?key={chat_input.api_key}&alt=sse"
        else:
            # User provided just a base URL, construct the full path
            base_api_url = chat_input.api_address.rstrip('/')
            target_url = f"{base_api_url}/v1beta/models/{model_name}:streamGenerateContent?key={chat_input.api_key}&alt=sse"
    else:
        target_url = f"{base_api_url}/v1beta/models/{model_name}:streamGenerateContent?key={chat_input.api_key}&alt=sse"
    
    logger.info(f"{log_prefix}: Using user-provided API key for Gemini request to {base_api_url}")

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

    # 添加Gemini格式化系统prompt
    messages_to_convert_or_use = add_system_prompt_to_gemini_messages(messages_to_convert_or_use, request_id)

    if not messages_to_convert_or_use:
        logger.error(f"{log_prefix}: No processable messages found for Gemini REST request.")
        json_payload["contents"] = []
    else:
        json_payload["contents"] = convert_parts_messages_to_rest_api_contents(messages_to_convert_or_use, request_id)
    if system_prompt:
        json_payload["systemInstruction"] = {"parts": [{"text": system_prompt}]}



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

    gemini_tools_payload = []
    if chat_input.use_web_search:
        gemini_tools_payload.append({"googleSearch": {}})
        logger.info(f"{log_prefix}: Enabled Google Search tool for Gemini.")

    if chat_input.tools:
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
                if "name" in declaration and "description" in declaration:
                    converted_declarations.append(declaration)
        
        if converted_declarations:
            gemini_tools_payload.append({"functionDeclarations": converted_declarations})

    if gemini_tools_payload:
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