import orjson
import logging
import copy
from typing import List, Dict, Any, Optional, Union, Tuple
from urllib.parse import urljoin, urlparse

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
MARKDOWN_FORMAT_SYSTEM_PROMPT = """你必须严格遵循以下输出规范，确保前端（Android Compose: EnhancedMarkdownText + CodePreview）可无损解析与预览。

# 0. 总则
- 使用标准 Markdown 撰写正文。除非用户明确要求“预览 Markdown”，否则不要把普通正文包裹在 ```markdown 或 ```md 代码块中。
- 仅将“纯代码/数据/可视化定义”置于三反引号代码块中，并在代码块开头标注语言。
- 任何代码块/数学块/表格必须自成原子（完整开始与结束），禁止跨段/跨流片断裂。

# 1. 数学（KaTeX 对齐）
- 行内：$...$；块级：$$...$$；不要混用；美元符成对且不嵌套。
- 常用写法：
  - 指数：$x^{2}$；分数：$\\frac{a}{b}$；根号：$\\sqrt{x}$；下标：$a_{i}$；希腊字母：$\\alpha$
- 行内/块级示例：
  - 正确：勾股定理：$a^{2} + b^{2} = c^{2}$
  - 正确（块）：$$e^{i\\pi}+1=0$$
- 数学块前后保留换行；行内前后保留一个空格以便排版。

# 2. 代码块（前端解析/预览关键）
- 语法：三反引号独占一行；开头必须带语言；结尾独占一行；示例：
```python
print("Hello")
```
- 允许的语言标识（请优先使用这些，确保前端预览/解析）：html, svg, css, javascript, js, json, xml, mermaid, python, java, kotlin, cpp, c, bash, sql, yaml
- 每个代码块必须在同一条消息内完整闭合；严禁输出未闭合的 `、``` 或 $$。
- 如需多段代码，使用多个完整的代码块；不要在代码块中夹杂解释文字。
- 代码块内容请不要包含原生字符序列 ```（如不可避免，请改写示例或用占位符说明）。

# 3. 预览型代码块特殊规则（映射 CodePreview）
- HTML（语言：html）：
  - 可输出完整页面或片段；不要在代码块外再追加 HTML 解释性文字。
  - 如为片段，前端会自动包裹模板，无需自行添加 <!DOCTYPE>。
- SVG（语言：svg）：
  - 根元素必须是 <svg ...>；不要添加 XML 声明头；确保闭合。
- CSS（语言：css）：
  - 仅输出纯 CSS 规则（不加 <style> 标签）；选择器和属性合法。
- JavaScript（语言：javascript 或 js）：
  - 用 console.log 展示结果；避免 alert/prompt、网络请求、无限循环和敏感 API。
- Mermaid（语言：mermaid）：
  - 第一行使用合法定义（如 graph TD、flowchart、sequenceDiagram 等），不要包裹额外 Markdown。
- JSON（语言：json）：
  - 严格 JSON 语法（双引号、无多余逗号、括号匹配）。仅在用户需要结构化数据时使用。
- Markdown 预览（语言：markdown 或 md）：
  - 只有当用户明确要求“以 Markdown 预览”为代码块时才使用；否则正文应为普通 Markdown 文本而非代码块。

# 4. 表格
- 使用标准 Markdown 表格：
| 列1 | 列2 |
| --- | --- |
| 值1 | 值2 |
- 必须包含：表头行 + 对齐分隔行（--- 或 :---: 等）+ 至少一行数据；列数对齐。

# 5. 标准 Markdown 细则
- 标题：# 后空格；列表项符号后空格；链接：[text](url)；引用：> 后空格。
- 段落之间使用空行分隔；不要输出大段无意义空行。

# 6. 流式/分片输出纪律
- 代码块、数学块、表格必须原子输出：开始标记 -> 完整内容 -> 结束标记；禁止中途插入解释文本。
- 如果开始了代码围栏（```lang），须在同一轮完成闭合（```）；如需撤回，立刻补出闭合标记，不留悬空。
- 行内代码使用单反引号配对，禁止在行尾遗留未闭合的 `。

请严格遵守以上规则，保证输出无需后处理即可被前端稳定解析与预览。"""

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
    # 根据用户规则构建目标 URL：
    # 1) 以 # 结尾：上层 openai.py 会直接用用户地址（去掉 #），此处返回一个合理的默认值占位
    # 2) 地址包含路径且不以 / 结尾：视为完整端点，原样使用
    # 3) 地址以 / 结尾：不要 v1，改为补 /chat/completions
    # 4) 地址既无路径也无 #：自动补 /v1/chat/completions
    api_addr = (request_data.api_address or "").strip()
    default_path = OPENAI_COMPATIBLE_PATH.lstrip('/')  # e.g. v1/chat/completions
    no_v1_path = default_path[len('v1/'):] if default_path.startswith('v1/') else 'chat/completions'

    if not api_addr:
        base_url = DEFAULT_OPENAI_API_BASE_URL.strip().rstrip('/')
        target_url = urljoin(f"{base_url}/", default_path)
    else:
        if api_addr.endswith('#'):
            # 占位：最终 URL 将在 openai.py 中用去掉 # 的地址覆盖
            base_url = DEFAULT_OPENAI_API_BASE_URL.strip().rstrip('/')
            target_url = urljoin(f"{base_url}/", default_path)
        else:
            # 使用 urlparse 判断是否包含路径（为避免无 schema 的误判，仅用于判定）
            parse_for_det = api_addr if '://' in api_addr else f"http://{api_addr}"
            parsed = urlparse(parse_for_det)
            path = parsed.path or ""
            if path == "":
                # 无路径 -> 自动补 /v1/chat/completions
                target_url = f"{api_addr.rstrip('/')}/{default_path}"
            elif path.endswith('/'):
                # 以 / 结尾 -> 不要 v1，补 /chat/completions
                target_url = f"{api_addr.rstrip('/')}/{no_v1_path}"
            else:
                # 已包含路径且不以 / 结尾 -> 视为完整端点
                target_url = api_addr
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