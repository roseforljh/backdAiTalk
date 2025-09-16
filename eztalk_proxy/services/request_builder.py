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

# 🎯 数学公式输出优先规则（CRITICAL）
**当输出包含数学公式时，请遵循以下关键原则：**
- 优先保证数学公式的正确显示，完全避免使用任何Markdown格式符号干扰数学内容
- 包含数学公式的段落必须使用纯文本表达，严禁同时使用**粗体**、*斜体*、`内联代码`、##标题等任何Markdown转换符号
- 禁止在数学公式前后使用标题符号（#、##、###等）和内联代码符号（`代码`），直接用纯文本描述
- 如需强调或分段，使用文字说明（如"重要："、"注意："、"第一部分："）替代所有Markdown格式符号
- 避免在数学公式周围使用任何Markdown格式组合，保持纯文本环境
- 示例正确："能量公式 $E = mc^2$ 很重要"
- 示例错误："**能量公式** $E = mc^2$ **很重要**"、"## 能量公式 $E = mc^2$"、"### 重要公式 $E = mc^2$"、"`能量公式` $E = mc^2$"

# 0. 总则
- 使用标准 Markdown 撰写正文。除非用户明确要求"预览 Markdown"，否则不要把普通正文包裹在 ```markdown 或 ```md 代码块中。
- 仅将"纯代码/数据/可视化定义"置于三反引号代码块中，并在代码块开头标注语言。
- 任何代码块/数学块/表格必须自成原子（完整开始与结束），禁止跨段/跨流片断裂。

# 1. 数学公式格式（CRITICAL - 避免换行问题）
- 行内数学：$...$（单行内使用，前后可无空格）；块级数学：$$...$$ 
- 禁止在数学公式和普通文本之间强制换行，保持内容连贯性
- 数学公式示例：
  - 行内正确：函数$f(x) = x^2 + 1$在定义域内连续
  - 行内正确：勾股定理$a^2 + b^2 = c^2$适用于直角三角形
  - 块级正确：$$e^{i\pi} + 1 = 0$$
- 常用写法：指数$x^{2}$、分数$\frac{a}{b}$、根号$\sqrt{x}$、下标$a_{i}$、希腊字母$\alpha$、省略号$\ldots$
- **重要**：避免在数学公式前后添加不必要的换行符，保持文本的自然流畅
- **🎯 数学公式与Markdown格式冲突规避**：
  - 当内容包含数学公式时，尽量使用纯文本表达，避免同时使用Markdown转换符号（如**粗体**、*斜体*、`代码`等）
  - 优先保证数学公式的正确显示，其他格式可以省略或使用替代表达
  - 如果必须强调某些内容，可以使用文字说明（如"重要："、"注意："）替代Markdown格式符号
  - 避免在包含数学公式的段落中使用复杂的Markdown格式组合
  - 示例：用"能量公式 $E = mc^2$ 很重要"而不是"**能量公式** $E = mc^2$ **很重要**"

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

# 4. 表格格式（CRITICAL - Android前端专用）
- 表格必须完整包含：表头行 + 分隔行 + 数据行
- 分隔行必须使用标准格式：| --- | --- | --- |
- 每行列数必须完全一致，缺失单元格用空字符串填充
- 标准表格示例：
| 功能 | LXC | KVM |
| --- | --- | --- |
| 虚拟化级别 | 容器级 | 硬件级 |
| 资源开销 | 低 | 高 |
- 错误示例：| 功能 | LXC | （缺少第三列）
- 错误示例：|---|---|---| （缺少空格）

# 5. 标题格式（CRITICAL - Android前端专用）
- 标题必须严格遵循：# 后接一个空格，然后是标题文本
- 正确：### LXC (Linux Containers)
- 错误：###LXC 或 ### LXC（两个空格）
- 标题前后必须有空行分隔
- 每个标题级别递增：# -> ## -> ### -> ####

# 6. 标准 Markdown 细则（CRITICAL - 防止格式问题）
- 标题：# 后空格；列表项符号后空格；链接：[text](url)；引用：> 后空格。
- **粗体文本**必须使用成对的双星号包围，如：**重要内容**
- 列表强约束：
  - 列表项起始仅允许使用 "-"、"*"、"+"、数字. 或 数字) 作为项目符号；其后必须紧跟一个空格。
  - 禁止在中文段首使用单个星号作为"强调"开头（会造成歧义）。如需强调中文词语，请使用成对的双星号（例如：**重要**），而非行首单星号。
  - 将全角符号（＊、﹡、•、·、・、﹒、∙）替换为半角列表符号，并补空格（例如：＊事项 -> * 事项；•要点 -> - 要点）。
- 段落之间使用空行分隔；不要输出大段无意义空行。
- **禁止在行内混合过多格式**：避免在同一行内出现"数学公式+粗体+列表符号"等复杂组合

# 6. 流式/分片输出纪律（CRITICAL - 保证内容完整性）
- 代码块、数学块、表格必须原子输出：开始标记 -> 完整内容 -> 结束标记；禁止中途插入解释文本。
- 如果开始了代码围栏（```lang），须在同一轮完成闭合（```）；如需撤回，立刻补出闭合标记，不留悬空。
- 行内代码使用单反引号配对，禁止在行尾遗留未闭合的 `。
- **数学公式与文本的连贯性**：当输出包含数学公式的段落时，确保公式与前后文本在同一流中输出，避免强制分行导致的显示问题。
- **粗体文本完整性**：确保**粗体**标记成对出现，避免只输出一个*导致格式混乱。

# 7. Android前端特殊优化（CRITICAL）
- 表格输出时确保每行都有相同数量的列，用空字符串填充缺失单元格
- 标题输出时，# ## ### 后必须有空格，前后必须有空行分隔
- **数学公式渲染优化**：避免在一行内混合多种格式（如标题+表格，数学公式+代码）
- **行内内容连贯性**：当输出"数学公式+文本"的组合时，确保它们在同一个段落内，不要强制换行
- 避免输出全角竖线｜、框线字符│┃，统一使用半角|
- 避免输出全角星号＊﹡，统一使用半角*
- 表格分隔行必须使用标准格式：| --- | --- |
- **省略号规范化**：统一使用LaTeX格式的省略号$\ldots$而不是纯文本的...或…

请严格遵守以上规则，特别是数学公式与文本的连贯性、粗体标记的完整性以及标题和表格格式，保证输出无需后处理即可被前端稳定解析与预览。"""

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
    # URL构建逻辑已移至 `openai.py`，此处仅返回一个占位符。
    # 实际请求URL将由调用方根据 `request_data.api_address` 决定。
    target_url = request_data.api_address or DEFAULT_OPENAI_API_BASE_URL

    # 更高兼容性的鉴权头：除 Bearer 外，补充 x-api-key；若为 Gemini 系列模型，再补充 x-goog-api-key
    headers = {
        "Authorization": f"Bearer {request_data.api_key}",
        "Content-Type": "application/json",
        "Accept": "text/event-stream",
        "x-api-key": request_data.api_key
    }
    # 对于以 OpenAI 兼容格式调用的 Gemini 模型，一些聚合商要求 x-goog-api-key
    if is_gemini_model_in_openai_format(request_data.model):
        headers["x-goog-api-key"] = request_data.api_key

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

    # 为官方 REST 同时携带请求头（兼容部分环境对 header 的要求）
    headers = {
        "Content-Type": "application/json",
        "x-goog-api-key": chat_input.api_key
    }
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