# eztalk_proxy/models.py
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Literal, Optional, Union

# --- OpenAI Tool Call Models (保持您原来的定义) ---
class OpenAIToolCallFunction(BaseModel):
    name: Optional[str] = None
    arguments: Optional[str] = None

class OpenAIToolCall(BaseModel):
    index: Optional[int] = None
    id: Optional[str] = None
    type: Optional[Literal["function"]] = "function" # 保持 "function" 为默认值
    function: OpenAIToolCallFunction

# --- API Content Part 模型 (用于Gemini多模态parts) ---
# 这些对应您 Kotlin 中的 ApiContentPart 子类序列化后的样子
# 假设 Kotlin 序列化时，如 {"text_content": {"text": "..."}} 或 {"file_uri_content": {"uri": "...", "mimeType": "..."}}

class PyTextPartData(BaseModel):
    text: str

class PyFileUriPartData(BaseModel):
    uri: str
    mime_type: str = Field(alias="mimeType")

class PyInlineDataPartData(BaseModel):
    base64_data: str = Field(alias="base64Data")
    mime_type: str = Field(alias="mimeType")

# Content Part 包装器 (Wrapper classes)
class TextPartWrapper(BaseModel):
    text_content: PyTextPartData = Field(alias="text_content")

class FileUriPartWrapper(BaseModel):
    file_uri_content: PyFileUriPartData = Field(alias="file_uri_content")

class InlineDataPartWrapper(BaseModel):
    inline_data_content: PyInlineDataPartData = Field(alias="inline_data_content")

# Union of different content part wrappers for PartsApiMessagePy
IncomingApiContentPart = Union[TextPartWrapper, FileUriPartWrapper, InlineDataPartWrapper]


# --- API Message 模型 (对应前端 AbstractApiMessage.kt 及其子类) ---
# 使用 "message_format_type" 作为辨别器字段名，对应前端 sealed class 序列化时添加的 "type" 字段
# 前端kotlinx.serialization的默认辨别器字段名是 "type"，
# 如果您的前端SimpleTextApiMessage的@SerialName是 "simple_text_api_message"
# 那么辨别器传过来的值就是 "simple_text_api_message"

class BaseApiMessagePy(BaseModel):
    role: str
    name: Optional[str] = None
    # Pydantic Discriminated Union Field
    # 这个字段的值将由前端 kotlinx.serialization 在序列化 AbstractApiMessage 的子类时，
    # 根据子类的 @SerialName (或类名) 自动填充 (通常填充到名为 "type" 的字段)。
    # 我们在这里指定后端用 "message_type" (或者您可以继续用 "type") 来接收这个辨别器。
    # 为清楚起见，我们假设前端的辨别器字段名就是 "type"。
    # 请确保这里的 Literal 值与前端kotlinx序列化 AbstractApiMessage 子类时生成的 `type` 字段值完全一致。
    # 例如，如果前端 @SerialName("simple_text_message")，那么这里的 Literal 应该是 "simple_text_message"

class SimpleTextApiMessagePy(BaseApiMessagePy):
    message_type: Literal["simple_text_message"] = Field(alias="type") # 假设前端辨别器字段叫 "type"
    content: str # 非Gemini模型期望的简单文本内容
    # 如果 SimpleTextApiMessage 也需要 tool_calls 等，可以在这里添加
    tool_call_id: Optional[str] = None
    tool_calls: Optional[List[OpenAIToolCall]] = None


class PartsApiMessagePy(BaseApiMessagePy):
    message_type: Literal["parts_message"] = Field(alias="type") # 假设前端辨别器字段叫 "type"
    parts: List[IncomingApiContentPart] # Gemini模型期望的parts列表
    # PartsApiMessagePy 通常不需要 content 字段，但如果OpenAI的vision模型也通过这个路径，它可能同时接受 text part 和 image part
    # tool_calls 等也可以按需加入
    tool_call_id: Optional[str] = None
    tool_calls: Optional[List[OpenAIToolCall]] = None


# 辨别联合类型 (Discriminated Union)
# Pydantic 会根据 "message_type" 字段的值来决定将消息解析成哪个具体类型
AbstractApiMessagePy = Union[SimpleTextApiMessagePy, PartsApiMessagePy]


# --- 主要的聊天请求模型 (ChatRequest) ---
class ChatRequestModel(BaseModel):
    # 使用您原有的字段名和别名，但 messages 类型更新
    api_address: Optional[str] = Field(None, alias="apiAddress") # 对应前端 ChatRequest.apiAddress
    messages: List[AbstractApiMessagePy] = Field(..., discriminator="message_type") # 使用 'message_type' 作为辨别器
    provider: str # 您原定义是 Literal["openai", "google"], 如果需要更灵活可以是 str
    model: str
    api_key: str = Field(alias="apiKey") # 对应前端 ChatRequest.apiKey

    temperature: Optional[float] = Field(None, ge=0.0, le=2.0)
    top_p: Optional[float] = Field(None, ge=0.0, le=1.0) # 前端是 topP
    max_tokens: Optional[int] = Field(None, gt=0, alias="maxTokens") # 前端是 maxTokens

    tools: Optional[List[Dict[str, Any]]] = None
    tool_choice: Optional[Union[str, Dict[str, Any]]] = Field(None, alias="toolChoice")

    use_web_search: Optional[bool] = Field(None, alias="useWebSearch")
    force_custom_reasoning_prompt: Optional[bool] = Field(None, alias="forceCustomReasoningPrompt")
    custom_model_parameters: Optional[Dict[str, Any]] = Field(None, alias="customModelParameters")
    custom_extra_body: Optional[Dict[str, Any]] = Field(None, alias="customExtraBody")

    class Config:
        populate_by_name = True # 允许通过别名或字段名填充


# --- 后端发送给前端的SSE事件结构 (AppStreamEventPy) ---
# 这个模型应该严格匹配您前端 AppStreamEvent.kt 的定义
# (保持与您之前确认的 AppStreamEventPy 结构一致)
class AppStreamEventPy(BaseModel):
    type: str
    stage: Optional[str] = None
    results: Optional[List[Dict[str, Any]]] = None
    text: Optional[str] = None
    toolCallsData: Optional[List[Dict[str, Any]]] = Field(None, alias="data")
    id: Optional[str] = None
    name: Optional[str] = None
    arguments_obj: Optional[Dict[str, Any]] = Field(None, alias="argumentsObj")
    is_reasoning_step: Optional[bool] = Field(None, alias="isReasoningStep")
    reason: Optional[str] = None
    message: Optional[str] = None
    upstream_status: Optional[int] = Field(None, alias="upstreamStatus")
    timestamp: Optional[str] = None

    class Config:
        populate_by_name = True # 允许通过别名或字段名填充