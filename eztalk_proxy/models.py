from pydantic import BaseModel, Field
from typing import List, Dict, Any, Literal, Optional, Union, Annotated

# 确保从正确的路径导入
# from .multimodal_models import IncomingApiContentPart, GenerationConfigPy # 假设这些定义在 multimodal_models.py
# 为了自包含和清晰，如果这些模型也需要Pydantic V2的config更新，它们也需要修改。
# 我们先假设 multimodal_models.py 中的模型也做了相应的 Pydantic V2 Config 更新。

# --- 从 multimodal_models.py 复制并更新 IncomingApiContentPart 和 GenerationConfigPy 的定义 ---
# (如果它们没有其他依赖，并且你也想在这里看到它们的V2配置)
# 否则，确保 multimodal_models.py 文件中的这些模型也更新了它们的 Config 为 model_config

class BasePyApiContentPart(BaseModel): # (来自 multimodal_models.py)
    type: str
    # model_config = {} # 示例，如果它有自己的配置

class PyTextContentPart(BasePyApiContentPart): # (来自 multimodal_models.py)
    type: Literal["text_content"] = "text_content"
    text: str
    # model_config = {}

class PyInlineDataContentPart(BasePyApiContentPart): # (来自 multimodal_models.py)
    type: Literal["inline_data_content"] = "inline_data_content"
    base64_data: str = Field(alias="base64Data")
    mime_type: str = Field(alias="mimeType")
    # model_config = {"populate_by_name": True} # 如果需要通过别名填充

IncomingApiContentPart = Annotated[ # (来自 multimodal_models.py)
    Union[
        PyTextContentPart,
        # PyFileUriContentPart, # 如果存在这个类型
        PyInlineDataContentPart
    ],
    Field(discriminator="type")
]

class ThinkingConfigPy(BaseModel): # (来自 multimodal_models.py)
    include_thoughts: Optional[bool] = Field(None, alias="includeThoughts")
    thinking_budget: Optional[int] = Field(None, alias="thinkingBudget", ge=0, le=24576)
    
    model_config = {"populate_by_name": True}

class GenerationConfigPy(BaseModel): # (来自 multimodal_models.py)
    temperature: Optional[float] = Field(None, ge=0.0, le=2.0)
    top_p: Optional[float] = Field(None, alias="topP", ge=0.0, le=1.0)
    max_output_tokens: Optional[int] = Field(None, alias="maxOutputTokens", gt=0)
    thinking_config: Optional[ThinkingConfigPy] = Field(None, alias="thinkingConfig")

    model_config = {"populate_by_name": True}
# --- 复制和更新结束 ---


class OpenAIToolCallFunction(BaseModel):
    name: Optional[str] = None
    arguments: Optional[str] = None
    # model_config = {} # 如果需要特定配置

class OpenAIToolCall(BaseModel):
    index: Optional[int] = None
    id: Optional[str] = None
    type: Optional[Literal["function"]] = "function"
    function: OpenAIToolCallFunction
    # model_config = {}

class BaseApiMessagePy(BaseModel):
    role: str
    name: Optional[str] = None
    message_type: str = Field(alias="type") # Pydantic 使用此字段进行辨别

    model_config = {"populate_by_name": True} # 允许通过别名 'type' 填充 'message_type'

class SimpleTextApiMessagePy(BaseApiMessagePy):
    # message_type 继承自 BaseApiMessagePy，这里用 Literal 约束其值
    # Field 的第一个参数是默认值
    message_type: Literal["simple_text_message"] = Field("simple_text_message", alias="type")
    content: str
    tool_call_id: Optional[str] = None
    tool_calls: Optional[List[OpenAIToolCall]] = None

    # model_config 继承自 BaseApiMessagePy，如果需要覆盖或添加，可以定义

class PartsApiMessagePy(BaseApiMessagePy):
    message_type: Literal["parts_message"] = Field("parts_message", alias="type")
    parts: List[IncomingApiContentPart]
    tool_call_id: Optional[str] = None
    tool_calls: Optional[List[OpenAIToolCall]] = None

    # model_config 继承自 BaseApiMessagePy

AbstractApiMessagePy = Annotated[
    Union[SimpleTextApiMessagePy, PartsApiMessagePy],
    Field(discriminator="message_type") # 辨别器字段是 message_type (Python名)
]

class ChatRequestModel(BaseModel):
    api_address: Optional[str] = Field(None, alias="apiAddress")
    messages: List[AbstractApiMessagePy]
    provider: str
    model: str
    api_key: str = Field(alias="apiKey")

    temperature: Optional[float] = Field(None, ge=0.0, le=2.0)
    top_p: Optional[float] = Field(None, alias="topP", ge=0.0, le=1.0)
    max_tokens: Optional[int] = Field(None, alias="maxTokens", gt=0) # 修正：确保 gt=0

    generation_config: Optional[GenerationConfigPy] = Field(None, alias="generationConfig")

    tools: Optional[List[Dict[str, Any]]] = None
    tool_choice: Optional[Union[str, Dict[str, Any]]] = Field(None, alias="toolChoice")
    use_web_search: Optional[bool] = Field(None, alias="useWebSearch")
    force_custom_reasoning_prompt: Optional[bool] = Field(None, alias="forceCustomReasoningPrompt")
    custom_model_parameters: Optional[Dict[str, Any]] = Field(None, alias="customModelParameters")
    custom_extra_body: Optional[Dict[str, Any]] = Field(None, alias="customExtraBody")

    model_config = {"populate_by_name": True}

class AppStreamEventPy(BaseModel):
    type: str # 注意：如果这也是一个辨别器，并且AppStreamEventPy是Union的一部分，需要相应处理
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

    model_config = {"populate_by_name": True}