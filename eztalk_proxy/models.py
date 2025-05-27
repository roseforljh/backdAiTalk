# eztalk_proxy/models.py
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Literal, Optional, Union, Annotated

from .multimodal_models import IncomingApiContentPart, GenerationConfigPy # GenerationConfigPy 从 multimodal_models 导入

class OpenAIToolCallFunction(BaseModel):
    name: Optional[str] = None
    arguments: Optional[str] = None

class OpenAIToolCall(BaseModel):
    index: Optional[int] = None
    id: Optional[str] = None
    type: Optional[Literal["function"]] = "function"
    function: OpenAIToolCallFunction

class BaseApiMessagePy(BaseModel):
    role: str
    name: Optional[str] = None
    message_type: str = Field(alias="type") # Pydantic 使用此字段进行辨别

class SimpleTextApiMessagePy(BaseApiMessagePy):
    message_type: Literal["simple_text_message"] = Field(alias="type")
    content: str
    tool_call_id: Optional[str] = None
    tool_calls: Optional[List[OpenAIToolCall]] = None

class PartsApiMessagePy(BaseApiMessagePy):
    message_type: Literal["parts_message"] = Field(alias="type")
    parts: List[IncomingApiContentPart]
    tool_call_id: Optional[str] = None
    tool_calls: Optional[List[OpenAIToolCall]] = None

# AbstractApiMessagePy 已经正确定义了辨别器
AbstractApiMessagePy = Annotated[
    Union[SimpleTextApiMessagePy, PartsApiMessagePy],
    Field(discriminator="message_type") 
]

class ChatRequestModel(BaseModel):
    api_address: Optional[str] = Field(None, alias="apiAddress")
    # --- 修改点：移除 messages 字段上的 Field 和 discriminator ---
    # Pydantic 会自动对 List中的 AbstractApiMessagePy 应用其已定义的辨别器
    messages: List[AbstractApiMessagePy] # <--- 直接使用类型，不再需要 Field(..., discriminator=...)
    # --- 修改结束 ---
    provider: str
    model: str
    api_key: str = Field(alias="apiKey")

    temperature: Optional[float] = Field(None, ge=0.0, le=2.0)
    top_p: Optional[float] = Field(None, ge=0.0, le=1.0, alias="topP")
    max_tokens: Optional[int] = Field(None, gt=0, alias="maxTokens")

    generation_config: Optional[GenerationConfigPy] = Field(None, alias="generationConfig")

    tools: Optional[List[Dict[str, Any]]] = None
    tool_choice: Optional[Union[str, Dict[str, Any]]] = Field(None, alias="toolChoice")
    use_web_search: Optional[bool] = Field(None, alias="useWebSearch")
    force_custom_reasoning_prompt: Optional[bool] = Field(None, alias="forceCustomReasoningPrompt")
    custom_model_parameters: Optional[Dict[str, Any]] = Field(None, alias="customModelParameters")
    custom_extra_body: Optional[Dict[str, Any]] = Field(None, alias="customExtraBody")

    class Config:
        populate_by_name = True


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
        populate_by_name = True