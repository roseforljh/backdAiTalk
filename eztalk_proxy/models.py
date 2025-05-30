from pydantic import BaseModel, Field
from typing import List, Dict, Any, Literal, Optional, Union, Annotated

# 确保从正确的路径导入
from .multimodal_models import IncomingApiContentPart, GenerationConfigPy

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
    message_type: Literal["simple_text_message"] = Field("simple_text_message", alias="type") # 确保有默认值
    content: str
    tool_call_id: Optional[str] = None
    tool_calls: Optional[List[OpenAIToolCall]] = None

class PartsApiMessagePy(BaseApiMessagePy):
    message_type: Literal["parts_message"] = Field("parts_message", alias="type") # 确保有默认值
    parts: List[IncomingApiContentPart]
    tool_call_id: Optional[str] = None
    tool_calls: Optional[List[OpenAIToolCall]] = None

AbstractApiMessagePy = Annotated[
    Union[SimpleTextApiMessagePy, PartsApiMessagePy],
    Field(discriminator="message_type") 
]

class ChatRequestModel(BaseModel):
    api_address: Optional[str] = Field(None, alias="apiAddress")
    messages: List[AbstractApiMessagePy] # <--- 确认这里没有多余的 Field(discriminator=...)
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
        populate_by_name = True # Pydantic V1 style
        # model_config = {"populate_by_name": True} # Pydantic V2 style

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
        populate_by_name = True # Pydantic V1 style
        # model_config = {"populate_by_name": True} # Pydantic V2 style