from pydantic import BaseModel, Field
from typing import List, Dict, Any, Literal, Optional, Union, Annotated

# --- Models from multimodal_models.py ---

class BasePyApiContentPart(BaseModel):
    type: str
    model_config = {"populate_by_name": True}

class PyTextContentPart(BasePyApiContentPart):
    type: Literal["text_content"] = "text_content"
    text: str

class PyFileUriContentPart(BasePyApiContentPart):
    type: Literal["file_uri_content"] = "file_uri_content"
    uri: str
    mime_type: str = Field(alias="mimeType")

class PyInlineDataContentPart(BasePyApiContentPart):
    type: Literal["inline_data_content"] = "inline_data_content"
    base64_data: str = Field(alias="base64Data")
    mime_type: str = Field(alias="mimeType")

IncomingApiContentPart = Annotated[
    Union[
        PyTextContentPart,
        PyFileUriContentPart,
        PyInlineDataContentPart
    ],
    Field(discriminator="type")
]

class ThinkingConfigPy(BaseModel):
    include_thoughts: Optional[bool] = Field(None, alias="includeThoughts")
    thinking_budget: Optional[int] = Field(None, alias="thinkingBudget", ge=0, le=24576)
    model_config = {"populate_by_name": True}

class GenerationConfigPy(BaseModel):
    temperature: Optional[float] = Field(None, ge=0.0, le=2.0)
    top_p: Optional[float] = Field(None, alias="topP", ge=0.0, le=1.0)
    max_output_tokens: Optional[int] = Field(None, alias="maxOutputTokens", gt=0)
    thinking_config: Optional[ThinkingConfigPy] = Field(None, alias="thinkingConfig")
    model_config = {"populate_by_name": True}

# --- Models from models.py ---

class OpenAIToolCallFunction(BaseModel):
    name: Optional[str] = None
    arguments: Optional[str] = None
    model_config = {"populate_by_name": True}

class OpenAIToolCall(BaseModel):
    index: Optional[int] = None
    id: Optional[str] = None
    type: Optional[Literal["function"]] = "function"
    function: OpenAIToolCallFunction
    model_config = {"populate_by_name": True}

class BaseApiMessagePy(BaseModel):
    role: str
    name: Optional[str] = None
    message_type: str = Field(alias="type")
    model_config = {"populate_by_name": True}

class SimpleTextApiMessagePy(BaseApiMessagePy):
    message_type: Literal["simple_text_message"] = Field("simple_text_message", alias="type")
    content: str
    tool_call_id: Optional[str] = None
    tool_calls: Optional[List[OpenAIToolCall]] = None

class PartsApiMessagePy(BaseApiMessagePy):
    message_type: Literal["parts_message"] = Field("parts_message", alias="type")
    parts: List[IncomingApiContentPart]
    tool_call_id: Optional[str] = None
    tool_calls: Optional[List[OpenAIToolCall]] = None

AbstractApiMessagePy = Annotated[
    Union[SimpleTextApiMessagePy, PartsApiMessagePy],
    Field(discriminator="message_type")
]

class ChatRequestModel(BaseModel):
    api_address: Optional[str] = Field(None, alias="apiAddress")
    messages: List[AbstractApiMessagePy]
    provider: str
    model: str
    api_key: str = Field(alias="apiKey")
    temperature: Optional[float] = Field(None, ge=0.0, le=2.0)
    top_p: Optional[float] = Field(None, alias="topP", ge=0.0, le=1.0)
    max_tokens: Optional[int] = Field(None, alias="maxTokens", gt=0)
    generation_config: Optional[GenerationConfigPy] = Field(None, alias="generationConfig")
    tools: Optional[List[Dict[str, Any]]] = None
    tool_choice: Optional[Union[str, Dict[str, Any]]] = Field(None, alias="toolChoice")
    use_web_search: Optional[bool] = Field(None, alias="use_web_search")
    qwen_enable_search: Optional[bool] = Field(None, alias="qwenEnableSearch")
    force_custom_reasoning_prompt: Optional[bool] = Field(None, alias="forceCustomReasoningPrompt")
    custom_model_parameters: Optional[Dict[str, Any]] = Field(None, alias="customModelParameters")
    custom_extra_body: Optional[Dict[str, Any]] = Field(None, alias="customExtraBody")
    model_config = {"populate_by_name": True}

class WebSearchResult(BaseModel):
    title: str
    url: str
    snippet: str

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
    web_search_results: Optional[List[WebSearchResult]] = Field(None, alias="webSearchResults")
    model_config = {"populate_by_name": True}