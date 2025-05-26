from pydantic import BaseModel, Field, validator
from typing import List, Dict, Any, Literal, Optional, Union

# --- Definition for parts within a multipart message ---
class ApiImageUrlPart(BaseModel):
    url: str  # Expected: data URI "data:image/jpeg;base64,..."
    detail: Optional[str] = "auto"

class ApiContentPart(BaseModel): # For elements within MultipartContentIn.parts
    type: Literal["text", "image_url"]
    text: Optional[str] = None
    image_url: Optional[ApiImageUrlPart] = None

    @validator('text', always=True)
    def validate_text_based_on_type(cls, v, values):
        if 'type' in values: # Ensure 'type' is already processed
            if values['type'] == 'text' and v is None:
                raise ValueError('text is required when type is "text"')
            if values['type'] == 'image_url' and v is not None:
                raise ValueError('text must be None when type is "image_url"')
        return v

    @validator('image_url', always=True)
    def validate_image_url_based_on_type(cls, v, values):
        if 'type' in values: # Ensure 'type' is already processed
            if values['type'] == 'image_url' and v is None:
                raise ValueError('image_url is required when type is "image_url"')
            if values['type'] == 'text' and v is not None:
                raise ValueError('image_url must be None when type is "text"')
        return v
# --- Content part models definition ends ---

# --- New models to represent the polymorphic content from the client ---
class TextContentIn(BaseModel):
    """Corresponds to Android's TextContent. e.g., {"type": "text_content", "text": "Hello"}"""
    type: Literal["text_content"]
    text: str

class MultipartContentIn(BaseModel):
    """Corresponds to Android's MultipartContent. e.g., {"type": "multipart_content", "parts": [...]}"""
    type: Literal["multipart_content"]
    parts: List[ApiContentPart]
# --- Polymorphic content models definition ends ---

# --- OpenAI specific models (as provided by you) ---
class OpenAIToolCallFunction(BaseModel):
    name: Optional[str] = None
    arguments: Optional[str] = None

class OpenAIToolCall(BaseModel):
    index: Optional[int] = None
    id: Optional[str] = None
    type: Optional[Literal["function"]] = "function"
    function: OpenAIToolCallFunction
# --- OpenAI specific models definition ends ---

# --- Main ApiMessage model ---
class ApiMessage(BaseModel):
    role: str
    # content can be one of the new polymorphic types, a raw string (e.g., for AI responses), or None.
    content: Union[TextContentIn, MultipartContentIn, str, None] = None
    name: Optional[str] = None
    tool_call_id: Optional[str] = Field(default=None, alias="tool_call_id") # Keep alias if client sends snake_case
    tool_calls: Optional[List[OpenAIToolCall]] = Field(default=None, alias="tool_calls") # Keep alias if client sends snake_case

# --- Main ChatRequest model ---
class ChatRequest(BaseModel):
    # Fields from Kotlin ChatRequest, with aliases for camelCase from client
    api_address: Optional[str] = Field(default=None, alias="apiAddress")
    messages: List[ApiMessage]
    provider: Literal["openai", "google"]
    model: str
    api_key: Optional[str] = Field(default=None, alias="apiKey") # Optional to match Kotlin
    temperature: Optional[float] = Field(default=None, ge=0.0, le=2.0)
    top_p: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    max_tokens: Optional[int] = Field(default=None, gt=0, alias="maxTokens")
    tools: Optional[List[Dict[str, Any]]] = None
    tool_choice: Optional[Union[str, Dict[str, Any]]] = Field(default=None, alias="toolChoice")
    use_web_search: Optional[bool] = Field(default=None, alias="useWebSearch")
    force_google_reasoning_prompt: Optional[bool] = Field(default=None, alias="forceGoogleReasoningPrompt") # Matched to Kotlin
    custom_model_parameters: Optional[Dict[str, Any]] = Field(default=None, alias="customModelParameters")
    custom_extra_body: Optional[Dict[str, Any]] = Field(default=None, alias="customExtraBody")

    class Config:
        allow_population_by_field_name = True # Allows Pydantic to map camelCase from request to snake_case fields if aliases are not exact
        anystr_strip_whitespace = True # Good practice