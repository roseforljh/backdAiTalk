from pydantic import BaseModel, Field
from typing import List, Union, Literal, Annotated, Optional, Dict, Any

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