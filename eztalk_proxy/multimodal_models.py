from pydantic import BaseModel, Field
from typing import List, Union, Literal, Annotated, Optional, Dict, Any

# --- API Content Part 模型 ---
class BasePyApiContentPart(BaseModel):
    type: str

class PyTextContentPart(BasePyApiContentPart):
    type: Literal["text_content"] = "text_content" # 确保有默认值或在实例化时提供
    text: str

class PyFileUriContentPart(BasePyApiContentPart):
    type: Literal["file_uri_content"] = "file_uri_content" # 确保有默认值或在实例化时提供
    uri: str
    mime_type: str = Field(alias="mimeType")

class PyInlineDataContentPart(BasePyApiContentPart):
    type: Literal["inline_data_content"] = "inline_data_content" # 确保有默认值或在实例化时提供
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

# --- 新增：ThinkingConfig 和 GenerationConfig Pydantic 模型 ---
# 这些配置主要针对 Gemini 等支持高级生成的模型
class ThinkingConfigPy(BaseModel):
    include_thoughts: Optional[bool] = Field(None, alias="includeThoughts")
    thinking_budget: Optional[int] = Field(None, alias="thinkingBudget", ge=0, le=24576)

    class Config:
        populate_by_name = True # Pydantic V1 style, for V2 use model_config
        # model_config = {"populate_by_name": True} # Pydantic V2 style

class GenerationConfigPy(BaseModel):
    temperature: Optional[float] = Field(None, ge=0.0, le=2.0)
    top_p: Optional[float] = Field(None, alias="topP", ge=0.0, le=1.0)
    max_output_tokens: Optional[int] = Field(None, alias="maxOutputTokens", gt=0)
    # candidate_count: Optional[int] = None # 按需添加
    # stop_sequences: Optional[List[str]] = None # 按需添加
    
    thinking_config: Optional[ThinkingConfigPy] = Field(None, alias="thinkingConfig")

    class Config:
        populate_by_name = True # Pydantic V1 style, for V2 use model_config
        # model_config = {"populate_by_name": True} # Pydantic V2 style