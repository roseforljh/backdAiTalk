from pydantic import BaseModel, Field
from typing import List, Union, Literal, Annotated, Optional, Dict, Any

# --- API Content Part 模型 ---
class BasePyApiContentPart(BaseModel):
    type: str
    # model_config = {} # 如果需要特定配置，例如 extra='ignore' 或 'forbid'

class PyTextContentPart(BasePyApiContentPart):
    type: Literal["text_content"] = "text_content"
    text: str
    # model_config = {}

class PyFileUriContentPart(BasePyApiContentPart):
    type: Literal["file_uri_content"] = "file_uri_content"
    uri: str
    mime_type: str = Field(alias="mimeType")

    # 如果你需要从一个包含 "mimeType" 键的字典来填充这个模型，
    # 并且希望 Pydantic 自动将 "mimeType" 映射到 Python 的 mime_type 字段，
    # 那么需要 populate_by_name = True。
    # 同样，当序列化回JSON时，如果希望Python的 mime_type 字段被序列化为 "mimeType"，
    # by_alias=True (在 model_dump 中) 会处理这个，但 populate_by_name 是关于输入的。
    model_config = {"populate_by_name": True}


class PyInlineDataContentPart(BasePyApiContentPart):
    type: Literal["inline_data_content"] = "inline_data_content"
    base64_data: str = Field(alias="base64Data")
    mime_type: str = Field(alias="mimeType")

    # 理由同上
    model_config = {"populate_by_name": True}


IncomingApiContentPart = Annotated[
    Union[
        PyTextContentPart,
        PyFileUriContentPart,
        PyInlineDataContentPart
    ],
    Field(discriminator="type") # 'type' 是这些 Union 成员的辨别器字段
]

# --- ThinkingConfig 和 GenerationConfig Pydantic 模型 ---
class ThinkingConfigPy(BaseModel):
    include_thoughts: Optional[bool] = Field(None, alias="includeThoughts")
    thinking_budget: Optional[int] = Field(None, alias="thinkingBudget", ge=0, le=24576)

    model_config = {"populate_by_name": True} # Pydantic V2 style

class GenerationConfigPy(BaseModel):
    temperature: Optional[float] = Field(None, ge=0.0, le=2.0)
    top_p: Optional[float] = Field(None, alias="topP", ge=0.0, le=1.0)
    max_output_tokens: Optional[int] = Field(None, alias="maxOutputTokens", gt=0)
    # candidate_count: Optional[int] = None
    # stop_sequences: Optional[List[str]] = None
    
    thinking_config: Optional[ThinkingConfigPy] = Field(None, alias="thinkingConfig")

    model_config = {"populate_by_name": True} # Pydantic V2 style