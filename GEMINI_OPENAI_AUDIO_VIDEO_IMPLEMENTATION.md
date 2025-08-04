# Gemini OpenAI兼容音频和视频理解功能实现文档

## 概述

根据谷歌官方OpenAI兼容接口文档，本项目已实现了Gemini大模型的音频和视频理解功能，支持OpenAI兼容格式的API调用。

## 实现的功能

### 1. 音频理解功能

#### 支持的音频格式
- WAV (`audio/wav`, `audio/x-wav`)
- MP3 (`audio/mpeg`, `audio/mp3`)
- AAC (`audio/aac`)
- OGG (`audio/ogg`)
- OPUS (`audio/opus`)
- FLAC (`audio/flac`)
- 3GP (`audio/3gpp`)
- AMR (`audio/amr`)
- AIFF (`audio/aiff`)
- M4A (`audio/x-m4a`)

#### 实现方式
根据官方文档，音频内容使用`input_audio`类型处理：

```python
{
    "type": "input_audio",
    "input_audio": {
        "data": "base64_encoded_audio_data",
        "format": "wav"  # 从MIME类型自动推断
    }
}
```

### 2. 视频理解功能

#### 支持的视频格式
- MP4 (`video/mp4`)
- MPEG (`video/mpeg`)
- QuickTime (`video/quicktime`)
- AVI (`video/x-msvideo`)
- FLV (`video/x-flv`)
- MKV (`video/x-matroska`)
- WebM (`video/webm`)
- WMV (`video/x-ms-wmv`)
- 3GP (`video/3gpp`)
- M4V (`video/x-m4v`)

#### 实现方式
根据官方文档，视频内容使用`image_url`类型通过data URI处理：

```python
{
    "type": "image_url",
    "image_url": {
        "url": "data:video/mp4;base64,base64_encoded_video_data"
    }
}
```

## 架构设计

### 双重处理逻辑分离

根据谷歌官方文档，本项目实现了两套完全分离的多模态处理逻辑：

1. **Gemini原生API处理** (`backend/eztalk_proxy/api/gemini.py`)
   - 使用Gemini原生格式：`inlineData`和`fileData`
   - 支持File API上传大文件（>20MB）
   - 直接使用Google的Python SDK
   - 支持视频剪辑、自定义帧率等高级功能

2. **OpenAI兼容API处理** (`backend/eztalk_proxy/api/openai.py`)
   - 使用OpenAI兼容格式：`input_audio`和`image_url`
   - 通过Base64编码处理文件
   - 遵循OpenAI API规范
   - 支持三行代码迁移

3. **智能路由** (`backend/eztalk_proxy/api/chat.py`)
   - 根据API地址自动选择处理方式
   - Google官方域名 → Gemini原生处理
   - 其他域名 → OpenAI兼容处理

## 核心实现文件

### 1. API模型更新 (`backend/eztalk_proxy/models/api_models.py`)

添加了新的音频内容部分类型：

```python
class PyInputAudioContentPart(BasePyApiContentPart):
    type: Literal["input_audio_content"] = "input_audio_content"
    data: str  # Base64 encoded audio data
    format: str  # Audio format like "wav", "mp3", etc.
```

### 2. OpenAI兼容处理器 (`backend/eztalk_proxy/api/openai.py`)

实现了完整的OpenAI兼容音频和视频处理逻辑：

- **音频格式映射函数**: `get_audio_format_from_mime_type()`
- **多模态内容处理**: 统一处理音频、视频和图像内容
- **官方限制处理**: 让Gemini官方API处理文件大小和格式限制

### 3. 请求构建器更新 (`backend/eztalk_proxy/services/request_builder.py`)

更新了Gemini REST API的消息转换逻辑，支持音频内容部分的转换。

## 使用示例

### 音频理解示例

```python
from openai import OpenAI

client = OpenAI(
    api_key="GEMINI_API_KEY",
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
)

with open("/path/to/audio.wav", "rb") as audio_file:
    base64_audio = base64.b64encode(audio_file.read()).decode('utf-8')

response = client.chat.completions.create(
    model="gemini-2.0-flash",
    messages=[
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": "请转录这个音频文件"
                },
                {
                    "type": "input_audio",
                    "input_audio": {
                        "data": base64_audio,
                        "format": "wav"
                    }
                }
            ]
        }
    ]
)
```

### 视频理解示例

```python
with open("/path/to/video.mp4", "rb") as video_file:
    base64_video = base64.b64encode(video_file.read()).decode('utf-8')

response = client.chat.completions.create(
    model="gemini-2.0-flash",
    messages=[
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": "请描述这个视频的内容"
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:video/mp4;base64,{base64_video}"
                    }
                }
            ]
        }
    ]
)
```

## 技术特性

### 1. 智能文件处理
- **大文件检测**: 自动检测大于10MB的音频/视频文件
- **编码优化**: 对小文件进行Base64编码，大文件提供友好提示
- **超时保护**: 30秒编码超时，防止系统阻塞

### 2. 格式兼容性
- **MIME类型映射**: 自动将MIME类型转换为OpenAI兼容格式
- **多格式支持**: 支持主流音频和视频格式
- **向后兼容**: 保持与现有图像处理逻辑的兼容性

### 3. 错误处理
- **编码失败处理**: 编码失败时提供详细错误信息
- **文件大小限制**: 对超大文件提供清晰的限制说明
- **格式验证**: 验证音频/视频格式的有效性

## 配置说明

### 环境变量
- `GEMINI_API_KEY`: Gemini API密钥
- `MAX_DOCUMENT_UPLOAD_SIZE_MB`: 最大文档上传大小（默认20MB）

### 支持的模型
- `gemini-2.0-flash`
- `gemini-2.5-flash`
- `gemini-2.5-pro`
- 其他Gemini系列模型

## 限制和注意事项

### 1. 文件大小限制
- **音频/视频文件**: 建议小于10MB以获得最佳性能
- **编码限制**: 大于5MB的文件会跳过编码以防止超时

### 2. 格式支持
- **音频格式**: 支持主流格式，自动格式检测
- **视频格式**: 通过data URI方式支持，与图像处理逻辑一致

### 3. 性能考虑
- **编码时间**: Base64编码可能需要时间，已添加超时保护
- **内存使用**: 大文件会占用较多内存，建议合理控制文件大小

## 测试建议

1. **小文件测试**: 使用小于5MB的音频/视频文件测试基本功能
2. **格式兼容性**: 测试不同音频和视频格式的支持情况
3. **错误处理**: 测试大文件和不支持格式的错误处理
4. **性能测试**: 测试编码时间和内存使用情况

## 更新日志

- **2025-01-25**: 初始实现音频和视频理解功能
- 添加了`PyInputAudioContentPart`模型类
- 实现了`get_audio_format_from_mime_type()`格式映射
- 更新了OpenAI兼容处理逻辑
- 添加了智能文件大小处理和超时保护