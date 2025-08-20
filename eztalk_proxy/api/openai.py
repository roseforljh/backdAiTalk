import os
import logging
import httpx
import orjson
import asyncio
import base64
import uuid
from typing import Dict, Any, AsyncGenerator, List
from io import BytesIO

try:
    from PIL import Image
except ImportError:
    Image = None
    logging.warning("Pillow library not found. Image resizing optimization will not be available.")

from fastapi import Request, UploadFile
from fastapi.responses import StreamingResponse

from ..models.api_models import (
    ChatRequestModel,
    SimpleTextApiMessagePy,
    PartsApiMessagePy,
    AppStreamEventPy,
    PyTextContentPart,
    PyInlineDataContentPart,
    PyInputAudioContentPart
)
from ..core.config import (
    TEMP_UPLOAD_DIR,
    API_TIMEOUT,
    GEMINI_SUPPORTED_UPLOAD_MIMETYPES
)
from ..utils.helpers import (
    orjson_dumps_bytes_wrapper,
    extract_text_from_uploaded_document
)
from ..services.request_builder import prepare_openai_request
from ..services.stream_processor import (
    process_openai_like_sse_stream,
    handle_stream_error,
    handle_stream_cleanup,
    should_apply_custom_separator_logic
)
from ..services.web_search import perform_web_search, generate_search_context_message_content

logger = logging.getLogger("EzTalkProxy.Handlers.OpenAI")

SUPPORTED_IMAGE_MIME_TYPES_FOR_OPENAI = ["image/jpeg", "image/png", "image/gif", "image/webp"]

# 音频和视频类型定义
AUDIO_MIME_TYPES = [
    "audio/wav", "audio/x-wav", "audio/mpeg", "audio/mp3", "audio/aac", "audio/ogg",
    "audio/opus", "audio/flac", "audio/3gpp", "audio/amr", "audio/aiff", "audio/x-m4a",
    "audio/midi", "audio/webm"
]
VIDEO_MIME_TYPES = [
    "video/mp4", "video/mpeg", "video/quicktime", "video/x-msvideo", "video/x-flv",
    "video/x-matroska", "video/webm", "video/x-ms-wmv", "video/3gpp", "video/x-m4v"
]

def get_audio_format_from_mime_type(mime_type: str) -> str:
    """从MIME类型提取音频格式，用于OpenAI兼容格式"""
    mime_to_format = {
        "audio/wav": "wav",
        "audio/x-wav": "wav",
        "audio/mpeg": "mp3",
        "audio/mp3": "mp3",
        "audio/aac": "aac",
        "audio/ogg": "ogg",
        "audio/opus": "opus",
        "audio/flac": "flac",
        "audio/3gpp": "3gp",
        "audio/amr": "amr",
        "audio/aiff": "aiff",
        "audio/x-m4a": "m4a",
        "audio/midi": "midi",
        "audio/webm": "webm"
    }
    return mime_to_format.get(mime_type.lower(), mime_type.split('/')[-1])

def is_gemini_model(model_name: str) -> bool:
    """检测是否为Gemini模型"""
    if not model_name:
        return False
    model_lower = model_name.lower()
    return "gemini" in model_lower

def is_gemini_openai_compatible_request(chat_input) -> bool:
    """检测是否为使用OpenAI兼容格式的Gemini模型请求"""
    return is_gemini_model(chat_input.model)

def supports_multimodal_content(model_name: str) -> bool:
    """检测模型是否支持多模态内容（音频、视频、图像）
    
    注意：根据用户要求，不再对模型类型进行判断，
    所有模型都被视为支持多模态内容，让模型自己决定如何处理。
    """
    # 始终返回 True，不再限制模型类型
    return True

def is_google_official_api(api_address: str) -> bool:
    """检测是否为Google官方API地址"""
    if not api_address:
        return False
    api_lower = api_address.lower()
    return "generativelanguage.googleapis.com" in api_lower or "aiplatform.googleapis.com" in api_lower

MAX_IMAGE_DIMENSION = 2048

def resize_and_encode_image_sync(image_bytes: bytes) -> str:
    """
    Resizes an image if it exceeds max dimensions and encodes it to Base64.
    This is a CPU-bound function and should be run in a thread.
    """
    if not Image:
        # Pillow not installed, just encode without resizing
        return base64.b64encode(image_bytes).decode('utf-8')

    try:
        with Image.open(BytesIO(image_bytes)) as img:
            if img.width > MAX_IMAGE_DIMENSION or img.height > MAX_IMAGE_DIMENSION:
                img.thumbnail((MAX_IMAGE_DIMENSION, MAX_IMAGE_DIMENSION))
                
                output_buffer = BytesIO()
                # Preserve original format if possible, default to JPEG for wide compatibility
                img_format = img.format if img.format in ['JPEG', 'PNG', 'WEBP', 'GIF'] else 'JPEG'
                img.save(output_buffer, format=img_format)
                image_bytes = output_buffer.getvalue()

        return base64.b64encode(image_bytes).decode('utf-8')
    except Exception as e:
        logger.error(f"Failed to resize or encode image: {e}", exc_info=True)
        # Fallback to encoding the original bytes if processing fails
        return base64.b64encode(image_bytes).decode('utf-8')


async def handle_openai_compatible_request(
    chat_input: ChatRequestModel,
    uploaded_documents: List[UploadFile],
    fastapi_request_obj: Request,
    http_client: httpx.AsyncClient,
    request_id: str,
):
    log_prefix = f"RID-{request_id}"
    
    # Read all file contents into memory immediately, as the file stream can be closed.
    multimodal_parts_in_memory = []
    document_texts = []
    # 检测是否为Gemini模型以启用增强的多模态支持
    is_gemini_request = is_gemini_openai_compatible_request(chat_input)
    # 检测是否为Google官方API
    is_official_google_api = is_google_official_api(chat_input.api_address or "")
    
    if uploaded_documents:
        for doc_file in uploaded_documents:
            content_type = doc_file.content_type.lower() if doc_file.content_type else ""
            
            # 处理所有多模态文件（音频/视频/图像），不再基于模型类型进行限制
            if content_type in GEMINI_SUPPORTED_UPLOAD_MIMETYPES:
                # 如果是Gemini模型但使用第三方API，给出警告
                if is_gemini_request and not is_official_google_api and content_type in (AUDIO_MIME_TYPES + VIDEO_MIME_TYPES):
                    logger.warning(f"{log_prefix}: Using third-party API '{chat_input.api_address}' for Gemini model with audio/video content. This may not work properly.")
                try:
                    await doc_file.seek(0)
                    file_bytes = await doc_file.read()
                    file_size = len(file_bytes)
                    
                    logger.info(f"{log_prefix}: Processing multimodal file '{doc_file.filename}' ({file_size / 1024 / 1024:.2f} MB) for model {chat_input.model}")
                    
                    multimodal_parts_in_memory.append({
                        "content_type": content_type,
                        "data": file_bytes,
                        "type": "inline_data",
                        "filename": doc_file.filename,
                        "file_size": file_size
                    })
                    logger.info(f"{log_prefix}: Staged multimodal file '{doc_file.filename}' ({content_type}, {file_size / 1024 / 1024:.2f} MB) for processing with {chat_input.model}.")
                except Exception as e:
                    logger.error(f"{log_prefix}: Failed to read multimodal file {doc_file.filename} into memory: {e}", exc_info=True)
            # Process other documents for text extraction
            else:
                temp_file_path = ""
                try:
                    temp_file_path = os.path.join(TEMP_UPLOAD_DIR, f"{request_id}-{uuid.uuid4()}-{doc_file.filename}")
                    await doc_file.seek(0)
                    with open(temp_file_path, "wb") as f:
                        f.write(await doc_file.read())
                    
                    extracted_text = await extract_text_from_uploaded_document(
                        uploaded_file_path=temp_file_path,
                        mime_type=doc_file.content_type,
                        original_filename=doc_file.filename
                    )
                    if extracted_text:
                        document_texts.append(extracted_text)
                        logger.info(f"{log_prefix}: Successfully extracted text from document '{doc_file.filename}'.")
                except Exception as e:
                    logger.error(f"{log_prefix}: Failed to process document for text extraction {doc_file.filename}: {e}", exc_info=True)
                finally:
                    if temp_file_path and os.path.exists(temp_file_path):
                        os.remove(temp_file_path)

    async def event_stream_generator() -> AsyncGenerator[bytes, None]:
        final_messages_for_llm: List[Dict[str, Any]] = []
        user_query_for_search = ""
        processing_state: Dict[str, Any] = {}
        upstream_ok = False
        first_chunk_received = False

        try:
            # --- Final Refactored Processing Logic ---

            # 1. Prepare context from newly uploaded files
            full_document_context = ""
            if document_texts:
                full_document_context = "\n\n".join(document_texts)
                full_document_context = f"--- Document Content ---\n{full_document_context}\n--- End Document ---\n\n"
            
            # 检查是否使用第三方API处理Gemini音频/视频，给出特殊提示
            if (is_gemini_request and not is_official_google_api and
                multimodal_parts_in_memory and
                any(part.get("content_type", "") in AUDIO_MIME_TYPES + VIDEO_MIME_TYPES for part in multimodal_parts_in_memory)):
                
                api_domain = chat_input.api_address or "third-party service"
                warning_message = f"""⚠️ 音频/视频处理可能不稳定

您正在使用第三方API服务 ({api_domain}) 处理Gemini模型的音频/视频内容。

**可能的问题：**
• 第三方服务可能不完全支持Gemini的多模态功能
• 音频/视频处理可能被模型拒绝或处理不当

**建议解决方案：**
1. 使用Google官方API地址：`https://generativelanguage.googleapis.com/v1beta/openai/`
2. 或者切换到支持音频/视频的其他模型（如GPT-4o）

我们仍会尝试处理您的请求，但结果可能不理想。"""
                
                yield orjson_dumps_bytes_wrapper(AppStreamEventPy(type="error", message=warning_message).model_dump(by_alias=True, exclude_none=True))

            new_multimodal_parts_for_openai: List[Dict[str, Any]] = []
            if multimodal_parts_in_memory:
                encoding_tasks = []
                valid_parts = []
                
                for part in multimodal_parts_in_memory:
                    content_type = part["content_type"]
                    file_size = part.get("file_size", 0)
                    filename = part.get("filename", "unknown")
                    
                    # 对于大视频文件（>20MB），跳过Base64编码，建议使用Gemini原生API
                    if content_type in VIDEO_MIME_TYPES and file_size > 20 * 1024 * 1024:
                        logger.warning(f"{log_prefix}: Large video file '{filename}' ({file_size / 1024 / 1024:.2f} MB) is too large for OpenAI compatible format.")
                        
                        # 发送错误信息到前端
                        error_message = f"Large video file '{filename}' ({file_size / 1024 / 1024:.1f} MB) is too large for OpenAI compatible format. Please use Gemini native API for large videos."
                        yield orjson_dumps_bytes_wrapper(AppStreamEventPy(type="error", message=error_message).model_dump(by_alias=True, exclude_none=True))
                        
                        document_texts.append(f"""[Large Video File: {filename} ({file_size / 1024 / 1024:.1f} MB)]

This video file is too large for OpenAI compatible format processing.

For better video processing, please use:
- Gemini native API endpoint (generativelanguage.googleapis.com)
- This will enable File API upload for large videos

The video was uploaded but cannot be analyzed in OpenAI compatible mode due to size limitations.""")
                        continue
                    
                    # 对于中等大小的视频文件（>5MB），给出警告
                    elif content_type in VIDEO_MIME_TYPES and file_size > 5 * 1024 * 1024:
                        logger.warning(f"{log_prefix}: Medium video file '{filename}' ({file_size / 1024 / 1024:.2f} MB) may take longer to process.")
                        yield orjson_dumps_bytes_wrapper(AppStreamEventPy(type="status_update", stage=f"Processing video file ({file_size / 1024 / 1024:.1f} MB)...").model_dump(by_alias=True, exclude_none=True))
                    
                    valid_parts.append(part)
                    if content_type in SUPPORTED_IMAGE_MIME_TYPES_FOR_OPENAI:
                        encoding_tasks.append(asyncio.to_thread(resize_and_encode_image_sync, part["data"]))
                    else:
                        # 对于音频/视频文件，直接进行Base64编码
                        encoding_tasks.append(asyncio.to_thread(base64.b64encode, part["data"]))

                if encoding_tasks:
                    try:
                        logger.info(f"{log_prefix}: Starting encoding for {len(encoding_tasks)} files...")
                        
                        # 发送处理状态更新
                        yield orjson_dumps_bytes_wrapper(AppStreamEventPy(type="status_update", stage="Processing video/audio files...").model_dump(by_alias=True, exclude_none=True))
                        
                        # 添加超时保护，防止大视频文件编码卡死
                        encoded_results = await asyncio.wait_for(
                            asyncio.gather(*encoding_tasks),
                            timeout=120.0  # 2分钟超时
                        )
                        logger.info(f"{log_prefix}: Encoding completed successfully for {len(encoded_results)} files")
                        
                        # 发送编码完成状态
                        yield orjson_dumps_bytes_wrapper(AppStreamEventPy(type="status_update", stage="Files processed, analyzing...").model_dump(by_alias=True, exclude_none=True))
                        
                        for i, encoded_data_bytes in enumerate(encoded_results):
                            encoded_data_str = encoded_data_bytes if isinstance(encoded_data_bytes, str) else encoded_data_bytes.decode('utf-8')
                            content_type = valid_parts[i]["content_type"]
                            filename = valid_parts[i].get("filename", "unknown")
                            
                            # 根据官方文档，为不同类型的媒体使用正确的格式
                            if content_type in AUDIO_MIME_TYPES:
                                # 音频使用input_audio格式，根据官方文档规范
                                audio_format = get_audio_format_from_mime_type(content_type)
                                new_multimodal_parts_for_openai.append({
                                    "type": "input_audio",
                                    "input_audio": {
                                        "data": encoded_data_str,
                                        "format": audio_format
                                    }
                                })
                                logger.info(f"{log_prefix}: Successfully encoded and added audio {filename} ({content_type}) using input_audio format with format '{audio_format}'.")
                            elif content_type in VIDEO_MIME_TYPES:
                                # 视频仍使用image_url格式（根据官方文档，视频也通过data URI处理）
                                data_uri = f"data:{content_type};base64,{encoded_data_str}"
                                new_multimodal_parts_for_openai.append({"type": "image_url", "image_url": {"url": data_uri}})
                                logger.info(f"{log_prefix}: Successfully encoded and added video {filename} ({content_type}) using image_url format with data URI.")
                            else:
                                # 图像和其他类型使用image_url格式
                                data_uri = f"data:{content_type};base64,{encoded_data_str}"
                                new_multimodal_parts_for_openai.append({"type": "image_url", "image_url": {"url": data_uri}})
                                logger.info(f"{log_prefix}: Successfully encoded and added {filename} ({content_type}) using image_url format. Data URI length: {len(data_uri)}")
                            
                    except asyncio.TimeoutError:
                        logger.error(f"{log_prefix}: File encoding timed out after 2 minutes. This usually happens with large video files.")
                        
                        # 发送超时错误信息到前端
                        timeout_message = "Video/Audio file encoding timed out after 2 minutes. Please try with smaller files or use Gemini native API for large files."
                        yield orjson_dumps_bytes_wrapper(AppStreamEventPy(type="error", message=timeout_message).model_dump(by_alias=True, exclude_none=True))
                        
                        document_texts.append("[Video/Audio files were uploaded but encoding timed out. Please try with smaller files or use Gemini native API for large files.]")
                    except Exception as e:
                        logger.error(f"{log_prefix}: Error during file encoding: {e}", exc_info=True)
                        
                        # 发送编码错误信息到前端
                        encoding_error_message = f"Error processing uploaded files: {str(e)}"
                        yield orjson_dumps_bytes_wrapper(AppStreamEventPy(type="error", message=encoding_error_message).model_dump(by_alias=True, exclude_none=True))
                        
                        document_texts.append(f"[Multimodal files were uploaded but could not be processed due to encoding error: {str(e)}]")

            # 2. Build the final message list, preserving history correctly
            # --- Refactored Message Processing Logic ---
            for i, msg_abstract in enumerate(chat_input.messages):
                msg_dict: Dict[str, Any] = {"role": msg_abstract.role}
                is_last_user_message = (i == len(chat_input.messages) - 1 and msg_abstract.role == "user")

                content_parts = []
                
                # Step 1: Convert message content into a unified 'parts' format
                if isinstance(msg_abstract, SimpleTextApiMessagePy):
                    if msg_abstract.content:
                        content_parts.append({"type": "text", "text": msg_abstract.content})
                elif isinstance(msg_abstract, PartsApiMessagePy):
                    for part in msg_abstract.parts:
                        if isinstance(part, PyTextContentPart) and part.text:
                            content_parts.append({"type": "text", "text": part.text})
                        elif isinstance(part, PyInlineDataContentPart):
                            # 检查是否为音频类型，使用正确的格式
                            if part.mime_type in AUDIO_MIME_TYPES:
                                audio_format = get_audio_format_from_mime_type(part.mime_type)
                                content_parts.append({
                                    "type": "input_audio",
                                    "input_audio": {
                                        "data": part.base64_data,
                                        "format": audio_format
                                    }
                                })
                            else:
                                # 视频和图像使用image_url格式
                                data_uri = f"data:{part.mime_type};base64,{part.base64_data}"
                                content_parts.append({"type": "image_url", "image_url": {"url": data_uri}})
                        elif isinstance(part, PyInputAudioContentPart):
                            # 根据官方文档格式处理音频内容
                            content_parts.append({
                                "type": "input_audio",
                                "input_audio": {
                                    "data": part.data,
                                    "format": part.format
                                }
                            })

                # Step 2: If it's the last user message, inject new context
                if is_last_user_message:
                    # Extract user query for web search BEFORE adding context
                    user_query_for_search = " ".join([p.get("text", "") for p in content_parts if p.get("type") == "text"]).strip()

                    # Prepend document context to the text parts
                    if full_document_context:
                        # Find first text part to prepend to, or insert at the beginning
                        text_part_index = next((idx for idx, p in enumerate(content_parts) if p["type"] == "text"), -1)
                        if text_part_index != -1:
                            content_parts[text_part_index]["text"] = full_document_context + content_parts[text_part_index]["text"]
                        else:
                            content_parts.insert(0, {"type": "text", "text": full_document_context})
                    
                    # Append new multimodal parts (e.g., uploaded images)
                    if new_multimodal_parts_for_openai:
                        logger.info(f"{log_prefix}: Adding {len(new_multimodal_parts_for_openai)} multimodal parts to the last user message")
                        content_parts.extend(new_multimodal_parts_for_openai)
                        for idx, part in enumerate(new_multimodal_parts_for_openai):
                            if part["type"] == "input_audio":
                                logger.info(f"{log_prefix}: Multimodal part {idx+1}: type={part['type']}, format={part['input_audio']['format']}")
                            else:
                                logger.info(f"{log_prefix}: Multimodal part {idx+1}: type={part['type']}, data_uri_length={len(part.get('image_url', {}).get('url', ''))}")

                # Step 3: Finalize the content for the message payload
                if not content_parts:
                    msg_dict["content"] = ""
                elif len(content_parts) == 1 and content_parts[0]["type"] == "text":
                    msg_dict["content"] = content_parts[0]["text"]
                else:
                    msg_dict["content"] = content_parts
                
                final_messages_for_llm.append(msg_dict)

            # --- Web Search Logic ---
            if chat_input.use_web_search and user_query_for_search:
                yield orjson_dumps_bytes_wrapper(AppStreamEventPy(type="status_update", stage="Searching web...").model_dump(by_alias=True, exclude_none=True))
                try:
                    # 对于Gemini模型，我们仍然使用自定义搜索来提供搜索结果
                    # 因为即使Gemini有内置搜索，在OpenAI兼容格式下可能需要额外处理
                    search_results = await perform_web_search(user_query_for_search, request_id)
                    if search_results:
                        yield orjson_dumps_bytes_wrapper(AppStreamEventPy(type="web_search_results", results=search_results).model_dump(by_alias=True, exclude_none=True))
                        
                        # 简化搜索上下文，让AI更原生地处理搜索结果
                        search_context_content = f"""Search results for "{user_query_for_search}":

"""
                        for i, res in enumerate(search_results):
                            search_context_content += f"""{i + 1}. {res.get('title', 'N/A')}
{res.get('snippet', 'N/A')}
{res.get('href', 'N/A')}

"""
                        
                        # 不再添加系统消息，直接将搜索结果作为用户消息的一部分
                        if final_messages_for_llm and final_messages_for_llm[-1].get("role") == "user":
                            last_user_msg = final_messages_for_llm[-1]
                            content = last_user_msg.get("content", "")
                            if isinstance(content, str):
                                last_user_msg["content"] = search_context_content + "\n\n" + content
                            elif isinstance(content, list):
                                # 如果是多模态消息，在文本部分前添加搜索上下文
                                for i, part in enumerate(content):
                                    if part.get("type") == "text":
                                        part["text"] = search_context_content + "\n\n" + part["text"]
                                        break
                        
                        logger.info(f"{log_prefix}: Injected web search context for {'Gemini' if is_gemini_request else 'OpenAI'} compatible request.")
                        yield orjson_dumps_bytes_wrapper(AppStreamEventPy(type="status_update", stage="Answering...").model_dump(by_alias=True, exclude_none=True))
                except Exception as e_search:
                    logger.error(f"{log_prefix}: Web search failed, proceeding without search context. Error: {e_search}", exc_info=True)
                    yield orjson_dumps_bytes_wrapper(AppStreamEventPy(type="status_update", stage="Web search failed, answering directly...").model_dump(by_alias=True, exclude_none=True))

            # --- API Request and Streaming ---
            current_api_url, current_api_headers, current_api_payload = prepare_openai_request(
                request_data=chat_input,
                processed_messages=final_messages_for_llm,
                request_id=request_id
            )

            final_api_url = current_api_url
            if chat_input.api_address and chat_input.api_address.endswith('#'):
                final_api_url = chat_input.api_address[:-1]
                logger.info(f"{log_prefix}: Overriding API URL to: {final_api_url}")

            async with http_client.stream("POST", final_api_url, headers=current_api_headers, json=current_api_payload, timeout=API_TIMEOUT) as response:
                upstream_ok = response.status_code == 200
                
                # 如果响应不是200，记录详细错误信息
                if response.status_code != 200:
                    try:
                        error_content = await response.aread()
                        error_text = error_content.decode('utf-8')
                        logger.error(f"{log_prefix}: API request failed with status {response.status_code}. Response: {error_text}")
                    except Exception as e:
                        logger.error(f"{log_prefix}: API request failed with status {response.status_code}. Could not read error response: {e}")
                
                response.raise_for_status()
                
                buffer = bytearray()
                try:
                    done = False
                    async for chunk in response.aiter_bytes():
                        if done:
                            break
                        buffer.extend(chunk)
                        while not done:
                            separator_pos = buffer.find(b'\n\n')
                            if separator_pos == -1:
                                break

                            message_data = buffer[:separator_pos]
                            buffer = buffer[separator_pos + 2:]

                            if not message_data.strip():
                                continue

                            for line in message_data.split(b'\n'):
                                line = line.strip()
                                if line.startswith(b"data:"):
                                    json_str = line[5:].strip()
                                    if json_str == b"[DONE]":
                                        done = True
                                        break
                                    try:
                                        sse_data = orjson.loads(json_str.decode('utf-8'))
                                        # The problematic markdown cleaning logic for non-Gemini models has been removed
                                        # to prevent corruption of formatted text like code or LaTeX.
                                        
                                        async for event in process_openai_like_sse_stream(sse_data, processing_state, request_id, chat_input):
                                            yield orjson_dumps_bytes_wrapper(AppStreamEventPy(**event).model_dump(by_alias=True, exclude_none=True))
                                    except (orjson.JSONDecodeError, UnicodeDecodeError) as e:
                                        logger.warning(f"{log_prefix}: Skipping corrupted SSE line: {e}. Line: {line[:100]}...")
                except httpx.StreamClosed:
                    logger.warning(f"{log_prefix}: The stream was closed by the server, possibly due to completion or timeout.")
                except Exception as e:
                    logger.error(f"{log_prefix}: An unexpected error occurred during the OpenAI-like stream: {e}", exc_info=True)
                    is_upstream_ok = 'upstream_ok' in locals() and upstream_ok
                    is_first_chunk_received = 'first_chunk_received' in locals() and first_chunk_received
                    async for error_event in handle_stream_error(e, request_id, is_upstream_ok, is_first_chunk_received):
                        yield error_event
        finally:
            is_upstream_ok_final = 'upstream_ok' in locals() and upstream_ok
            use_custom_sep = should_apply_custom_separator_logic(chat_input, request_id, is_google_like_path=False, is_native_thinking_active=False)
            async for final_event in handle_stream_cleanup(processing_state, request_id, is_upstream_ok_final, use_custom_sep, chat_input.provider):
                yield final_event
            
            # No temp files to delete as we are reading into memory
            pass

    return StreamingResponse(event_stream_generator(), media_type="text/event-stream")