import logging
import httpx
import orjson
import asyncio
import base64
import io
from typing import List

import google.generativeai as genai
import docx
from fastapi import Request, UploadFile
from fastapi.responses import StreamingResponse

from ..models.api_models import (
    ChatRequestModel,
    AppStreamEventPy,
    PartsApiMessagePy,
    AbstractApiMessagePy,
    SimpleTextApiMessagePy,
    PyTextContentPart,
    PyInlineDataContentPart,
    IncomingApiContentPart,
    PyFileUriContentPart,
    WebSearchResult
)
from ..core.config import (
    API_TIMEOUT,
    GOOGLE_API_KEY_ENV,
    MAX_DOCUMENT_UPLOAD_SIZE_MB
)
from ..utils.helpers import (
    get_current_time_iso,
    orjson_dumps_bytes_wrapper
)
from ..services.request_builder import prepare_gemini_rest_api_request
from ..services.stream_processor import (
    process_openai_like_sse_stream,
    handle_stream_error,
    handle_stream_cleanup,
    should_apply_custom_separator_logic
)
from ..services.web_search import perform_web_search, generate_search_context_message_content

logger = logging.getLogger("EzTalkProxy.Handlers.Gemini")

IMAGE_MIME_TYPES = ["image/png", "image/jpeg", "image/webp", "image/heic", "image/heif"]
DOCUMENT_MIME_TYPES = [
    "application/pdf",
    "application/x-javascript", "text/javascript",
    "application/x-python", "text/x-python",
    "text/plain",
    "text/html",
    "text/css",
    "text/md",
    "text/markdown",
    "text/csv",
    "text/xml",
    "text/rtf"
]
VIDEO_MIME_TYPES = [
    "video/mp4", "video/mpeg", "video/quicktime", "video/x-msvideo", "video/x-flv",
    "video/x-matroska", "video/webm", "video/x-ms-wmv", "video/3gpp", "video/x-m4v"
]
AUDIO_MIME_TYPES = [
    "audio/wav", "audio/mpeg", "audio/aac", "audio/ogg", "audio/opus", "audio/flac", "audio/3gpp"
]

import re

def cleanup_dirty_markdown(text: str) -> str:
    """
    Cleans up markdown text that may have escaped newlines or have extra spaces around markdown tokens
    due to chunked streaming.
    """
    if not isinstance(text, str):
        return ""
    
    # 1. Replace escaped newlines with actual newlines.
    text = text.replace('\\n', '\n')

    # 2. Collapse multiple spaces into a single space, but not for lines starting with spaces (to preserve code indentation).
    lines = text.split('\n')
    cleaned_lines = []
    for line in lines:
        if line.startswith(' '):
            cleaned_lines.append(line)
        else:
            cleaned_lines.append(re.sub(r' +', ' ', line))
    text = '\n'.join(cleaned_lines)

    # 3. Fix spaces around markdown bold/italic markers.
    # Example: "** text **" -> "**text**"
    text = re.sub(r'\*\*\s+(.*?)\s+\*\*', r'**\1**', text)
    text = re.sub(r'\*\s+(.*?)\s+\*', r'*\1*', text)

    # 4. Fix spaces inside parentheses or brackets that are adjacent to markdown.
    # Example: "**Team Vitality** ( France )" -> "**Team Vitality** (France)"
    text = re.sub(r'\(\s+(.*?)\s+\)', r'(\1)', text)
    text = re.sub(r'\[\s+(.*?)\s+\]', r'[\1]', text)
    
    return text


async def sse_event_serializer_rest(event_data: AppStreamEventPy) -> bytes:
    return orjson_dumps_bytes_wrapper(event_data.model_dump(by_alias=True, exclude_none=True))

async def handle_gemini_request(
    gemini_chat_input: ChatRequestModel,
    uploaded_files: List[UploadFile],
    fastapi_request_obj: Request,
    http_client: httpx.AsyncClient,
    request_id: str,
):
    log_prefix = f"RID-{request_id}"
    active_messages_for_llm: List[AbstractApiMessagePy] = [msg.model_copy(deep=True) for msg in gemini_chat_input.messages]
    newly_created_multimodal_parts: List[IncomingApiContentPart] = []

    if GOOGLE_API_KEY_ENV:
        genai.configure(api_key=GOOGLE_API_KEY_ENV)

    if uploaded_files:
        for uploaded_file in uploaded_files:
            mime_type = uploaded_file.content_type.lower() if uploaded_file.content_type else ""
            filename = uploaded_file.filename or "unknown"
            
            try:
                if mime_type in IMAGE_MIME_TYPES:
                    await uploaded_file.seek(0)
                    file_bytes = await uploaded_file.read()
                    base64_data = base64.b64encode(file_bytes).decode('utf-8')
                    newly_created_multimodal_parts.append(PyInlineDataContentPart(
                        type="inline_data_content", mimeType=mime_type, base64Data=base64_data
                    ))
                elif mime_type in DOCUMENT_MIME_TYPES:
                    logger.info(f"{log_prefix}: Processing document for Gemini: {filename} ({mime_type})")
                    await uploaded_file.seek(0)
                    file_bytes = await uploaded_file.read()
                    base64_data = base64.b64encode(file_bytes).decode('utf-8')
                    newly_created_multimodal_parts.append(PyInlineDataContentPart(
                        type="inline_data_content", mimeType=mime_type, base64Data=base64_data
                    ))
                elif mime_type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
                    logger.info(f"{log_prefix}: Extracting text from DOCX file for Gemini: {filename}")
                    await uploaded_file.seek(0)
                    file_bytes = await uploaded_file.read()
                    try:
                        doc_stream = io.BytesIO(file_bytes)
                        document = docx.Document(doc_stream)
                        full_text = "\n".join([para.text for para in document.paragraphs])
                        
                        extracted_text_content = f"\n\n--- START OF DOCUMENT: {filename} ---\n\n{full_text}\n\n--- END OF DOCUMENT: {filename} ---\n"
                        
                        newly_created_multimodal_parts.append(PyTextContentPart(
                            type="text_content", text=extracted_text_content
                        ))
                    except Exception as docx_e:
                        logger.error(f"{log_prefix}: Failed to extract text from DOCX file {filename}: {docx_e}", exc_info=True)

                elif mime_type in AUDIO_MIME_TYPES:
                    logger.info(f"{log_prefix}: Processing audio for Gemini: {filename} ({mime_type})")
                    await uploaded_file.seek(0)
                    file_bytes = await uploaded_file.read()
                    base64_data = base64.b64encode(file_bytes).decode('utf-8')
                    newly_created_multimodal_parts.append(PyInlineDataContentPart(
                        type="inline_data_content", mimeType=mime_type, base64Data=base64_data
                    ))
                elif mime_type in VIDEO_MIME_TYPES:
                    logger.info(f"{log_prefix}: Processing video for Gemini: {filename} ({mime_type})")
                    await uploaded_file.seek(0)
                    file_bytes = await uploaded_file.read()
                    file_size = len(file_bytes)
                    
                    # Use File API for large files as recommended by Google
                    if file_size > (MAX_DOCUMENT_UPLOAD_SIZE_MB * 1024 * 1024):
                        logger.info(f"{log_prefix}: Uploading large video '{filename}' ({file_size / 1024 / 1024:.2f} MB) to Gemini File API.")
                        try:
                            # We need to run this in a separate thread as the SDK is synchronous
                            loop = asyncio.get_running_loop()
                            video_file = await loop.run_in_executor(
                                None,
                                lambda: genai.upload_file(
                                    path=io.BytesIO(file_bytes),
                                    display_name=filename,
                                    mime_type=mime_type
                                )
                            )
                            logger.info(f"{log_prefix}: Uploaded '{filename}', waiting for processing. URI: {video_file.uri}")
                            
                            # Wait for the file to be processed
                            while video_file.state.name == "PROCESSING":
                                await asyncio.sleep(5) # Non-blocking sleep
                                video_file = await loop.run_in_executor(None, lambda: genai.get_file(video_file.name))
                                logger.info(f"{log_prefix}: File '{filename}' state: {video_file.state.name}")

                            if video_file.state.name == "ACTIVE":
                                newly_created_multimodal_parts.append(PyFileUriContentPart(
                                    type="file_uri_content", fileUri=video_file.uri, mimeType=mime_type
                                ))
                                logger.info(f"{log_prefix}: File '{filename}' is active and ready to use.")
                            else:
                                logger.error(f"{log_prefix}: File '{filename}' failed to process. State: {video_file.state.name}")

                        except Exception as file_api_e:
                            logger.error(f"{log_prefix}: Gemini File API upload failed for {filename}: {file_api_e}", exc_info=True)
                    else:
                        base64_data = base64.b64encode(file_bytes).decode('utf-8')
                        newly_created_multimodal_parts.append(PyInlineDataContentPart(
                            type="inline_data_content", mimeType=mime_type, base64Data=base64_data
                        ))
                else:
                    logger.warning(f"{log_prefix}: Skipping unsupported file type for Gemini: {filename} ({mime_type})")
            except Exception as e:
                logger.error(f"{log_prefix}: Error processing file {filename} for Gemini: {e}", exc_info=True)

    if newly_created_multimodal_parts:
        last_user_message_idx = -1
        for i in range(len(active_messages_for_llm) - 1, -1, -1):
            if active_messages_for_llm[i].role == "user":
                last_user_message_idx = i
                break
        
        if last_user_message_idx != -1:
            user_msg = active_messages_for_llm[last_user_message_idx]
            if isinstance(user_msg, PartsApiMessagePy):
                user_msg.parts.extend(newly_created_multimodal_parts)
            elif isinstance(user_msg, SimpleTextApiMessagePy):
                initial_text_part = [PyTextContentPart(type="text_content", text=user_msg.content)] if user_msg.content else []
                active_messages_for_llm[last_user_message_idx] = PartsApiMessagePy(
                    role="user", parts=initial_text_part + newly_created_multimodal_parts
                )
        else:
            active_messages_for_llm.append(PartsApiMessagePy(role="user", parts=newly_created_multimodal_parts))

    try:
        target_url, headers, json_payload = prepare_gemini_rest_api_request(
            chat_input=gemini_chat_input.model_copy(update={'messages': active_messages_for_llm}),
            request_id=request_id
        )
    except Exception as e_prepare:
        async def error_gen():
            yield await sse_event_serializer_rest(AppStreamEventPy(type="error", message=f"请求准备错误: {e_prepare}", timestamp=get_current_time_iso()))
            yield await sse_event_serializer_rest(AppStreamEventPy(type="finish", reason="request_error", timestamp=get_current_time_iso()))
        return StreamingResponse(error_gen(), media_type="text/event-stream")

    async def stream_generator():
        processing_state = {}
        upstream_ok = False
        first_chunk_received = False
        full_text = ""
        grounding_supports = []
        grounding_chunks_storage = []
        
        try:
            async with http_client.stream("POST", target_url, headers=headers, json=json_payload, timeout=API_TIMEOUT) as response:
                upstream_ok = response.is_success
                if not upstream_ok:
                    error_body = await response.aread()
                    logger.error(f"{log_prefix}: Gemini upstream error {response.status_code}: {error_body.decode(errors='ignore')}")
                    response.raise_for_status()

                async for line in response.aiter_lines():
                    if not first_chunk_received:
                        first_chunk_received = True
                    
                    if line.startswith("data:"):
                        json_str = line[len("data:"):].strip()
                        try:
                            logger.debug(f"Received from Gemini: {json_str}")
                            sse_data = orjson.loads(json_str)
                            openai_like_sse = {"id": f"gemini-{request_id}", "choices": []}
                            
                            for candidate in sse_data.get("candidates", []):
                                grounding_metadata = candidate.get("groundingMetadata")
                                if grounding_metadata:
                                    search_queries = grounding_metadata.get("webSearchQueries", [])
                                    if "groundingChunks" in grounding_metadata:
                                        grounding_chunks_storage.extend(grounding_metadata["groundingChunks"])
                                    
                                    if "groundingSupports" in grounding_metadata:
                                        grounding_supports.extend(grounding_metadata["groundingSupports"])

                                    if search_queries:
                                        logger.info(f"{log_prefix}: Gemini used web search with queries: {search_queries}")

                                    if grounding_chunks_storage:
                                        web_results = [
                                            WebSearchResult(
                                                title=chunk.get("web", {}).get("title", "Unknown Source"),
                                                url=chunk.get("web", {}).get("uri", "#"),
                                                snippet=f"Source: {chunk.get('web', {}).get('title', 'N/A')}"
                                            )
                                            for chunk in grounding_chunks_storage if chunk.get("web")
                                        ]
                                        if web_results:
                                            yield await sse_event_serializer_rest(AppStreamEventPy(
                                                type="web_search_results",
                                                web_search_results=web_results
                                            ))

                                content_parts = candidate.get("content", {}).get("parts", [])
                                is_thought_part = any("thought" in part for part in content_parts)

                                if is_thought_part:
                                    for part in content_parts:
                                        if "thought" in part and part.get("text"):
                                            reasoning_text = part["text"]
                                            yield await sse_event_serializer_rest(AppStreamEventPy(type="reasoning", text=reasoning_text))
                                else:
                                    delta = {}
                                    for part in content_parts:
                                        if "text" in part:
                                            text_chunk = part["text"]
                                            delta["content"] = text_chunk
                                            full_text += text_chunk
                                    
                                    if delta:
                                        if "content" in delta and isinstance(delta["content"], str):
                                            delta["content"] = cleanup_dirty_markdown(delta["content"])
                                        choice = {
                                            "delta": delta,
                                            "finish_reason": candidate.get("finishReason")
                                        }
                                        openai_like_sse = {"id": f"gemini-{request_id}", "choices": [choice]}
                                        async for event in process_openai_like_sse_stream(openai_like_sse, processing_state, request_id):
                                            yield await sse_event_serializer_rest(AppStreamEventPy(**event))

                        except orjson.JSONDecodeError:
                            logger.warning(f"{log_prefix}: Skipping non-JSON line in Gemini stream: {line}")

        except Exception as e:
            logger.error(f"{log_prefix}: An error occurred during the Gemini stream: {e}", exc_info=True)
            async for error_event in handle_stream_error(e, request_id, upstream_ok, first_chunk_received):
                yield error_event
        finally:
            is_native_thinking = bool(gemini_chat_input.generation_config and gemini_chat_input.generation_config.thinking_config)
            use_custom_sep = should_apply_custom_separator_logic(gemini_chat_input, request_id, is_google_like_path=True, is_native_thinking_active=is_native_thinking)
            if grounding_supports and grounding_chunks_storage:
                logger.info(f"Processing citations for full text. Supports: {len(grounding_supports)}, Chunks: {len(grounding_chunks_storage)}")
                
                sorted_supports = sorted(grounding_supports, key=lambda s: s.get("segment", {}).get("endIndex", 0), reverse=True)

                for support in sorted_supports:
                    segment = support.get("segment", {})
                    end_index = segment.get("endIndex")
                    if end_index is None:
                        continue

                    chunk_indices = support.get("groundingChunkIndices", [])
                    if chunk_indices:
                        citation_links = []
                        for i in chunk_indices:
                            if i < len(grounding_chunks_storage):
                                uri = grounding_chunks_storage[i].get("web", {}).get("uri")
                                if uri:
                                    citation_links.append(f"[{i + 1}]({uri})")

                        if citation_links:
                            citation_string = " " + ", ".join(citation_links)
                            full_text = full_text[:end_index] + citation_string + full_text[end_index:]
                
                # Since we cannot re-stream, we will log the result for now.
                logger.info(f"Text with citations: {full_text}")


            async for final_event in handle_stream_cleanup(processing_state, request_id, upstream_ok, use_custom_sep, gemini_chat_input.provider):
                yield final_event

    return StreamingResponse(stream_generator(), media_type="text/event-stream")