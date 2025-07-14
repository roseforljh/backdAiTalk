import os
import logging
import httpx
import orjson
import asyncio
import base64
import shutil
import uuid
from typing import Optional, Dict, Any, AsyncGenerator, List, Union
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
    PyInlineDataContentPart
)
from ..core.config import (
    TEMP_UPLOAD_DIR,
    MAX_DOCUMENT_UPLOAD_SIZE_MB,
    API_TIMEOUT,
    GEMINI_SUPPORTED_UPLOAD_MIMETYPES
)
from ..utils.helpers import (
    get_current_time_iso,
    orjson_dumps_bytes_wrapper,
    extract_text_from_uploaded_document,
    extract_sse_lines
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


def cleanup_dirty_markdown(text: str) -> str:
    """
    Cleans up markdown text that may have escaped newlines.
    This is the primary fix for models that escape newlines (e.g., sending '\n' instead of a literal newline).
    Other aggressive replacements for characters like '*', '`', or '_' have been removed
    as they can corrupt complex, sensitive formats like LaTeX mathematical formulas or code snippets.
    """
    if not isinstance(text, str):
        return ""
    
    # Replace escaped newlines with actual newlines. 
    return text.replace('\\n', '\n')



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
    if uploaded_documents:
        for doc_file in uploaded_documents:
            content_type = doc_file.content_type.lower() if doc_file.content_type else ""
            # Process images and other directly supported multimodal types by Gemini
            if content_type in GEMINI_SUPPORTED_UPLOAD_MIMETYPES:
                try:
                    await doc_file.seek(0)
                    file_bytes = await doc_file.read()
                    multimodal_parts_in_memory.append({
                        "content_type": content_type,
                        "data": file_bytes,
                        "type": "inline_data" # Generalize to handle more than just images
                    })
                    logger.info(f"{log_prefix}: Staged multimodal file '{doc_file.filename}' for Base64 encoding.")
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

            new_multimodal_parts_for_openai: List[Dict[str, Any]] = []
            if multimodal_parts_in_memory:
                encoding_tasks = []
                for part in multimodal_parts_in_memory:
                    # For images, use the resizing function; for others, just encode
                    if part["content_type"] in SUPPORTED_IMAGE_MIME_TYPES_FOR_OPENAI:
                        encoding_tasks.append(asyncio.to_thread(resize_and_encode_image_sync, part["data"]))
                    else: # For audio, video etc.
                        encoding_tasks.append(asyncio.to_thread(base64.b64encode, part["data"]))

                encoded_results = await asyncio.gather(*encoding_tasks)
                
                for i, encoded_data_bytes in enumerate(encoded_results):
                    encoded_data_str = encoded_data_bytes if isinstance(encoded_data_bytes, str) else encoded_data_bytes.decode('utf-8')
                    content_type = multimodal_parts_in_memory[i]["content_type"]
                    data_uri = f"data:{content_type};base64,{encoded_data_str}"
                    
                    # OpenAI uses 'image_url' for all multimodal content via data URI
                    new_multimodal_parts_for_openai.append({"type": "image_url", "image_url": {"url": data_uri}})

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
                            data_uri = f"data:{part.mime_type};base64,{part.base64_data}"
                            content_parts.append({"type": "image_url", "image_url": {"url": data_uri}})

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
                        content_parts.extend(new_multimodal_parts_for_openai)

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
                search_results = await perform_web_search(user_query_for_search, request_id)
                if search_results:
                    yield orjson_dumps_bytes_wrapper(AppStreamEventPy(type="web_search_results", results=search_results).model_dump(by_alias=True, exclude_none=True))
                    search_context_content = generate_search_context_message_content(user_query_for_search, search_results)
                    
                    system_message_found = False
                    for msg in final_messages_for_llm:
                        if msg.get("role") == "system":
                            content = msg.get("content")
                            if isinstance(content, str):
                                msg["content"] = search_context_content + "\n\n" + content
                            system_message_found = True
                            break
                    if not system_message_found:
                        final_messages_for_llm.insert(0, {"role": "system", "content": search_context_content})
                    
                    logger.info(f"{log_prefix}: Injected web search context for OpenAI compatible request.")
                    yield orjson_dumps_bytes_wrapper(AppStreamEventPy(type="status_update", stage="Answering...").model_dump(by_alias=True, exclude_none=True))

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
                response.raise_for_status()
                
                async for line in response.aiter_lines():
                    if not first_chunk_received: first_chunk_received = True
                    if line.startswith("data:"):
                        json_str = line[5:].strip()
                        if json_str == "[DONE]": break
                        try:
                            sse_data = orjson.loads(json_str)
                            
                            # --- NEW: Clean up dirty markdown from non-Gemini models ---
                            if not chat_input.model.startswith("gemini"):
                                for choice in sse_data.get('choices', []):
                                    delta = choice.get('delta', {})
                                    if 'content' in delta and isinstance(delta['content'], str):
                                        delta['content'] = cleanup_dirty_markdown(delta['content'])
                            # --- END NEW ---

                            async for event in process_openai_like_sse_stream(sse_data, processing_state, request_id):
                                yield orjson_dumps_bytes_wrapper(AppStreamEventPy(**event).model_dump(by_alias=True, exclude_none=True))
                        except orjson.JSONDecodeError:
                            logger.warning(f"{log_prefix}: Skipping non-JSON line: {line}")

        except Exception as e:
            logger.error(f"{log_prefix}: An error occurred during the OpenAI-like stream: {e}", exc_info=True)
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