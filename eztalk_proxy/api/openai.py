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
    PyTextContentPart
)
from ..core.config import (
    TEMP_UPLOAD_DIR,
    MAX_DOCUMENT_UPLOAD_SIZE_MB,
    API_TIMEOUT
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


async def handle_openai_compatible_request(
    chat_input: ChatRequestModel,
    uploaded_documents: List[UploadFile],
    fastapi_request_obj: Request,
    http_client: httpx.AsyncClient,
    request_id: str,
):
    log_prefix = f"RID-{request_id}"
    
    # Ensure the temporary upload directory exists
    if not os.path.exists(TEMP_UPLOAD_DIR):
        os.makedirs(TEMP_UPLOAD_DIR)
        
    # Read all file contents into memory immediately, as the file stream can be closed.
    image_parts_in_memory = []
    document_texts = []
    if uploaded_documents:
        for doc_file in uploaded_documents:
            # Process images
            if doc_file.content_type and doc_file.content_type.lower() in SUPPORTED_IMAGE_MIME_TYPES_FOR_OPENAI:
                try:
                    await doc_file.seek(0)
                    image_bytes = await doc_file.read()
                    image_parts_in_memory.append({
                        "content_type": doc_file.content_type,
                        "data": image_bytes
                    })
                except Exception as e:
                    logger.error(f"{log_prefix}: Failed to read image file {doc_file.filename} into memory: {e}", exc_info=True)
            # Process other documents
            else:
                temp_file_path = ""
                try:
                    # Use a unique filename to avoid collisions
                    temp_file_path = os.path.join(TEMP_UPLOAD_DIR, f"{request_id}-{uuid.uuid4()}-{doc_file.filename}")
                    
                    # Ensure the directory exists before writing the file
                    os.makedirs(os.path.dirname(temp_file_path), exist_ok=True)
                    
                    # Save the uploaded file to the temporary path
                    await doc_file.seek(0)
                    with open(temp_file_path, "wb") as f:
                        f.write(await doc_file.read())
                    
                    # Extract text content from the saved document, passing the correct content_type
                    extracted_text = await extract_text_from_uploaded_document(
                        uploaded_file_path=temp_file_path,
                        mime_type=doc_file.content_type,
                        original_filename=doc_file.filename
                    )
                    if extracted_text:
                        document_texts.append(extracted_text)
                        logger.info(f"{log_prefix}: Successfully extracted text from document '{doc_file.filename}'.")
                except Exception as e:
                    logger.error(f"{log_prefix}: Failed to process document {doc_file.filename}: {e}", exc_info=True)
                finally:
                    # Clean up the temporary file
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

            new_image_parts_for_openai: List[Dict[str, Any]] = []
            if image_parts_in_memory:
                # Offload CPU-bound resizing and Base64 encoding to a thread pool
                encoding_tasks = [
                    asyncio.to_thread(resize_and_encode_image_sync, part["data"])
                    for part in image_parts_in_memory
                ]
                encoded_results = await asyncio.gather(*encoding_tasks)
                for i, encoded_data in enumerate(encoded_results):
                    content_type = image_parts_in_memory[i]["content_type"]
                    data_uri = f"data:{content_type};base64,{encoded_data}"
                    new_image_parts_for_openai.append({"type": "image_url", "image_url": {"url": data_uri}})

            # 2. Build the final message list, preserving history correctly
            for i, msg_abstract in enumerate(chat_input.messages):
                msg_dict: Dict[str, Any] = {"role": msg_abstract.role}
                is_last_user_message = (i == len(chat_input.messages) - 1 and msg_abstract.role == "user")
                
                current_content_parts = []
                
                # A. Reconstruct content from history, preserving all parts
                if isinstance(msg_abstract, SimpleTextApiMessagePy):
                    if msg_abstract.content:
                        current_content_parts.append({"type": "text", "text": msg_abstract.content})
                elif isinstance(msg_abstract, PartsApiMessagePy):
                    for part in msg_abstract.parts:
                        if isinstance(part, PyTextContentPart) and part.text:
                            current_content_parts.append({"type": "text", "text": part.text})
                        # This is the critical fix for historical images:
                        elif isinstance(part, PyInlineDataContentPart):
                            data_uri = f"data:{part.mime_type};base64,{part.base64_data}"
                            current_content_parts.append({"type": "image_url", "image_url": {"url": data_uri}})

                # B. For the last user message, add new document/image context
                if is_last_user_message:
                    # Extract user query text for web search from the original parts
                    user_query_for_search = " ".join([p.get("text", "") for p in current_content_parts if p.get("type") == "text"]).strip()

                    if full_document_context:
                        current_content_parts.insert(0, {"type": "text", "text": full_document_context})
                    if new_image_parts_for_openai:
                        current_content_parts.extend(new_image_parts_for_openai)

                # C. Finalize content format for the message
                if not current_content_parts:
                    msg_dict["content"] = ""
                elif len(current_content_parts) == 1 and current_content_parts[0]["type"] == "text":
                    msg_dict["content"] = current_content_parts[0]["text"]
                else:
                    msg_dict["content"] = current_content_parts
                
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

            async with http_client.stream("POST", current_api_url, headers=current_api_headers, json=current_api_payload, timeout=API_TIMEOUT) as response:
                upstream_ok = response.status_code == 200
                response.raise_for_status()
                
                async for line in response.aiter_lines():
                    if not first_chunk_received: first_chunk_received = True
                    if line.startswith("data:"):
                        json_str = line[5:].strip()
                        if json_str == "[DONE]": break
                        try:
                            sse_data = orjson.loads(json_str)
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
            # 检查是否是Gemini模型并且有思考配置
            is_gemini_model = "gemini" in chat_input.model.lower()
            is_native_thinking_active = False
            
            if is_gemini_model and chat_input.generation_config and chat_input.generation_config.thinking_config:
                is_native_thinking_active = True
            
            # 检查是否有reasoning_effort参数
            if is_gemini_model and chat_input.custom_model_parameters and "reasoning_effort" in chat_input.custom_model_parameters:
                is_native_thinking_active = True
                
            use_custom_sep = should_apply_custom_separator_logic(chat_input, request_id, is_google_like_path=False, is_native_thinking_active=is_native_thinking_active)
            async for final_event in handle_stream_cleanup(processing_state, request_id, is_upstream_ok_final, use_custom_sep, chat_input.provider):
                yield final_event
            
            # No temp files to delete as we are reading into memory
            pass

    return StreamingResponse(event_stream_generator(), media_type="text/event-stream")