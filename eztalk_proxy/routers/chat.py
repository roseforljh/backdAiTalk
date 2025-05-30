import os
import logging
import httpx
import orjson
import asyncio
import shutil
import uuid
from typing import Optional, Dict, Any, AsyncGenerator, List

from fastapi import APIRouter, Depends, Request, HTTPException, File, UploadFile, Form
from fastapi.responses import StreamingResponse

from eztalk_proxy.models import (
    ChatRequestModel,
    SimpleTextApiMessagePy,
    PartsApiMessagePy,
    AppStreamEventPy
)
from eztalk_proxy.config import (
    COMMON_HEADERS,
    TEMP_UPLOAD_DIR,
    MAX_DOCUMENT_UPLOAD_SIZE_MB,
    # MAX_DOCUMENT_CONTENT_CHARS_FOR_PROMPT, # 这个现在主要在 utils.extract_text_from_uploaded_document 中使用
    API_TIMEOUT
)
from eztalk_proxy.utils import (
    # extract_sse_lines, # 如果 generate_non_gemini_events 中用到，则保留
    get_current_time_iso,
    orjson_dumps_bytes_wrapper,
    # strip_potentially_harmful_html_and_normalize_newlines, # 如果 generate_non_gemini_events 中用到，则保留
    extract_text_from_uploaded_document
)
from eztalk_proxy.api_helpers import prepare_openai_request # 用于非Gemini路径
# 确保 multimodal_router 正确导入
from eztalk_proxy.routers import multimodal_chat as multimodal_router
from eztalk_proxy.stream_processors import ( # 用于非Gemini路径
    process_openai_like_sse_stream,
    handle_stream_error,
    handle_stream_cleanup,
    should_apply_custom_separator_logic
)
from eztalk_proxy.web_search import perform_web_search, generate_search_context_message_content

logger = logging.getLogger("EzTalkProxy.Routers.Chat")
router = APIRouter()

async def get_http_client(request: Request) -> httpx.AsyncClient:
    # (此函数保持不变，与你之前提供的一致)
    client = getattr(request.app.state, "http_client", None)
    if client is None or (hasattr(client, 'is_closed') and client.is_closed):
        logger.error("HTTP client not available or closed in app.state.")
        raise HTTPException(status_code=503, detail="Service unavailable: HTTP client not initialized or closed.")
    return client

@router.post("/chat", response_class=StreamingResponse, summary="AI聊天完成代理", tags=["AI Proxy"])
async def chat_proxy_entrypoint(
    fastapi_request_obj: Request,
    # 确保客户端发送的表单字段名是 "chat_request_json"
    chat_request_json_str: str = Form(..., alias="chat_request_json"), # 根据错误日志，后端期望此名称
    http_client: httpx.AsyncClient = Depends(get_http_client),
    # 确保客户端发送的文件字段名是 "uploaded_documents"
    uploaded_documents: List[UploadFile] = File(default_factory=list)
):
    request_id = os.urandom(8).hex() # 更标准的 UUID 可能更好: str(uuid.uuid4())
    log_prefix = f"RID-{request_id}"
    logger.info(f"{log_prefix}: Received /chat request.") # 简化初始日志

    try:
        # 确保 chat_request_json_str 是从 Form 中获得的字段名
        logger.debug(f"{log_prefix}: Raw chat_request_json_str: {chat_request_json_str[:500]}...")
        chat_input_data = orjson.loads(chat_request_json_str) # 使用 chat_request_json_str
        chat_input = ChatRequestModel(**chat_input_data)
        logger.info(f"{log_prefix}: Parsed ChatRequestModel successfully.")
    except orjson.JSONDecodeError as e_json:
        logger.error(f"{log_prefix}: Failed to parse chat_request_json_str: {e_json}. Data: {chat_request_json_str[:500]}")
        # 为了安全，生产环境中可能不直接暴露原始数据到 detail
        raise HTTPException(status_code=400, detail=f"Invalid JSON format in 'chat_request_json' field.")
    except Exception as e_pydantic: # 更具体可以是 pydantic.ValidationError
        logger.error(f"{log_prefix}: Failed to validate ChatRequestModel from JSON: {e_pydantic}. Data: {chat_request_json_str[:500]}", exc_info=True)
        raise HTTPException(status_code=422, detail=f"Invalid data structure for chat request: {e_pydantic}")

    logger.info(
        f"{log_prefix}: Provider='{chat_input.provider}', "
        f"Model='{chat_input.model}', WebSearch={chat_input.use_web_search}, "
        f"Uploaded Documents (for text extraction/multimodal): {len(uploaded_documents)}"
    )

    processed_document_text_parts: List[str] = []
    temp_files_created_for_text_extraction: List[str] = [] # 重命名以明确用途

    # --- 文档处理：主要用于文本提取，多模态文件将直接传递给Gemini处理逻辑 ---
    # 即使是Gemini，也可能需要从非图片/视频的文档中提取文本作为补充
    files_for_gemini_multimodal: List[UploadFile] = []
    
    if uploaded_documents:
        logger.info(f"{log_prefix}: Processing {len(uploaded_documents)} uploaded items.")
        # 确保临时上传目录存在
        if not os.path.exists(TEMP_UPLOAD_DIR):
            try:
                os.makedirs(TEMP_UPLOAD_DIR)
                logger.info(f"{log_prefix}: Created temporary upload directory: {TEMP_UPLOAD_DIR}")
            except OSError as e_mkdir:
                logger.error(f"{log_prefix}: Failed to create temporary upload directory {TEMP_UPLOAD_DIR}: {e_mkdir}", exc_info=True)
                # 如果无法创建目录，可能需要提前返回错误
                # (此处省略，但生产代码应处理)

        for doc_file in uploaded_documents:
            if not doc_file.filename:
                logger.warning(f"{log_prefix}: Skipping uploaded file with no filename.")
                try: await doc_file.close()
                except Exception: pass
                continue
            
            # 文件大小检查
            file_too_large = False
            actual_size = -1
            # FastAPI的UploadFile可能需要先读取才能得到准确size，或依赖于header
            # 我们这里先信任客户端提供的，或者在读取后检查
            # 更好的方式是直接使用 UploadFile.size (如果可用且准确)
            try:
                # 尝试获取大小，如果 doc_file.file 是 SpooledTemporaryFile，size 可能可用
                if hasattr(doc_file, 'size') and doc_file.size is not None:
                     actual_size = doc_file.size
                elif hasattr(doc_file.file, 'tell') and hasattr(doc_file.file, 'seek'): # 尝试计算
                    doc_file.file.seek(0, os.SEEK_END)
                    actual_size = doc_file.file.tell()
                    doc_file.file.seek(0) # 重置指针
                
                if actual_size != -1 and actual_size > MAX_DOCUMENT_UPLOAD_SIZE_MB * 1024 * 1024:
                    logger.warning(f"{log_prefix}: Document '{doc_file.filename}' (size: {actual_size} B) exceeds max size ({MAX_DOCUMENT_UPLOAD_SIZE_MB} MB). Skipping.")
                    file_too_large = True
            except Exception as e_size:
                logger.warning(f"{log_prefix}: Could not reliably determine size of '{doc_file.filename}' before full read. Error: {e_size}. Proceeding with caution or relying on later checks.")


            if file_too_large:
                try: await doc_file.close(); logger.debug(f"{log_prefix}: Closed oversized file '{doc_file.filename}'")
                except Exception: pass
                continue

            # --- 为Gemini多模态处理保存原始文件对象 ---
            # 我们会将所有文件先尝试用于Gemini多模态，multimodal_chat.py内部会根据MIME类型筛选
            files_for_gemini_multimodal.append(doc_file)
            # 注意：如果文件在这里被消耗（读取），需要确保后续逻辑（如文本提取）能重新读取或使用副本。
            # FastAPI 的 UploadFile 通常可以多次读取，但最佳实践是只读一次或显式 seek(0)。
            # 由于我们要同时用于可能的文本提取和直接的多模态输入，这里先收集。
            # 文本提取会创建临时文件副本。

            # --- 为文本提取创建临时文件 ---
            # 只有当需要文本提取时（例如非Gemini模型，或Gemini也需要补充文本）
            # 并且文件类型适合文本提取时，才创建和处理这个临时文件
            # （这部分逻辑可以根据最终需求调整，是否所有文件都尝试提取文本）
            
            # 创建临时文件用于文本提取
            _, file_extension = os.path.splitext(doc_file.filename)
            if not file_extension: file_extension = ".tmp" # 避免没有扩展名
            safe_original_filename_part = "".join(c if c.isalnum() or c in ['.', '_', '-'] else '_' for c in doc_file.filename.rsplit('.', 1)[0])[:50]
            temp_filename = f"{request_id}_{safe_original_filename_part}_{uuid.uuid4().hex[:8]}{file_extension}"
            temp_file_path = os.path.join(TEMP_UPLOAD_DIR, temp_filename)
            
            try:
                logger.debug(f"{log_prefix}: Saving copy of '{doc_file.filename}' to '{temp_file_path}' for text extraction. MIME: {doc_file.content_type}")
                with open(temp_file_path, "wb") as buffer:
                    # 为了确保能再次读取 UploadFile，如果它已经被部分读取
                    await doc_file.seek(0) # 重置文件指针到开头
                    shutil.copyfileobj(doc_file.file, buffer)
                temp_files_created_for_text_extraction.append(temp_file_path)

                # 现在从临时文件副本中提取文本
                document_text = await extract_text_from_uploaded_document(
                    temp_file_path, # 使用临时文件路径
                    doc_file.content_type,
                    doc_file.filename # 原始文件名用于日志和可能的上下文
                )
                if document_text:
                    doc_text_part = f"\n\n--- 来自文档: {doc_file.filename} ---\n{document_text}\n--- 文档结束: {doc_file.filename} ---\n"
                    processed_document_text_parts.append(doc_text_part)
                    logger.info(f"{log_prefix}: Successfully processed text from '{doc_file.filename}' (via temp copy). Length: {len(document_text)}")
                else:
                    logger.warning(f"{log_prefix}: Failed to extract text or no text found in '{doc_file.filename}' (via temp copy).")
            except Exception as e_doc_proc:
                logger.error(f"{log_prefix}: Error processing copy of document '{doc_file.filename}' for text extraction: {e_doc_proc}", exc_info=True)
            # 注意：原始的 doc_file (来自 files_for_gemini_multimodal) 此时不应该关闭，
            # 因为 multimodal_router 可能还需要读取它。multimodal_router 内部负责关闭它传递的文件。
            # 如果 multimodal_router 不需要这个文件了，或者它处理完后，我们需要一个机制来关闭。
            # 目前的策略是，multimodal_router 会关闭它收到的 UploadFile。
    
    combined_document_text_for_prompt = "".join(processed_document_text_parts)

    # --- 路由分发 ---
    if chat_input.provider.lower() == "google" and chat_input.model.lower().startswith("gemini"):
        logger.info(f"{log_prefix}: Provider is 'google' and model '{chat_input.model}' is Gemini. Dispatching to Gemini REST API multimodal handler.")
        return StreamingResponse(
            multimodal_router.generate_gemini_rest_api_events_with_docs( # 使用你更新后的函数
                gemini_chat_input=chat_input,
                uploaded_files_for_gemini=files_for_gemini_multimodal, # <--- 传递原始文件列表
                additional_extracted_text=combined_document_text_for_prompt,
                fastapi_request_obj=fastapi_request_obj,
                http_client=http_client,
                request_id=request_id,
                temp_files_to_delete_after_stream=temp_files_created_for_text_extraction # 传递为文本提取创建的临时文件
            ),
            media_type="text/event-stream",
            headers=COMMON_HEADERS
        )
    else:
        # --- 非Gemini模型的处理逻辑 (与你之前代码基本一致) ---
        logger.info(f"{log_prefix}: Model '{chat_input.model}' with provider '{chat_input.provider}' will be handled by non-Gemini-REST path.")
        
        # (以下非Gemini处理逻辑与你之前提供的代码类似，为了简洁，我只保留框架)
        # 你需要将这部分逻辑恢复，并确保它使用 combined_document_text_for_prompt

        simple_text_messages_for_upstream: List[Dict[str, Any]] = []
        user_query_for_search = ""
        original_user_text_found = False

        # (此处粘贴你原有的处理 chat_input.messages 并构建 simple_text_messages_for_upstream 的循环)
        # 确保在这个循环中，如果 combined_document_text_for_prompt 存在，
        # 它被正确地附加到最后一个用户消息中。
        for i, msg_abstract in enumerate(chat_input.messages):
            # ... (你原有的逻辑，将 SimpleTextApiMessagePy 和 PartsApiMessagePy 转换为字典)
            # ... (并确保 combined_document_text_for_prompt 被附加) ...
            current_message_content = ""
            is_last_user_message = (i == len(chat_input.messages) - 1 and msg_abstract.role == "user")

            if isinstance(msg_abstract, SimpleTextApiMessagePy):
                current_message_content = msg_abstract.content or ""
                msg_dict = {"role": msg_abstract.role, "content": current_message_content}
                if msg_abstract.role == "user": original_user_text_found = True; user_query_for_search = current_message_content.strip()
                if hasattr(msg_abstract, 'tool_calls') and msg_abstract.tool_calls: msg_dict["tool_calls"] = [tc.model_dump(exclude_none=True) for tc in msg_abstract.tool_calls]
                if msg_abstract.role == "tool":
                    if hasattr(msg_abstract, 'tool_call_id') and msg_abstract.tool_call_id: msg_dict["tool_call_id"] = msg_abstract.tool_call_id
                    if msg_abstract.name: msg_dict["name"] = msg_abstract.name
                
                if is_last_user_message and combined_document_text_for_prompt:
                    msg_dict["content"] = (current_message_content + "\n" + combined_document_text_for_prompt).strip()
                    user_query_for_search = msg_dict["content"] # 更新用于搜索的查询
                simple_text_messages_for_upstream.append(msg_dict)

            elif isinstance(msg_abstract, PartsApiMessagePy):
                text_from_parts = " ".join([part.text for part in msg_abstract.parts if isinstance(part, PyTextContentPart) and part.text])
                current_message_content = text_from_parts.strip()
                if current_message_content:
                    if msg_abstract.role == "user": original_user_text_found = True; user_query_for_search = current_message_content
                    if is_last_user_message and combined_document_text_for_prompt:
                        current_message_content = (current_message_content + "\n" + combined_document_text_for_prompt).strip()
                        user_query_for_search = current_message_content
                    simple_text_messages_for_upstream.append({"role": msg_abstract.role, "content": current_message_content})
        # (处理结束)

        if not original_user_text_found and combined_document_text_for_prompt:
            # ... (你原有的逻辑) ...
            new_user_content_with_docs = f"请基于以下文档内容进行处理或回答：\n{combined_document_text_for_prompt}"
            simple_text_messages_for_upstream.append({"role": "user", "content": new_user_content_with_docs})
            user_query_for_search = new_user_content_with_docs


        if not simple_text_messages_for_upstream or not any(m.get("role") != "system" for m in simple_text_messages_for_upstream):
            # ... (你原有的无消息错误处理) ...
            async def no_msg_gen_err(): # (你的 no_msg_gen_err 实现)
                yield orjson_dumps_bytes_wrapper(AppStreamEventPy(type="error", message="No processable messages for this model.", timestamp=get_current_time_iso()).model_dump(by_alias=True, exclude_none=True))
                yield orjson_dumps_bytes_wrapper(AppStreamEventPy(type="finish", reason="bad_request", timestamp=get_current_time_iso()).model_dump(by_alias=True, exclude_none=True))
                for temp_file in temp_files_created_for_text_extraction: # 使用正确的临时文件列表
                    if os.path.exists(temp_file):
                        try: os.remove(temp_file)
                        except Exception as e_del_err: logger.error(f"{log_prefix}: Error deleting temp file {temp_file} in no_msg_gen_err: {e_del_err}")
            return StreamingResponse(no_msg_gen_err(), media_type="text/event-stream", headers=COMMON_HEADERS)

        # 关闭传递给非Gemini路径但未被使用的原始上传文件 (如果它们没有被Gemini路径消耗)
        # 这是一个复杂点：如果文件既可能被Gemini用，也可能被非Gemini路径的文本提取用。
        # 目前 files_for_gemini_multimodal 包含了所有文件。
        # 如果走到这里，说明不是Gemini路径，这些文件对象如果没被其他地方关闭，需要关闭。
        # 但文本提取已经对它们进行了 seek(0) 和 copyfileobj，所以原始 UploadFile 理论上仍可关闭。
        # 为了安全，非Gemini路径现在不直接使用 files_for_gemini_multimodal。
        # 文本提取后，原始 UploadFile 应该被关闭。
        # 这里我们假设在上面的文档处理循环中，如果文件没被Gemini消耗，应该被关闭。
        # 但由于我们将所有文件都收集到了 files_for_gemini_multimodal，
        # 并且 multimodal_router 会负责关闭它收到的文件，所以这里不需要额外关闭。
        # temp_files_created_for_text_extraction 是需要此路径清理的。

        return StreamingResponse(
            generate_non_gemini_events( # 确认这个函数存在并正确导入
                request_data=chat_input,
                processed_upstream_messages=simple_text_messages_for_upstream,
                user_query_for_search=user_query_for_search, # 确保这个查询是最新的
                http_client=http_client,
                fastapi_request_obj=fastapi_request_obj,
                request_id=request_id,
                temp_files_to_delete_after_stream=temp_files_created_for_text_extraction
            ),
            media_type="text/event-stream",
            headers=COMMON_HEADERS
        )

# generate_non_gemini_events 函数 (与你之前代码基本一致)
async def generate_non_gemini_events(
    request_data: ChatRequestModel,
    processed_upstream_messages: List[Dict[str, Any]],
    user_query_for_search: str,
    http_client: httpx.AsyncClient,
    fastapi_request_obj: Request,
    request_id: str,
    temp_files_to_delete_after_stream: List[str]
) -> AsyncGenerator[bytes, None]:
    # (这个函数的内容与你之前提供的应保持一致)
    # ... (粘贴你完整的 generate_non_gemini_events 函数实现) ...
    # 我将只保留框架，你需要填充具体实现
    log_prefix = f"RID-{request_id}"
    final_messages_for_llm = list(processed_upstream_messages)
    search_results_generated_this_time = False

    if request_data.use_web_search and user_query_for_search:
        # ... (Web搜索逻辑) ...
        logger.info(f"{log_prefix}: (Non-Gemini-REST) Web search initiated for query: '{user_query_for_search[:100]}'")
        yield orjson_dumps_bytes_wrapper(AppStreamEventPy(type="status_update", stage="web_search_started", timestamp=get_current_time_iso()).model_dump(by_alias=True, exclude_none=True))
        search_results_list = await perform_web_search(user_query_for_search, request_id)
        if search_results_list:
            # ... (处理并注入搜索结果的逻辑) ...
            search_context_content = generate_search_context_message_content(user_query_for_search, search_results_list)
            # 决定如何注入，例如作为新的 system message
            # 示例： final_messages_for_llm.insert(0, {"role": "system", "content": search_context_content})
            new_system_message_dict = {"role": "system", "content": search_context_content}
            last_user_idx = -1 # (你的逻辑来找到插入点)
            # ...
            if last_user_idx != -1: final_messages_for_llm.insert(last_user_idx, new_system_message_dict)
            else: final_messages_for_llm.insert(0, new_system_message_dict)
            search_results_generated_this_time = True
            yield orjson_dumps_bytes_wrapper(AppStreamEventPy(type="status_update", stage="web_search_complete_with_results", query=user_query_for_search, timestamp=get_current_time_iso()).model_dump(by_alias=True, exclude_none=True))
            yield orjson_dumps_bytes_wrapper(AppStreamEventPy(type="web_search_results", results=search_results_list, timestamp=get_current_time_iso()).model_dump(by_alias=True, exclude_none=True))
        else:
            yield orjson_dumps_bytes_wrapper(AppStreamEventPy(type="status_update", stage="web_search_complete_no_results", query=user_query_for_search, timestamp=get_current_time_iso()).model_dump(by_alias=True, exclude_none=True))


    try:
        current_api_url, current_api_headers, current_api_payload = prepare_openai_request(
            request_data=request_data,
            processed_messages=final_messages_for_llm, # 使用更新后的消息
            request_id=request_id
        )
    except Exception as e_prepare:
        # ... (错误处理) ...
        yield orjson_dumps_bytes_wrapper(AppStreamEventPy(type="error", message=f"Request preparation error: {e_prepare}", timestamp=get_current_time_iso()).model_dump(by_alias=True, exclude_none=True))
        yield orjson_dumps_bytes_wrapper(AppStreamEventPy(type="finish", reason="request_error", timestamp=get_current_time_iso()).model_dump(by_alias=True, exclude_none=True))
        # 清理临时文件
        for temp_file in temp_files_to_delete_after_stream:
            if os.path.exists(temp_file):
                try: os.remove(temp_file)
                except Exception: pass
        return

    # ... (你的 stream_processors 调用和 SSE 处理循环) ...
    # 例如:
    buffer = bytearray()
    upstream_ok_flag = False
    first_chunk_llm_received = False
    stream_proc_state: Dict[str, Any] = { # (初始化 stream_proc_state)
        "accumulated_openai_content": "", "accumulated_openai_reasoning": "",
        "openai_had_any_reasoning": False, "openai_had_any_content_or_tool_call": False,
        "openai_reasoning_finish_event_sent": False,
        "final_finish_event_sent_by_llm_reason": False,
        "final_finish_event_sent_flag_for_cleanup": False
    }

    try:
        logger.debug(f"{log_prefix}: (Non-Gemini-REST) Sending to URL: {current_api_url} ...")
        async with http_client.stream("POST", current_api_url, headers=current_api_headers, json=current_api_payload, timeout=API_TIMEOUT) as response:
            # ... (你的响应处理和 SSE 解析)
            # (粘贴你完整的响应处理循环和错误分支)
            logger.info(f"{log_prefix}: (Non-Gemini-REST) Upstream LLM response status: {response.status_code}")
            if not (200 <= response.status_code < 300):
                # ... (错误处理) ...
                return

            upstream_ok_flag = True
            async for raw_chunk_bytes in response.aiter_raw():
                # ... (你的 SSE 处理) ...
                if await fastapi_request_obj.is_disconnected(): break
                if not first_chunk_llm_received: # (处理 web_search 状态更新)
                    # ...
                    first_chunk_llm_received = True
                # (调用 process_openai_like_sse_stream 等)
                pass # 代表你原有的 SSE 处理代码
            
            if not stream_proc_state.get("final_finish_event_sent_by_llm_reason") and \
               not stream_proc_state.get("final_finish_event_sent_flag_for_cleanup"):
                logger.info(f"{log_prefix}: (Non-Gemini-REST) Stream ended, ensuring finish event.")


    except httpx.RequestError as e_req: # (你的 httpx.RequestError 处理)
        async for event_bytes in handle_stream_error(e_req, request_id, upstream_ok_flag, first_chunk_llm_received): yield event_bytes
        stream_proc_state["final_finish_event_sent_flag_for_cleanup"] = True
    except Exception as e_gen: # (你的通用 Exception 处理)
        async for event_bytes in handle_stream_error(e_gen, request_id, upstream_ok_flag, first_chunk_llm_received): yield event_bytes
        stream_proc_state["final_finish_event_sent_flag_for_cleanup"] = True
    finally: # (你的 finally 清理逻辑)
        async for event_bytes in handle_stream_cleanup(stream_proc_state, request_id, upstream_ok_flag, False, request_data.provider): yield event_bytes
        logger.info(f"{log_prefix}: Deleting {len(temp_files_to_delete_after_stream)} temporary document file(s) for non-Gemini path.")
        for temp_file in temp_files_to_delete_after_stream:
            if os.path.exists(temp_file):
                try: os.remove(temp_file)
                except Exception as e_del: logger.error(f"{log_prefix}: Error deleting temp file {temp_file}: {e_del}")