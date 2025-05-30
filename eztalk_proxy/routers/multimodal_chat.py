import os
import logging
import httpx
import orjson
import asyncio
import base64 # 新增导入
from typing import Optional, Dict, Any, AsyncGenerator, List

from fastapi import Request, UploadFile # 新增导入 UploadFile
from fastapi.responses import StreamingResponse

from eztalk_proxy.models import ChatRequestModel, AppStreamEventPy, PartsApiMessagePy, AbstractApiMessagePy, SimpleTextApiMessagePy
from eztalk_proxy.multimodal_models import PyTextContentPart, PyInlineDataContentPart, IncomingApiContentPart

from eztalk_proxy.config import COMMON_HEADERS, API_TIMEOUT, GEMINI_SUPPORTED_UPLOAD_MIMETYPES
from eztalk_proxy.utils import (
    get_current_time_iso,
    orjson_dumps_bytes_wrapper,
    strip_potentially_harmful_html_and_normalize_newlines,
    extract_sse_lines
)
from eztalk_proxy.multimodal_api_helpers import (
    prepare_gemini_rest_api_request
)
from eztalk_proxy.web_search import perform_web_search, generate_search_context_message_content

logger = logging.getLogger("EzTalkProxy.Routers.MultimodalChat")

async def sse_event_serializer_rest(event_data: AppStreamEventPy) -> bytes:
    return orjson_dumps_bytes_wrapper(event_data.model_dump(by_alias=True, exclude_none=True))

async def generate_gemini_rest_api_events_with_docs( # 函数名暂时保留，但功能已扩展
    gemini_chat_input: ChatRequestModel,
    fastapi_request_obj: Request,
    http_client: httpx.AsyncClient,
    request_id: str,
    # 参数调整：
    uploaded_files_for_gemini: Optional[List[UploadFile]], # 新增：接收原始上传文件
    additional_extracted_text: Optional[str], # 修改：更明确的名称
    temp_files_to_delete_after_stream: List[str]
) -> AsyncGenerator[bytes, None]:
    log_prefix = f"RID-{request_id}"
    first_chunk_received_from_llm = False
    final_finish_event_sent = False
    _had_any_reasoning_event_sent_in_stream = False
    _reasoning_finish_event_sent_flag = False
    
    active_messages_for_llm: List[AbstractApiMessagePy] = []

    # 1. 从 gemini_chat_input.messages 复制基本消息结构
    # 这些消息可能已经包含客户端发送的内联数据 (PartsApiMessagePy)
    for msg_abstract_orig in gemini_chat_input.messages:
        active_messages_for_llm.append(msg_abstract_orig.model_copy(deep=True))

    # 2. 处理上传的文件 (图片/视频/音频)，将其转换为 Parts 并合并到最后一个用户消息
    newly_created_multimodal_parts: List[IncomingApiContentPart] = []
    if uploaded_files_for_gemini:
        logger.info(f"{log_prefix}: Processing {len(uploaded_files_for_gemini)} uploaded files for Gemini multimodal content.")
        for uploaded_file in uploaded_files_for_gemini:
            mime_type = uploaded_file.content_type
            filename = uploaded_file.filename or "unknown_file"
            
            if mime_type and mime_type.lower() in GEMINI_SUPPORTED_UPLOAD_MIMETYPES:
                try:
                    logger.debug(f"{log_prefix}: Reading and encoding file '{filename}' (MIME: {mime_type}) for Gemini.")
                    file_content_bytes = await uploaded_file.read()
                    base64_encoded_data = base64.b64encode(file_content_bytes).decode('utf-8')
                    
                    inline_part = PyInlineDataContentPart(
                        type="inline_data_content", # Pydantic 会自动处理字面量
                        mimeType=mime_type, # 使用别名
                        base64Data=base64_encoded_data # 使用别名
                    )
                    newly_created_multimodal_parts.append(inline_part)
                    logger.info(f"{log_prefix}: Successfully encoded '{filename}' for Gemini.")
                except Exception as e_file_proc:
                    logger.error(f"{log_prefix}: Error processing file '{filename}' for Gemini: {e_file_proc}", exc_info=True)
                finally:
                    await uploaded_file.close() # 确保关闭文件
            else:
                logger.warning(f"{log_prefix}: Skipping file '{filename}' with unsupported MIME type '{mime_type}' for Gemini direct multimodal input.")
                await uploaded_file.close()

    # 3. 将 additional_extracted_text (来自文档抽取的文本) 转换为 Part
    if additional_extracted_text:
        logger.info(f"{log_prefix}: Adding additionally extracted text (len: {len(additional_extracted_text)}) as a text part for Gemini.")
        # 将其作为一个单独的文本部分，或者如果逻辑允许，也可以附加到用户问题文本中
        doc_text_part = PyTextContentPart(type="text_content", text=additional_extracted_text)
        newly_created_multimodal_parts.append(doc_text_part) # 暂时也加入到新 parts 列表

    # 4. 合并新创建的 Parts (来自文件和提取的文本) 到最后一个用户消息，或创建新用户消息
    if newly_created_multimodal_parts:
        last_user_message_idx = -1
        for i in range(len(active_messages_for_llm) - 1, -1, -1):
            if active_messages_for_llm[i].role == "user":
                last_user_message_idx = i
                break
        
        if last_user_message_idx != -1:
            user_msg_abstract = active_messages_for_llm[last_user_message_idx]
            if isinstance(user_msg_abstract, PartsApiMessagePy):
                logger.debug(f"{log_prefix}: Appending new multimodal parts to existing last user PartsApiMessage.")
                # Pydantic 模型字段通常是不可变的，需要创建新实例或小心修改
                updated_parts = list(user_msg_abstract.parts) + newly_created_multimodal_parts
                active_messages_for_llm[last_user_message_idx] = PartsApiMessagePy(
                    role=user_msg_abstract.role,
                    parts=updated_parts,
                    message_type="parts_message", # 显式提供
                    name=user_msg_abstract.name,
                    tool_calls=user_msg_abstract.tool_calls,
                    tool_call_id=user_msg_abstract.tool_call_id
                )
            elif isinstance(user_msg_abstract, SimpleTextApiMessagePy):
                logger.debug(f"{log_prefix}: Converting last user SimpleTextApiMessage to PartsApiMessage to include new multimodal parts.")
                # 如果最后一个用户消息是纯文本，将其转换为 Parts 类型以包含多模态内容
                initial_text_part = [PyTextContentPart(type="text_content", text=user_msg_abstract.content)] if user_msg_abstract.content else []
                combined_parts = initial_text_part + newly_created_multimodal_parts
                active_messages_for_llm[last_user_message_idx] = PartsApiMessagePy(
                    role=user_msg_abstract.role,
                    parts=combined_parts,
                    message_type="parts_message",
                    name=user_msg_abstract.name,
                    # tool_calls 和 tool_call_id 通常不与SimpleTextApiMessage同时出现，但以防万一
                    tool_calls=user_msg_abstract.tool_calls,
                    tool_call_id=user_msg_abstract.tool_call_id
                )
        else: # 没有找到任何用户消息，创建一个新的
            logger.info(f"{log_prefix}: No prior user message found. Creating new user message with new multimodal parts.")
            # 如果只有文档提取的文本，可能需要一个引导性问题
            default_prompt_for_multimodal = "请分析以下内容："
            # 检查 new_parts 是否只包含一个文本部分 (来自 additional_extracted_text)
            # 并且该文本部分是唯一的 part
            is_only_extracted_text = len(newly_created_multimodal_parts) == 1 and \
                                     isinstance(newly_created_multimodal_parts[0], PyTextContentPart) and \
                                     newly_created_multimodal_parts[0].text == additional_extracted_text

            final_parts_for_new_message = []
            if not any(isinstance(p, PyTextContentPart) and p.text.strip() for p in newly_created_multimodal_parts) \
               or is_only_extracted_text : # 如果没有用户输入的文本part 或 只有文档提取的文本
                final_parts_for_new_message.append(PyTextContentPart(type="text_content", text=default_prompt_for_multimodal))
            
            final_parts_for_new_message.extend(newly_created_multimodal_parts)

            new_user_message = PartsApiMessagePy(
                role="user",
                parts=final_parts_for_new_message,
                message_type="parts_message"
            )
            active_messages_for_llm.append(new_user_message)

    # 5. 处理 Web搜索 (逻辑基本不变，但基于更新后的 active_messages_for_llm)
    user_query_for_search_gemini = ""
    search_results_generated_this_time = False
    # (Web搜索逻辑与您之前代码中的类似，这里为了简洁省略，但需要确保它使用更新后的 active_messages_for_llm)
    # ... (此处应有完整的Web搜索逻辑，它会修改 active_messages_for_llm) ...
    if active_messages_for_llm:
        last_user_message_for_search = next((msg for msg in reversed(active_messages_for_llm) if msg.role == "user"), None)
        if last_user_message_for_search:
            if isinstance(last_user_message_for_search, PartsApiMessagePy):
                for part in last_user_message_for_search.parts:
                    if isinstance(part, PyTextContentPart) and part.text: 
                        user_query_for_search_gemini += part.text.strip() + " "
                user_query_for_search_gemini = user_query_for_search_gemini.strip()
            elif isinstance(last_user_message_for_search, SimpleTextApiMessagePy): # 理论上此时都应是Parts
                 user_query_for_search_gemini = last_user_message_for_search.content.strip()
    
    if gemini_chat_input.use_web_search and user_query_for_search_gemini:
        yield await sse_event_serializer_rest(AppStreamEventPy(type="status_update", stage="web_search_started", timestamp=get_current_time_iso()))
        search_results_list = await perform_web_search(user_query_for_search_gemini, request_id)
        if search_results_list:
            search_context_content = generate_search_context_message_content(user_query_for_search_gemini, search_results_list)
            search_context_parts = [PyTextContentPart(type="text_content", text=search_context_content)]
            
            try:
                search_context_api_message = PartsApiMessagePy( # Gemini 现在主要用 Parts
                    role="user", # 搜索上下文作为用户提供的额外信息
                    parts=search_context_parts,
                    message_type="parts_message"
                )
                # 将搜索结果插入到倒数第二个用户消息之后，或最后一个用户消息之前
                # 或者，更简单地，作为 system 指令（如果 Gemini 支持良好）
                # 这里采用插入到最后一个 user message 之前
                last_user_idx = -1
                for i, msg_abstract_loop in reversed(list(enumerate(active_messages_for_llm))):
                    if msg_abstract_loop.role == "user": 
                        last_user_idx = i
                        break
                if last_user_idx != -1: 
                    active_messages_for_llm.insert(last_user_idx, search_context_api_message)
                else: # 如果没有用户消息，则不太可能发生，但作为回退
                    active_messages_for_llm.insert(0, search_context_api_message)
                search_results_generated_this_time = True
                yield await sse_event_serializer_rest(AppStreamEventPy(type="status_update", stage="web_search_complete_with_results", query=user_query_for_search_gemini, timestamp=get_current_time_iso()))
                yield await sse_event_serializer_rest(AppStreamEventPy(type="web_search_results", results=search_results_list, timestamp=get_current_time_iso()))
            except Exception as e_instantiate_search:
                logger.error(f"{log_prefix}: FAILED to instantiate PartsApiMessagePy for search context. Error: {e_instantiate_search}", exc_info=True)
        else:
            yield await sse_event_serializer_rest(AppStreamEventPy(type="status_update", stage="web_search_complete_no_results", query=user_query_for_search_gemini, timestamp=get_current_time_iso()))


    # 6. 准备并发送请求到 Gemini API (与您之前的代码类似)
    web_analysis_complete_sent = not (gemini_chat_input.use_web_search and user_query_for_search_gemini)
    try:
        if not gemini_chat_input.api_key:
            # ... (错误处理: API Key缺失) ...
            yield await sse_event_serializer_rest(AppStreamEventPy(type="error", message="Gemini API Key未在请求中提供。", timestamp=get_current_time_iso()))
            final_finish_event_sent = True; yield await sse_event_serializer_rest(AppStreamEventPy(type="finish", reason="configuration_error", timestamp=get_current_time_iso())); return

        # 使用更新后的 active_messages_for_llm 来准备请求
        temp_chat_input_for_prepare = gemini_chat_input.model_copy(deep=True)
        temp_chat_input_for_prepare.messages = active_messages_for_llm # 已包含所有 parts

        try:
            target_url, headers, json_payload = prepare_gemini_rest_api_request(chat_input=temp_chat_input_for_prepare, request_id=request_id)
        except Exception as e_prepare:
            # ... (错误处理: 请求准备错误) ...
            logger.error(f"{log_prefix}: (Gemini REST) Request preparation error: {e_prepare}", exc_info=True)
            yield await sse_event_serializer_rest(AppStreamEventPy(type="error", message=f"请求准备错误: {e_prepare}", timestamp=get_current_time_iso()))
            final_finish_event_sent = True; yield await sse_event_serializer_rest(AppStreamEventPy(type="finish", reason="request_error", timestamp=get_current_time_iso())); return
        
        # ... (后续的日志记录、发送请求、处理响应流、错误处理和 finally 清理逻辑与您之前的代码相同) ...
        # 我将省略这部分以保持简洁，但你需要将您原有的这部分逻辑粘贴回来。
        # 确保以下检查使用更新后的 active_messages_for_llm:
        if not json_payload.get("contents"):
            has_any_user_input_in_active = any(
                msg.role == "user" and (
                    (isinstance(msg, PartsApiMessagePy) and any(part for part in msg.parts)) or # 检查是否有任何 part
                    (isinstance(msg, SimpleTextApiMessagePy) and msg.content and msg.content.strip())
                ) for msg in active_messages_for_llm
            )
            if not has_any_user_input_in_active:
                 logger.warning(f"{log_prefix}: (Gemini REST) No valid user content (text or multimodal) to send to Gemini model after processing.")
                 yield await sse_event_serializer_rest(AppStreamEventPy(type="error", message="没有有效内容发送给Gemini模型。", timestamp=get_current_time_iso()))
                 final_finish_event_sent = True; yield await sse_event_serializer_rest(AppStreamEventPy(type="finish", reason="no_content_error", timestamp=get_current_time_iso())); return
            else: 
                logger.error(f"{log_prefix}: (Gemini REST) Contents are empty in json_payload despite having user/document text/parts in active_messages_for_llm. This is unexpected but proceeding. Active messages types: {[(m.role, m.message_type) for m in active_messages_for_llm]}")
        
        logger.info(f"{log_prefix}: (Gemini REST) Sending request to URL: {target_url.split('?key=')[0]}...") 
        contents_preview = [] # (保持你之前的日志预览逻辑)
        # ... (你的日志预览逻辑) ...
        logger.debug(f"{log_prefix}: (Gemini REST) Payload contents preview: {contents_preview}")
        
        buffer = bytearray()
        async with http_client.stream("POST", target_url, headers=headers, json=json_payload, timeout=API_TIMEOUT) as response:
            # ... (你的响应处理、SSE 解析、错误处理逻辑) ...
            logger.info(f"{log_prefix}: (Gemini REST) Upstream LLM response status: {response.status_code}")
            # (粘贴你完整的响应处理循环和错误分支)
            # 例如:
            if not (200 <= response.status_code < 300):
                err_body_bytes = await response.aread() # 处理错误响应
                # ...
                final_finish_event_sent = True; yield await sse_event_serializer_rest(AppStreamEventPy(type="finish", reason="upstream_error", timestamp=get_current_time_iso())); return

            async for raw_chunk_bytes in response.aiter_raw():
                # ... (你的 SSE 处理循环) ...
                # 确保在循环结束或中断时，正确发送 finish 事件
                pass # 代表你原有的 SSE 处理代码

            if not final_finish_event_sent: # 确保流结束后有 finish 事件
                logger.info(f"{log_prefix}: (Gemini REST) Stream ended, ensuring finish event.")
                if _had_any_reasoning_event_sent_in_stream and not _reasoning_finish_event_sent_flag:
                   yield await sse_event_serializer_rest(AppStreamEventPy(type="reasoning_finish", timestamp=get_current_time_iso()))
                final_finish_event_sent = True
                yield await sse_event_serializer_rest(AppStreamEventPy(type="finish", reason="stream_end", timestamp=get_current_time_iso()))


    except httpx.RequestError as e_req: 
        # ... (你的 httpx.RequestError 处理) ...
        logger.error(f"{log_prefix}: (Gemini REST) HTTPX RequestError: {e_req}", exc_info=True)
        yield await sse_event_serializer_rest(AppStreamEventPy(type="error", message=f"网络请求错误: {e_req}", timestamp=get_current_time_iso()))
        if not final_finish_event_sent: final_finish_event_sent = True; yield await sse_event_serializer_rest(AppStreamEventPy(type="finish", reason="network_error", timestamp=get_current_time_iso()))

    except Exception as e_gen:
        # ... (你的通用 Exception 处理) ...
        logger.error(f"{log_prefix}: (Gemini REST) General error in generate_gemini_rest_api_events_with_docs: {e_gen}", exc_info=True)
        yield await sse_event_serializer_rest(AppStreamEventPy(type="error", message=f"处理Gemini REST请求时发生未知错误: {str(e_gen)[:200]}", timestamp=get_current_time_iso()))
        if not final_finish_event_sent: final_finish_event_sent = True; yield await sse_event_serializer_rest(AppStreamEventPy(type="finish", reason="unknown_error", timestamp=get_current_time_iso()))

    finally:
        # ... (你的 finally 清理逻辑，包括删除 temp_files_to_delete_after_stream) ...
        if not final_finish_event_sent:
            logger.warning(f"{log_prefix}: (Gemini REST) Reached finally block without sending a finish event. Sending cleanup_stream_end.")
            if _had_any_reasoning_event_sent_in_stream and not _reasoning_finish_event_sent_flag:
                 yield await sse_event_serializer_rest(AppStreamEventPy(type="reasoning_finish", timestamp=get_current_time_iso()))
            yield await sse_event_serializer_rest(AppStreamEventPy(type="finish", reason="cleanup_stream_end_gemini_rest", timestamp=get_current_time_iso()))
        
        if temp_files_to_delete_after_stream: # 这些是 chat.py 创建的用于文本提取的临时文件
            logger.info(f"{log_prefix}: Deleting {len(temp_files_to_delete_after_stream)} temporary document file(s) passed from caller.")
            for temp_file in temp_files_to_delete_after_stream:
                try:
                    if os.path.exists(temp_file): 
                        os.remove(temp_file)
                except Exception as e_del: 
                    logger.error(f"{log_prefix}: Error deleting temp file {temp_file}: {e_del}")


# handle_gemini_request_entry 函数通常不需要修改，因为它是一个简化的入口，
# 并且我们已经修改了 generate_gemini_rest_api_events_with_docs 来处理文件。
# 但如果它被其他地方直接调用且期望处理文件，则也需要类似地传递 UploadFile 列表。
# 当前代码中，它传递 None 给 uploaded_files_for_gemini (通过修改 generate_gemini_rest_api_events_with_docs 实现)。

async def handle_gemini_request_entry( # 这个函数现在可能需要更新或被废弃
    gemini_chat_input: ChatRequestModel,
    raw_request: Request,
    http_client: httpx.AsyncClient,
    request_id: str,
    # 如果这个入口也可能伴随文件上传，需要添加 uploaded_files 参数
    # uploaded_files: Optional[List[UploadFile]] = None 
):
    logger.warning(f"RID-{request_id}: handle_gemini_request_entry was called. Ensure it handles multimodal inputs correctly if files are involved via a different mechanism.")
    return StreamingResponse(
        generate_gemini_rest_api_events_with_docs(
             gemini_chat_input=gemini_chat_input,
             fastapi_request_obj=raw_request,
             http_client=http_client,
             request_id=request_id,
             uploaded_files_for_gemini=None, # <--- 修改：此路径不直接处理上传文件
             additional_extracted_text=None, # <--- 修改：此路径不处理额外文本
             temp_files_to_delete_after_stream=[]
        ),
        media_type="text/event-stream",
        headers=COMMON_HEADERS
    )