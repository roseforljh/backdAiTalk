# eztalk_proxy/routers/chat.py
import os
import logging
import httpx
import orjson
from typing import Optional, Dict, Any, AsyncGenerator, List

from fastapi import APIRouter, Depends, Request, HTTPException
from fastapi.responses import StreamingResponse

# 使用绝对导入
from eztalk_proxy.models import (
    ChatRequestModel,
    SimpleTextApiMessagePy, # Assuming this is for non-Gemini or non-Google-provider Gemini
    PartsApiMessagePy,      # Assuming this is for Google-provider Gemini
    AppStreamEventPy
    # AbstractApiMessagePy, # If ChatRequestModel.messages is List[AbstractApiMessagePy]
)
from eztalk_proxy.config import COMMON_HEADERS # GOOGLE_API_BASE_URL might not be needed here directly
from eztalk_proxy.utils import (
    extract_sse_lines,
    get_current_time_iso,
    orjson_dumps_bytes_wrapper,
    strip_potentially_harmful_html_and_normalize_newlines
)
# api_helpers for OpenAI compatible paths
from eztalk_proxy.api_helpers import prepare_openai_request
# multimodal_router for Google provider Gemini via REST API
from eztalk_proxy.routers import multimodal_chat as multimodal_router
# stream_processors for OpenAI compatible paths
from eztalk_proxy.stream_processors import (
    process_openai_like_sse_stream, # Or your specific OpenAI response processor
    handle_stream_error,
    handle_stream_cleanup,
    should_apply_custom_separator_logic
)
# Web search (if applicable to non-Gemini paths)
from eztalk_proxy.web_search import perform_web_search, generate_search_context_message_content

logger = logging.getLogger("EzTalkProxy.Routers.Chat")
router = APIRouter()

# --- HTTP 客户端依赖注入 ---
async def get_http_client(request: Request) -> httpx.AsyncClient:
    client = getattr(request.app.state, "http_client", None)
    if client is None or (hasattr(client, 'is_closed') and client.is_closed):
        logger.error("HTTP client not available or closed in app.state.")
        raise HTTPException(status_code=503, detail="Service unavailable: HTTP client not initialized or closed.")
    return client

# --- 主聊天代理端点 ---
@router.post("/chat", response_class=StreamingResponse, summary="AI聊天完成代理", tags=["AI Proxy"])
async def chat_proxy_entrypoint(
    chat_input: ChatRequestModel, # Uses Pydantic model with discriminated union for messages
    fastapi_request_obj: Request,
    http_client: httpx.AsyncClient = Depends(get_http_client)
):
    request_id = os.urandom(8).hex()
    log_prefix = f"RID-{request_id}"

    logger.info(
        f"{log_prefix}: Received /chat request: Provider='{chat_input.provider}', "
        f"Model='{chat_input.model}', WebSearch={chat_input.use_web_search}"
    )

    # --- Core Dispatch Logic ---
    # Only if provider is "google" AND model name starts with "gemini", use the Gemini REST API multimodal handler
    if chat_input.provider.lower() == "google" and chat_input.model.lower().startswith("gemini"):
        logger.info(f"{log_prefix}: Provider is 'google' and model '{chat_input.model}' is Gemini. Dispatching to Gemini REST API multimodal handler.")
        # Frontend should send PartsApiMessagePy for this route
        return await multimodal_router.handle_gemini_request_entry(
            gemini_chat_input=chat_input,
            raw_request=fastapi_request_obj,
            http_client=http_client, # multimodal_router might use this if it makes direct calls (though current one doesn't)
            request_id=request_id
        )
    else:
        # All other cases (non-Gemini models, or Gemini models via non-"google" providers)
        # will be handled by a generic path, typically OpenAI compatible.
        logger.info(f"{log_prefix}: Model '{chat_input.model}' with provider '{chat_input.provider}' will be handled by non-Gemini-REST path (e.g., OpenAI compatible).")
        
        simple_text_messages_for_upstream: List[Dict[str, Any]] = []
        user_query_for_search = ""

        for i, msg_abstract in enumerate(chat_input.messages):
            # Expect SimpleTextApiMessagePy for this path from frontend
            if isinstance(msg_abstract, SimpleTextApiMessagePy):
                msg_dict = {"role": msg_abstract.role, "content": msg_abstract.content or ""}
                if msg_abstract.role == "user" and msg_abstract.content:
                    user_query_for_search = msg_abstract.content.strip()
                
                # Handle tool calls if SimpleTextApiMessagePy supports them
                if hasattr(msg_abstract, 'tool_calls') and msg_abstract.tool_calls:
                    msg_dict["tool_calls"] = [tc.model_dump(exclude_none=True) for tc in msg_abstract.tool_calls]
                if msg_abstract.role == "tool":
                    if hasattr(msg_abstract, 'tool_call_id') and msg_abstract.tool_call_id: msg_dict["tool_call_id"] = msg_abstract.tool_call_id
                    if msg_abstract.name: msg_dict["name"] = msg_abstract.name # OpenAI tool role needs name
                
                simple_text_messages_for_upstream.append(msg_dict)

            elif isinstance(msg_abstract, PartsApiMessagePy):
                # Fallback: If PartsApiMessagePy is received on this path, try to extract text
                logger.warning(f"{log_prefix}: Non-Gemini-REST path received PartsApiMessage for model '{chat_input.model}' (provider: {chat_input.provider}). Extracting text.")
                text_from_parts = ""
                for part_model in msg_abstract.parts: # part_model is PyTextContentPart etc.
                    if hasattr(part_model, 'text') and isinstance(part_model.text, str):
                         text_from_parts += part_model.text + " "
                
                if text_from_parts.strip():
                    simple_text_messages_for_upstream.append({"role": msg_abstract.role, "content": text_from_parts.strip()})
                    if msg_abstract.role == "user":
                        user_query_for_search = text_from_parts.strip()
                else:
                    logger.warning(f"{log_prefix}: Could not extract text from PartsApiMessage for '{chat_input.model}' on non-Gemini-REST path. Skipping.")
            else:
                logger.error(f"{log_prefix}: Unknown message type '{type(msg_abstract)}' in messages list for non-Gemini-REST path. Skipping.")

        if not simple_text_messages_for_upstream or not any(m.get("role") != "system" for m in simple_text_messages_for_upstream):
            if not any(m.get("role") == "system" and m.get("content") for m in simple_text_messages_for_upstream):
                logger.warning(f"{log_prefix}: No processable non-system messages for '{chat_input.model}' on non-Gemini-REST path.")
                async def no_msg_gen_err():
                    yield orjson_dumps_bytes_wrapper(AppStreamEventPy(type="error", message="No processable messages for this model.", timestamp=get_current_time_iso()).model_dump(by_alias=True, exclude_none=True))
                    yield orjson_dumps_bytes_wrapper(AppStreamEventPy(type="finish", reason="bad_request", timestamp=get_current_time_iso()).model_dump(by_alias=True, exclude_none=True))
                return StreamingResponse(no_msg_gen_err(), media_type="text/event-stream", headers=COMMON_HEADERS)
        
        return StreamingResponse(
            generate_non_gemini_events(
                chat_input,
                simple_text_messages_for_upstream,
                user_query_for_search,
                http_client,
                fastapi_request_obj,
                request_id
            ),
            media_type="text/event-stream",
            headers=COMMON_HEADERS
        )

async def generate_non_gemini_events(
    request_data: ChatRequestModel, # Original request data for API key, model, temp, etc.
    processed_upstream_messages: List[Dict[str, Any]], # Messages formatted for the upstream API
    user_query_for_search: str, # Last user query for web search
    http_client: httpx.AsyncClient,
    fastapi_request_obj: Request,
    request_id: str
) -> AsyncGenerator[bytes, None]:
    log_prefix = f"RID-{request_id}"
    final_messages_for_llm = list(processed_upstream_messages) # Create a mutable copy
    search_results_generated_this_time = False

    # --- Web Search (if enabled for this non-Gemini path) ---
    if request_data.use_web_search and user_query_for_search:
        logger.info(f"{log_prefix}: (Non-Gemini-REST) Web search initiated for query: '{user_query_for_search[:100]}'")
        yield orjson_dumps_bytes_wrapper(AppStreamEventPy(type="status_update", stage="web_search_started", timestamp=get_current_time_iso()).model_dump(by_alias=True, exclude_none=True))
        
        search_results_list = await perform_web_search(user_query_for_search, request_id)
        
        if search_results_list:
            search_context_content = generate_search_context_message_content(user_query_for_search, search_results_list)
            new_system_message_dict = {"role": "system", "content": search_context_content}
            
            # Intelligent insertion of system message (e.g., before last user message or at start)
            last_user_idx = -1
            for i, msg in reversed(list(enumerate(final_messages_for_llm))):
                if msg.get("role") == "user":
                    last_user_idx = i
                    break
            if last_user_idx != -1:
                final_messages_for_llm.insert(last_user_idx, new_system_message_dict)
            else: # If no user message (unlikely but safe)
                final_messages_for_llm.insert(0, new_system_message_dict)
            
            search_results_generated_this_time = True
            logger.info(f"{log_prefix}: (Non-Gemini-REST) Web search context injected.")
            yield orjson_dumps_bytes_wrapper(AppStreamEventPy(type="status_update", stage="web_search_complete_with_results", query=user_query_for_search, timestamp=get_current_time_iso()).model_dump(by_alias=True, exclude_none=True))
            yield orjson_dumps_bytes_wrapper(AppStreamEventPy(type="web_search_results", results=search_results_list, timestamp=get_current_time_iso()).model_dump(by_alias=True, exclude_none=True))
        else:
            logger.info(f"{log_prefix}: (Non-Gemini-REST) Web search yielded no results for query '{user_query_for_search[:100]}'.")
            yield orjson_dumps_bytes_wrapper(AppStreamEventPy(type="status_update", stage="web_search_complete_no_results", query=user_query_for_search, timestamp=get_current_time_iso()).model_dump(by_alias=True, exclude_none=True))
        
        
    # --- Prepare API request (e.g., for OpenAI compatible endpoints) ---
    try:
        current_api_url, current_api_headers, current_api_payload = prepare_openai_request(
            request_data=request_data,
            processed_messages=final_messages_for_llm,
            request_id=request_id
        )
    except Exception as e_prepare:
        logger.error(f"{log_prefix}: Error preparing non-Gemini-REST request: {e_prepare}", exc_info=True)
        yield orjson_dumps_bytes_wrapper(AppStreamEventPy(type="error", message=f"Request preparation error: {e_prepare}", timestamp=get_current_time_iso()).model_dump(by_alias=True, exclude_none=True))
        yield orjson_dumps_bytes_wrapper(AppStreamEventPy(type="finish", reason="request_error", timestamp=get_current_time_iso()).model_dump(by_alias=True, exclude_none=True))
        return

    # --- Stream request and process SSE ---
    buffer = bytearray()
    upstream_ok_flag = False
    first_chunk_llm_received = False
    stream_proc_state: Dict[str, Any] = {
        "accumulated_openai_content": "", "accumulated_openai_reasoning": "",
        "openai_had_any_reasoning": False, "openai_had_any_content_or_tool_call": False,
        "openai_reasoning_finish_event_sent": False,
        "final_finish_event_sent_by_llm_reason": False,
        "final_finish_event_sent_flag_for_cleanup": False
    }
    # Determine if custom separator logic should be used (based on your stream_processors.py)
    # For a generic OpenAI path, this is usually False unless specific models require it.
    use_old_custom_separator_branch_flag = should_apply_custom_separator_logic(
        request_data, request_id,
        False, # is_google_like_path (this is the non-Google path)
        False  # is_native_thinking_active (not applicable here)
    )

    try:
        logger.debug(f"{log_prefix}: (Non-Gemini-REST) Sending to URL: {current_api_url}. Payload (first 500 of messages): {str(current_api_payload.get('messages',[]))[:500]}")
        async with http_client.stream(
            "POST", current_api_url,
            headers=current_api_headers,
            json=current_api_payload,
            timeout=300.0
        ) as response:
            logger.info(f"{log_prefix}: (Non-Gemini-REST) Upstream LLM response status: {response.status_code}")
            if not (200 <= response.status_code < 300):
                err_body_bytes = await response.aread()
                err_text = err_body_bytes.decode("utf-8", errors="replace")
                logger.error(f"{log_prefix}: (Non-Gemini-REST) Upstream LLM error {response.status_code}: {err_text[:1000]}")
                yield orjson_dumps_bytes_wrapper(AppStreamEventPy(type="error", message=f"LLM API Error: {err_text[:200]}", upstream_status=response.status_code, timestamp=get_current_time_iso()).model_dump(by_alias=True, exclude_none=True))
                upstream_ok_flag = False
                # Cleanup will send the finish event
                return

            upstream_ok_flag = True
            async for raw_chunk_bytes in response.aiter_raw():
                if await fastapi_request_obj.is_disconnected():
                    logger.info(f"{log_prefix}: (Non-Gemini-REST) Client disconnected.")
                    break
                
                if not first_chunk_llm_received:
                    if request_data.use_web_search and user_query_for_search: # Send web_analysis_complete after first LLM chunk
                        stage_after_search = "web_analysis_complete" if search_results_generated_this_time else "web_analysis_skipped_no_results"
                        yield orjson_dumps_bytes_wrapper(AppStreamEventPy(type="status_update", stage=stage_after_search, timestamp=get_current_time_iso()).model_dump(by_alias=True, exclude_none=True))
                    first_chunk_llm_received = True
                
                buffer.extend(raw_chunk_bytes)
                sse_lines, buffer = extract_sse_lines(buffer)

                for sse_line_bytes in sse_lines:
                    if not sse_line_bytes.strip(): continue
                    sse_data_bytes = b""
                    if sse_line_bytes.startswith(b"data: "):
                        sse_data_bytes = sse_line_bytes[len(b"data: "):].strip()
                    if not sse_data_bytes: continue
                    
                    logger.debug(f"{log_prefix}: (Non-Gemini-REST) Raw SSE: {sse_data_bytes!r}")

                    if sse_data_bytes == b"[DONE]": # OpenAI [DONE] signal
                        logger.info(f"{log_prefix}: Received [DONE] from non-Gemini (OpenAI-like) endpoint.")
                        stream_proc_state["final_finish_reason_from_llm"] = stream_proc_state.get("final_finish_reason_from_llm","stop")
                        stream_proc_state["final_finish_event_sent_by_llm_reason"] = True
                        break 

                    try:
                        parsed_sse_data = orjson.loads(sse_data_bytes)
                        logger.debug(f"{log_prefix}: (Non-Gemini-REST) Parsed SSE: {parsed_sse_data}")
                        async for event_dict in process_openai_like_sse_stream(parsed_sse_data, stream_proc_state, request_id):
                            yield orjson_dumps_bytes_wrapper(AppStreamEventPy(**event_dict).model_dump(by_alias=True, exclude_none=True))
                            if event_dict.get("type") == "finish" or stream_proc_state.get("final_finish_event_sent_by_llm_reason"):
                                return 
                    except orjson.JSONDecodeError:
                        logger.warning(f"{log_prefix}: Failed to parse non-Gemini SSE JSON: {sse_data_bytes.decode(errors='ignore')[:100]}")
                    except Exception as e_proc_sse:
                        logger.error(f"{log_prefix}: Error processing non-Gemini SSE line: {sse_data_bytes.decode(errors='ignore')[:100]}, error: {e_proc_sse}", exc_info=True)
                
                if stream_proc_state.get("final_finish_event_sent_by_llm_reason"):
                    break
            
            # If loop finishes and LLM hasn't sent a finish reason (e.g. stream just ends)
            if not stream_proc_state.get("final_finish_event_sent_by_llm_reason") and not stream_proc_state.get("final_finish_event_sent_flag_for_cleanup"):
                logger.info(f"{log_prefix}: (Non-Gemini-REST) Stream ended without explicit LLM finish signal.")
                # The cleanup function will handle sending the final "finish" event.

    except httpx.RequestError as e_req:
        logger.error(f"{log_prefix}: httpx.RequestError for non-Gemini model '{request_data.model}': {e_req}", exc_info=True)
        async for event_bytes in handle_stream_error(e_req, request_id, upstream_ok_flag, first_chunk_llm_received): yield event_bytes
        stream_proc_state["final_finish_event_sent_flag_for_cleanup"] = True # Mark that error handler sent finish
    except Exception as e_gen:
        logger.error(f"{log_prefix}: Generic error in non-Gemini stream for model '{request_data.model}': {e_gen}", exc_info=True)
        async for event_bytes in handle_stream_error(e_gen, request_id, upstream_ok_flag, first_chunk_llm_received): yield event_bytes
        stream_proc_state["final_finish_event_sent_flag_for_cleanup"] = True # Mark that error handler sent finish
    finally:
        logger.info(f"{log_prefix}: Cleaning up non-Gemini stream for model '{request_data.model}'.")
        # Pass the provider from request_data for logging in cleanup
        async for event_bytes in handle_stream_cleanup(
            stream_proc_state, request_id, upstream_ok_flag,
            use_old_custom_separator_branch_flag,
            request_data.provider # Pass provider for logging/logic in cleanup
        ):
            yield event_bytes