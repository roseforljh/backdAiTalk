# eztalk_proxy/routers/multimodal_chat.py
import os
import logging
import httpx # httpx 客户端实例会从 main.py 传递过来
import orjson
import base64 # 用于解码前端传来的 base64 图片数据
from typing import Optional, Dict, Any, AsyncGenerator, List
import asyncio # 用于 asyncio.to_thread

from fastapi import APIRouter, Depends, Request, HTTPException # Request 用于检查客户端断开
from fastapi.responses import StreamingResponse

# 导入 Pydantic 模型
from ..models import (
    ChatRequestModel, PartsApiMessagePy, AppStreamEventPy,
    IncomingApiContentPart, TextPartWrapper, FileUriPartWrapper, InlineDataPartWrapper
)
from ..config import COMMON_HEADERS, GOOGLE_APPLICATION_CREDENTIALS_STRING # 假设凭证在config中管理
from ..utils import (
    extract_sse_lines, get_current_time_iso,
    orjson_dumps_bytes_wrapper,
    strip_potentially_harmful_html_and_normalize_newlines
)
# 导入可能的 Web 搜索辅助函数 (如果Gemini也需要Web搜索)
from ..web_search import perform_web_search, generate_search_context_message_content

# 导入 Vertex AI SDK
import vertexai
from vertexai.generative_models import (
    GenerativeModel, Part, Content, FinishReason, Tool
)
# from vertexai.generative_models import HarmCategory, HarmBlockThreshold # 按需导入安全设置
from vertexai.preview.generative_models import GenerationConfig as VertexGenerationConfig # 使用 preview 下的，如果需要 thoughts 等功能
from google.auth import credentials as auth_credentials # 用于从字符串创建凭证
from google.oauth2 import service_account

logger = logging.getLogger("EzTalkProxy.Routers.MultimodalChat")
# router = APIRouter() # 这个模块不直接定义路由，而是被 chat.py 调用

# --- Vertex AI 初始化相关的辅助函数 ---
_vertex_ai_initialized = False
_google_credentials_object = None

def initialize_vertex_ai_and_credentials():
    global _vertex_ai_initialized, _google_credentials_object
    if _vertex_ai_initialized:
        return _google_credentials_object

    try:
        if GOOGLE_APPLICATION_CREDENTIALS_STRING:
            logger.info("使用 config.py 中的 GOOGLE_APPLICATION_CREDENTIALS_STRING 初始化 Vertex AI 凭证...")
            creds_json = orjson.loads(GOOGLE_APPLICATION_CREDENTIALS_STRING)
            _google_credentials_object = service_account.Credentials.from_service_account_info(creds_json)
            vertexai.init(credentials=_google_credentials_object) # 可以不指定 project 和 location，让 SDK 自动从凭证获取
            logger.info("Vertex AI SDK (带字符串凭证) 初始化成功。")
        else:
            # 尝试使用应用默认凭证 (ADC)
            logger.info("尝试使用应用默认凭证 (ADC) 初始化 Vertex AI...")
            vertexai.init() # project 和 location 可以让SDK从环境中自动获取，或者在此处指定
            _google_credentials_object = None # 表示使用的是ADC
            logger.info("Vertex AI SDK (ADC) 初始化成功。")
        _vertex_ai_initialized = True
        return _google_credentials_object
    except Exception as e:
        logger.error(f"Vertex AI 初始化失败: {e}", exc_info=True)
        _vertex_ai_initialized = False # 标记为未成功初始化
        _google_credentials_object = None
        raise RuntimeError(f"Vertex AI SDK 初始化失败: {e}") from e


# --- SSE 事件序列化 ---
async def sse_event_serializer_multimodal(event_data: AppStreamEventPy) -> bytes:
    # 注意：orjson_dumps_bytes_wrapper 应该能处理 Pydantic 模型
    return orjson_dumps_bytes_wrapper(event_data.model_dump(exclude_none=True, by_alias=True))


# --- 主要的事件生成器 ---
async def generate_gemini_events_internal(
    gemini_chat_input: ChatRequestModel,
    raw_request: Request, # FastAPI Request object
    http_client: httpx.AsyncClient, # 保持签名一致性，但Vertex SDK不直接使用它
    request_id: str # 用于日志
) -> AsyncGenerator[bytes, None]:
    log_prefix = f"RID-{request_id}"

    try:
        initialize_vertex_ai_and_credentials() # 确保Vertex AI已初始化

        # 1. 模型名称映射 (如果需要)
        #    (与您在 chat.py 中为 Gemini 模型准备的逻辑类似)
        model_name_map = {
            # "gemini-2.5-pro-preview-05-06": "gemini-1.5-pro-preview-0409", # 旧示例，根据实际可用调整
            "gemini-1.5-pro-latest": "gemini-1.5-pro-latest",
            "gemini-1.5-flash-latest": "gemini-1.5-flash-latest",
            # 添加其他您前端可能发送的 Gemini 模型名称到 Vertex AI SDK 可识别名称的映射
        }
        sdk_model_name = model_name_map.get(gemini_chat_input.model.lower(), gemini_chat_input.model)
        logger.info(f"{log_prefix}: Using Vertex AI model '{sdk_model_name}' (original: '{gemini_chat_input.model}')")
        
        gemini_model_vertex = GenerativeModel(sdk_model_name)

        # 2. 构建 Vertex AI SDK 的 contents 列表
        contents_for_vertex: List[Content] = []
        user_query_for_search_parts: List[str] = [] # 收集用户文本用于Web搜索

        for i, msg_abstract in enumerate(gemini_chat_input.messages):
            if not isinstance(msg_abstract, PartsApiMessagePy):
                logger.warning(f"{log_prefix}: Gemini handler expected PartsApiMessagePy but got {type(msg_abstract)} at index {i}. Skipping.")
                continue

            vertex_parts: List[Part] = []
            for part_wrapper in msg_abstract.parts: # part_wrapper 是 IncomingApiContentPart
                actual_part_data = None
                try:
                    if isinstance(part_wrapper, TextPartWrapper):
                        actual_part_data = part_wrapper.text_content
                        vertex_parts.append(Part.from_text(actual_part_data.text))
                        if msg_abstract.role == "user": # 仅收集用户文本用于搜索
                            user_query_for_search_parts.append(actual_part_data.text)
                    elif isinstance(part_wrapper, FileUriPartWrapper):
                        actual_part_data = part_wrapper.file_uri_content
                        if not (actual_part_data.uri.startswith("gs://") or actual_part_data.uri.startswith("https://")):
                            logger.warning(f"{log_prefix}: Invalid URI scheme for FileUriPart: {actual_part_data.uri}. Skipping.")
                            # 可以选择yield一个错误事件给前端
                            # yield await sse_event_serializer_multimodal(AppStreamEventPy(type="error", message=f"图片URI格式无效: {actual_part_data.uri}", timestamp=get_current_time_iso()))
                            continue
                        vertex_parts.append(Part.from_uri(uri=actual_part_data.uri, mime_type=actual_part_data.mime_type))
                    elif isinstance(part_wrapper, InlineDataPartWrapper):
                        actual_part_data = part_wrapper.inline_data_content
                        decoded_data = base64.b64decode(actual_part_data.base64_data)
                        vertex_parts.append(Part.from_data(data=decoded_data, mime_type=actual_part_data.mime_type))
                except Exception as e_part:
                    logger.error(f"{log_prefix}: Error processing message part for Gemini: {part_wrapper}, Error: {e_part}", exc_info=True)
                    # yield await sse_event_serializer_multimodal(AppStreamEventPy(type="error", message=f"处理消息部分失败: {e_part}", timestamp=get_current_time_iso()))
                    continue # 跳过这个损坏的part

            if vertex_parts:
                vertex_role = "model" if msg_abstract.role == "assistant" else msg_abstract.role
                if vertex_role not in ["user", "model"]:
                    logger.warning(f"{log_prefix}: Invalid role '{msg_abstract.role}' for Gemini, mapping to 'user'.")
                    vertex_role = "user"
                contents_for_vertex.append(Content(role=vertex_role, parts=vertex_parts))
        
        if not contents_for_vertex:
            logger.warning(f"{log_prefix}: No valid contents to send to Gemini model '{sdk_model_name}'.")
            yield await sse_event_serializer_multimodal(AppStreamEventPy(type="error", message="没有有效内容发送给模型。", timestamp=get_current_time_iso()))
            yield await sse_event_serializer_multimodal(AppStreamEventPy(type="finish", reason="no_content", timestamp=get_current_time_iso()))
            return

        user_query_for_search = " ".join(user_query_for_search_parts).strip()

        # 3. Web搜索逻辑 (如果Gemini也需要)
        #    与 chat.py 中的逻辑类似，但注入的 search_context_message 应该是 Content 对象或 Part 对象
        if gemini_chat_input.use_web_search and user_query_for_search:
            yield await sse_event_serializer_multimodal(AppStreamEventPy(type="status_update", stage="web_search_started", timestamp=get_current_time_iso()))
            # search_results_list = await perform_web_search(user_query_for_search, request_id)
            # ... (注入搜索结果为新的 Content(role="user", parts=[Part.from_text(...)]) 到 contents_for_vertex 合适的位置)
            # yield orjson_dumps_bytes_wrapper(...) for web_search_results and web_analysis_started
            logger.info(f"{log_prefix}: Web search for Gemini (query: {user_query_for_search[:100]}) - Placeholder, implement if needed.")
            # 假设这里Web搜索完成
            yield await sse_event_serializer_multimodal(AppStreamEventPy(type="status_update", stage="web_analysis_started", timestamp=get_current_time_iso()))


        # 4. 构建 GenerationConfig
        gen_config_dict = {}
        if gemini_chat_input.temperature is not None: gen_config_dict["temperature"] = gemini_chat_input.temperature
        if gemini_chat_input.top_p is not None: gen_config_dict["top_p"] = gemini_chat_input.top_p
        if gemini_chat_input.max_tokens is not None: gen_config_dict["max_output_tokens"] = gemini_chat_input.max_tokens
        # if gemini_chat_input.candidate_count is not None: gen_config_dict["candidate_count"] = gemini_chat_input.candidate_count # 如果支持
        # if gemini_chat_input.stop_sequences is not None: gen_config_dict["stop_sequences"] = gemini_chat_input.stop_sequences # 如果支持
        
        # 处理 thoughts (如果您的模型和SDK版本支持)
        # Vertex AI SDK 的 `GenerationConfig` (preview 版) 支持 `include_internal_details` 来获取 `Segment` 等。
        # 对于类似 "thoughts" 的输出，您可能需要检查 `gemini_chat_input.custom_model_parameters`
        # 并相应设置 `tools` 或 `tool_config` (如果Gemini使用类似方式输出思考过程)
        # 或者，Vertex AI 的某些Gemini版本直接在GenerationConfig中支持思考过程，例如：
        # if gemini_chat_input.custom_model_parameters and gemini_chat_input.custom_model_parameters.get("google", {}).get("include_thoughts"):
        #    gen_config_dict["include_thoughts"] = True # 示例，具体参数名需查阅SDK文档

        vertex_gen_config = VertexGenerationConfig(**gen_config_dict) if gen_config_dict else None
        
        # TODO: 处理 tools 和 tool_choice (如果Gemini请求需要函数调用)
        # gemini_tools: Optional[List[Tool]] = None
        # if gemini_chat_input.tools:
        #     try:
        #         gemini_tools = [Tool.from_dict(t) for t in gemini_chat_input.tools] # 假设格式兼容
        #     except Exception as e_tool:
        #         logger.error(f"{log_prefix}: Error parsing tools for Gemini: {e_tool}")
        #         # yield error event
        
        logger.debug(f"{log_prefix}: Sending to Vertex Gemini. Model: {sdk_model_name}, Contents Count: {len(contents_for_vertex)}, Config: {vertex_gen_config}")

        # 5. 调用 Gemini API 并流式处理响应
        first_chunk_received = False
        stream = await asyncio.to_thread(
            gemini_model_vertex.generate_content,
            contents_for_vertex,
            generation_config=vertex_gen_config,
            # tools=gemini_tools, # 如果支持并已准备好
            stream=True
        )

        for chunk in stream:
            if await raw_request.is_disconnected():
                logger.info(f"{log_prefix}: Client disconnected during Gemini stream.")
                break
            
            if not first_chunk_received:
                if gemini_chat_input.use_web_search and user_query_for_search: # 假设Web搜索已完成
                     yield await sse_event_serializer_multimodal(AppStreamEventPy(type="status_update", stage="web_analysis_complete", timestamp=get_current_time_iso()))
                first_chunk_received = True

            # Vertex AI SDK chunk 结构:
            # chunk.text (快捷方式，如果只有文本)
            # chunk.candidates[0].content.parts (更通用)
            # chunk.candidates[0].finish_reason
            # chunk.candidates[0].safety_ratings
            # chunk.candidates[0].function_calls (如果使用了函数调用)
            # chunk.candidates[0].internal_details (如果 GenerationConfig 中开启了 include_internal_details)

            # 示例：处理文本内容
            if chunk.candidates and chunk.candidates[0].content and chunk.candidates[0].content.parts:
                full_chunk_text = ""
                for part in chunk.candidates[0].content.parts:
                    if hasattr(part, 'text') and part.text:
                        full_chunk_text += part.text
                
                if full_chunk_text:
                    # TODO: 区分 "reasoning" (thoughts) 和 "content"
                    # 这取决于 Vertex AI SDK 如何返回 "thoughts"。
                    # 如果 thoughts 通过特定的 part 类型或字段返回，在这里处理。
                    # 假设现在所有文本都是 "content"。
                    yield await sse_event_serializer_multimodal(AppStreamEventPy(type="content", text=full_chunk_text, timestamp=get_current_time_iso()))

            # 示例：处理函数调用 (如果支持)
            # if chunk.candidates and chunk.candidates[0].function_calls:
            #     for fc in chunk.candidates[0].function_calls:
            #         # 转换为前端期望的 AppStreamEventPy 格式 (type="tool_calls_chunk" 或 "google_function_call_request")
            #         # yield await sse_event_serializer_multimodal(AppStreamEventPy(type="google_function_call_request", name=fc.name, arguments_obj=fc.args, ...))
            #         pass


            if chunk.candidates and chunk.candidates[0].finish_reason != FinishReason.UNSPECIFIED:
                finish_reason_str = FinishReason(chunk.candidates[0].finish_reason).name.lower()
                logger.info(f"{log_prefix}: Gemini stream finished with reason: {finish_reason_str}")
                yield await sse_event_serializer_multimodal(AppStreamEventPy(type="finish", reason=finish_reason_str, timestamp=get_current_time_iso()))
                break
        
        # 如果循环正常结束但没有收到明确的finish_reason (不太可能，但作为保险)
        # yield await sse_event_serializer_multimodal(AppStreamEventPy(type="finish", reason="stream_end", timestamp=get_current_time_iso()))


    except vertexai.generative_models.generation_utils.BlockedBySafetySettingError as e_safety:
        logger.error(f"{log_prefix}: Vertex Gemini content blocked by safety: {e_safety}", exc_info=True)
        yield await sse_event_serializer_multimodal(AppStreamEventPy(type="error", message="内容被安全策略阻止。", reason="SAFETY", timestamp=get_current_time_iso()))
        yield await sse_event_serializer_multimodal(AppStreamEventPy(type="finish", reason="safety", timestamp=get_current_time_iso()))
    except RuntimeError as e_runtime: # 例如 Vertex AI 初始化失败
        logger.error(f"{log_prefix}: Runtime error during Gemini processing: {e_runtime}", exc_info=True)
        yield await sse_event_serializer_multimodal(AppStreamEventPy(type="error", message=f"服务内部错误: {e_runtime}", timestamp=get_current_time_iso()))
        yield await sse_event_serializer_multimodal(AppStreamEventPy(type="finish", reason="internal_error", timestamp=get_current_time_iso()))
    except Exception as e:
        logger.error(f"{log_prefix}: Generic error in Gemini chat events: {e}", exc_info=True)
        yield await sse_event_serializer_multimodal(AppStreamEventPy(type="error", message=str(e), timestamp=get_current_time_iso()))
        yield await sse_event_serializer_multimodal(AppStreamEventPy(type="finish", reason="unknown_error", timestamp=get_current_time_iso()))


# 这个函数将被 routers/chat.py 调用
async def handle_gemini_request_entry(
    gemini_chat_input: ChatRequestModel,
    raw_request: Request,
    http_client: httpx.AsyncClient, # 传递以保持接口一致性，尽管Vertex SDK可能不直接用它
    request_id: str
):
    logger.info(f"RID-{request_id}: Entering Gemini request handler for model {gemini_chat_input.model}")
    return StreamingResponse(
        generate_gemini_events_internal(gemini_chat_input, raw_request, http_client, request_id),
        media_type="text/event-stream",
        headers=COMMON_HEADERS # 使用通用头部
    )