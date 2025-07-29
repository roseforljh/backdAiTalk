/**
 * OpenAI compatible handler for processing chat requests.
 * This logic is aligned with `backend-docker/eztalk_proxy/api/openai.py`.
 */
import { arrayBufferToBase64 } from '../utils/api_logic.js';

// Supported MIME types, aligned with the Python version
const SUPPORTED_IMAGE_MIME_TYPES = ["image/jpeg", "image/png", "image/gif", "image/webp"];
const AUDIO_MIME_TYPES = [
    "audio/wav", "audio/x-wav", "audio/mpeg", "audio/mp3", "audio/aac", "audio/ogg",
    "audio/opus", "audio/flac", "audio/3gpp", "audio/amr", "audio/aiff", "audio/x-m4a"
];
const VIDEO_MIME_TYPES = [
    "video/mp4", "video/mpeg", "video/quicktime", "video/x-msvideo", "video/x-flv",
    "video/x-matroska", "video/webm", "video/x-ms-wmv", "video/3gpp", "video/x-m4v"
];

/**
 * Handles requests intended for OpenAI-compatible APIs.
 * @param {object} requestData The parsed request data from the client.
 * @param {object} env The Cloudflare environment variables.
 * @param {string} requestId The unique ID for this request.
 * @returns {Promise<Response>}
 */
export async function handleOpenAIRequest(requestData, env, requestId) {
    const logPrefix = `RID-${requestId}`;
    console.log(`${logPrefix}: Processing with OpenAI compatible handler for model ${requestData.model}`);

    try {
        // 1. Process messages and uploaded files to create the final message list
        const finalMessages = await processMessagesAndFiles(requestData, logPrefix);

        // 2. Prepare the final payload for the upstream API
        const upstreamPayload = prepareUpstreamPayload(requestData, finalMessages);

        // 3. Determine the target URL
        const targetUrl = requestData.api_address || 'https://api.openai.com/v1/chat/completions';

        // 4. Make the API call
        const response = await fetch(targetUrl, {
            method: 'POST',
            headers: {
                'Authorization': `Bearer ${requestData.api_key}`,
                'Content-Type': 'application/json',
                'Accept': 'text/event-stream',
            },
            body: JSON.stringify(upstreamPayload),
        });

        // 5. Stream the response back to the client
        if (!response.ok) {
            const errorBody = await response.text();
            console.error(`${logPrefix}: Upstream API error ${response.status}: ${errorBody}`);
            return new Response(JSON.stringify({ error: 'Upstream API Error', message: errorBody }), { status: response.status });
        }

        return new Response(response.body, {
            status: response.status,
            headers: {
                'Content-Type': 'text/event-stream',
                'Cache-Control': 'no-cache',
                'Connection': 'keep-alive',
            }
        });

    } catch (error) {
        console.error(`${logPrefix}: OpenAI handler error:`, error);
        return new Response(JSON.stringify({ error: 'Internal Server Error', message: error.message }), { status: 500 });
    }
}

/**
 * Processes messages and integrates uploaded files, converting them to the OpenAI-compatible format.
 * @param {object} requestData The request data.
 * @param {string} logPrefix The logging prefix.
 * @returns {Promise<Array<object>>} The final list of messages for the API.
 */
async function processMessagesAndFiles(requestData, logPrefix) {
    const finalMessages = [];
    const newMultimodalParts = [];

    // Process uploaded files first
    if (requestData.uploadedFiles && requestData.uploadedFiles.length > 0) {
        console.log(`${logPrefix}: Processing ${requestData.uploadedFiles.length} uploaded files.`);
        for (const file of requestData.uploadedFiles) {
            const arrayBuffer = await file.arrayBuffer();
            const base64Data = arrayBufferToBase64(arrayBuffer);
            const mimeType = file.type.toLowerCase();

            if (SUPPORTED_IMAGE_MIME_TYPES.includes(mimeType)) {
                newMultimodalParts.push({
                    type: 'image_url',
                    image_url: { url: `data:${mimeType};base64,${base64Data}` }
                });
            } else if (AUDIO_MIME_TYPES.includes(mimeType)) {
                newMultimodalParts.push({
                    type: 'input_audio', // Custom type for Gemini compatibility
                    input_audio: { data: base64Data, format: mimeType.split('/')[1] }
                });
            } else if (VIDEO_MIME_TYPES.includes(mimeType)) {
                 newMultimodalParts.push({
                    type: 'image_url', // Videos are sent as data URIs in image_url for Gemini OpenAI compat
                    image_url: { url: `data:${mimeType};base64,${base64Data}` }
                });
            } else {
                console.warn(`${logPrefix}: Skipping unsupported file type: ${mimeType}`);
            }
        }
    }

    // Process message history
    for (const msg of requestData.messages) {
        const messagePayload = { role: msg.role };
        let contentParts = [];

        if (msg.type === 'simple_text_message' || msg.message_type === 'simple_text_message') {
            contentParts.push({ type: 'text', text: msg.content });
        } else if (msg.type === 'parts_message' || msg.message_type === 'parts_message') {
            for (const part of msg.parts) {
                if (part.type === 'text_content') {
                    contentParts.push({ type: 'text', text: part.text });
                } else if (part.type === 'inline_data_content') {
                     contentParts.push({
                        type: 'image_url',
                        image_url: { url: `data:${part.mime_type};base64,${part.base64_data}` }
                    });
                }
            }
        }
        messagePayload.content = contentParts;
        finalMessages.push(messagePayload);
    }

    // Attach new multimodal parts to the last user message
    const lastUserMessage = finalMessages.slice().reverse().find(m => m.role === 'user');
    if (lastUserMessage && newMultimodalParts.length > 0) {
        if (!Array.isArray(lastUserMessage.content)) {
             lastUserMessage.content = [{ type: 'text', text: lastUserMessage.content || '' }];
        }
        lastUserMessage.content.push(...newMultimodalParts);
    }
    
    // Simplify content field if possible
    finalMessages.forEach(msg => {
        if (Array.isArray(msg.content) && msg.content.length === 1 && msg.content[0].type === 'text') {
            msg.content = msg.content[0].text;
        }
    });

    return finalMessages;
}

/**
 * Prepares the final JSON payload for the upstream API call.
 * @param {object} requestData The original request data.
 * @param {Array<object>} finalMessages The processed messages.
 * @returns {object} The final payload.
 */
function prepareUpstreamPayload(requestData, finalMessages) {
    const payload = {
        model: requestData.model,
        messages: finalMessages,
        stream: true,
    };

    // Add optional parameters
    const optionalParams = ['temperature', 'top_p', 'max_tokens', 'tools', 'tool_choice'];
    optionalParams.forEach(param => {
        if (requestData[param] !== undefined && requestData[param] !== null) {
            payload[param] = requestData[param];
        }
    });
    
    // Handle aliased names
    if (requestData.maxTokens) payload.max_tokens = requestData.maxTokens;
    if (requestData.topP) payload.top_p = requestData.topP;
    if (requestData.toolChoice) payload.tool_choice = requestData.toolChoice;

    return payload;
}