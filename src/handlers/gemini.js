/**
 * Gemini handler for processing Google AI requests.
 * This logic is aligned with `backend-docker/eztalk_proxy/api/gemini.py`.
 */
import { arrayBufferToBase64 } from '../utils/api_logic.js';

/**
 * Handles requests intended for Google Gemini APIs.
 * @param {object} requestData The parsed request data from the client.
 * @param {object} env The Cloudflare environment variables.
 * @param {string} requestId The unique ID for this request.
 * @returns {Promise<Response>}
 */
export async function handleGeminiRequest(requestData, env, requestId) {
    const logPrefix = `RID-${requestId}`;
    console.log(`${logPrefix}: Processing with Gemini handler for model ${requestData.model}`);

    try {
        // 1. Prepare the final payload for the Gemini API
        const upstreamPayload = await prepareUpstreamPayload(requestData, logPrefix);

        // 2. Determine the target URL
        const targetUrl = buildGeminiApiUrl(requestData);

        // 3. Make the API call
        const response = await fetch(targetUrl, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
                'Accept': 'text/event-stream',
                // Note: Gemini REST API uses a query parameter for the key, not a header.
            },
            body: JSON.stringify(upstreamPayload),
        });

        // 4. Stream the response back to the client
        if (!response.ok) {
            const errorBody = await response.text();
            console.error(`${logPrefix}: Upstream API error ${response.status}: ${errorBody}`);
            return new Response(JSON.stringify({ error: 'Upstream API Error', message: errorBody }), { status: response.status });
        }
        
        // Gemini stream needs to be transformed into OpenAI SSE format for the client.
        const transformedStream = transformGeminiStreamToOpenAI(response.body, requestId, requestData.model);
        return new Response(transformedStream, {
            status: 200,
            headers: {
                'Content-Type': 'text/event-stream',
                'Cache-Control': 'no-cache',
                'Connection': 'keep-alive',
            }
        });

    } catch (error) {
        console.error(`${logPrefix}: Gemini handler error:`, error);
        return new Response(JSON.stringify({ error: 'Internal Server Error', message: error.message }), { status: 500 });
    }
}

/**
 * Builds the correct Gemini API URL.
 * @param {object} requestData The request data.
 * @returns {string} The final API URL.
 */
function buildGeminiApiUrl(requestData) {
    const baseUrl = requestData.api_address || 'https://generativelanguage.googleapis.com';
    const model = requestData.model;
    const apiKey = requestData.api_key;
    return `${baseUrl}/v1beta/models/${model}:streamGenerateContent?key=${apiKey}&alt=sse`;
}

/**
 * Prepares the final JSON payload for the Gemini API call.
 * @param {object} requestData The original request data.
 * @param {string} logPrefix The logging prefix.
 * @returns {Promise<object>} The final payload.
 */
async function prepareUpstreamPayload(requestData, logPrefix) {
    const payload = {
        contents: await convertMessagesToGeminiFormat(requestData, logPrefix),
        generationConfig: {},
    };

    // Map generation config
    const genConfig = requestData.generation_config || {};
    if (requestData.temperature !== undefined) genConfig.temperature = requestData.temperature;
    if (requestData.top_p !== undefined || requestData.topP !== undefined) genConfig.topP = requestData.top_p || requestData.topP;
    if (requestData.max_tokens !== undefined || requestData.maxTokens !== undefined) genConfig.maxOutputTokens = requestData.max_tokens || requestData.maxTokens;
    
    if (Object.keys(genConfig).length > 0) {
        payload.generationConfig = genConfig;
    }

    // Tools are not yet implemented in this simplified version.

    return payload;
}

/**
 * Converts the standard message format to Gemini's `contents` format.
 * @param {object} requestData The request data containing messages and files.
 * @param {string} logPrefix The logging prefix.
 * @returns {Promise<Array<object>>} The `contents` array for the Gemini API.
 */
async function convertMessagesToGeminiFormat(requestData, logPrefix) {
    const contents = [];
    const newMultimodalParts = [];

    // Process uploaded files into Gemini parts
    if (requestData.uploadedFiles && requestData.uploadedFiles.length > 0) {
        console.log(`${logPrefix}: Processing ${requestData.uploadedFiles.length} files for Gemini format.`);
        for (const file of requestData.uploadedFiles) {
            const arrayBuffer = await file.arrayBuffer();
            const base64Data = arrayBufferToBase64(arrayBuffer);
            newMultimodalParts.push({
                inlineData: {
                    mimeType: file.type,
                    data: base64Data,
                }
            });
        }
    }

    // Process message history
    for (const msg of requestData.messages) {
        const content = {
            role: msg.role === 'assistant' ? 'model' : 'user', // Map roles
            parts: [],
        };

        if (msg.type === 'simple_text_message' || msg.message_type === 'simple_text_message') {
            content.parts.push({ text: msg.content });
        } else if (msg.type === 'parts_message' || msg.message_type === 'parts_message') {
            for (const part of msg.parts) {
                if (part.type === 'text_content') {
                    content.parts.push({ text: part.text });
                } else if (part.type === 'inline_data_content') {
                    content.parts.push({
                        inlineData: {
                            mimeType: part.mime_type,
                            data: part.base64_data,
                        }
                    });
                }
            }
        }
        contents.push(content);
    }

    // Attach new multimodal parts to the last user message
    const lastUserMessage = contents.slice().reverse().find(c => c.role === 'user');
    if (lastUserMessage && newMultimodalParts.length > 0) {
        lastUserMessage.parts.push(...newMultimodalParts);
    }

    return contents;
}

/**
 * Transforms a Gemini SSE stream into an OpenAI-compatible SSE stream.
 * @param {ReadableStream} geminiStream The original stream from the Gemini API.
 * @param {string} requestId The request ID.
 * @param {string} model The model name.
 * @returns {ReadableStream} A new stream with OpenAI-formatted events.
 */
function transformGeminiStreamToOpenAI(geminiStream, requestId, model) {
    const reader = geminiStream.getReader();
    const decoder = new TextDecoder();
    
    return new ReadableStream({
        async pull(controller) {
            const { done, value } = await reader.read();
            if (done) {
                controller.close();
                return;
            }

            const chunk = decoder.decode(value, { stream: true });
            const lines = chunk.split('\n');

            for (const line of lines) {
                if (line.startsWith('data: ')) {
                    try {
                        const jsonStr = line.substring(5).trim();
                        if (!jsonStr) continue;
                        
                        const geminiEvent = JSON.parse(jsonStr);
                        const text = geminiEvent?.candidates?.[0]?.content?.parts?.[0]?.text || '';
                        
                        if (text) {
                            const openaiEvent = {
                                id: `chatcmpl-${requestId}`,
                                object: 'chat.completion.chunk',
                                created: Math.floor(Date.now() / 1000),
                                model: model,
                                choices: [{
                                    index: 0,
                                    delta: { content: text },
                                    finish_reason: null,
                                }],
                            };
                            controller.enqueue(`data: ${JSON.stringify(openaiEvent)}\n\n`);
                        }
                    } catch (e) {
                        console.error('Error parsing Gemini stream chunk:', e, 'Chunk:', line);
                    }
                }
            }
        }
    });
}