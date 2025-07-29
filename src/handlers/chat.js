/**
 * Main chat handler that acts as a dispatcher.
 * It parses the request and routes it to the appropriate handler (OpenAI or Gemini)
 * based on the `api_address`, mirroring the logic in `backend-docker/eztalk_proxy/api/chat.py`.
 */
import { isGoogleOfficialAPI, parseRequestBody, generateRequestId } from '../utils/api_logic.js';
import { handleOpenAIRequest } from './openai.js';
import { handleGeminiRequest } from './gemini.js';

/**
 * The main entry point for handling chat requests.
 * @param {Request} request The incoming Fetch API request.
 * @param {object} env The Cloudflare environment variables.
 * @returns {Promise<Response>}
 */
export async function handleChatRequest(request, env) {
    const requestId = generateRequestId();
    const logPrefix = `RID-${requestId}`;

    try {
        // 1. Parse the incoming request, which could be JSON or multipart/form-data
        const requestData = await parseRequestBody(request);
        console.log(`${logPrefix}: Parsed request for model ${requestData.model}`);

        // 2. Determine the routing strategy based on the API address
        const useGeminiHandler = isGoogleOfficialAPI(requestData.api_address);

        // 3. Dispatch to the appropriate handler
        if (useGeminiHandler) {
            console.log(`${logPrefix}: Dispatching to Gemini handler.`);
            return await handleGeminiRequest(requestData, env, requestId);
        } else {
            console.log(`${logPrefix}: Dispatching to OpenAI compatible handler.`);
            return await handleOpenAIRequest(requestData, env, requestId);
        }

    } catch (error) {
        console.error(`${logPrefix}: Error in main chat handler:`, error);
        const errorResponse = {
            error: 'Bad Request',
            message: error.message,
        };
        return new Response(JSON.stringify(errorResponse), {
            status: 400,
            headers: { 'Content-Type': 'application/json' }
        });
    }
}