/**
 * EzTalk Proxy for Cloudflare Workers
 *
 * Main entry point and router. This file is responsible for receiving all incoming
 * requests, handling CORS preflight, routing to the appropriate handler,
 * and returning a final response.
 *
 * The routing logic is designed to be fully compatible with the `backend-docker` version.
 */
import { handleChatRequest } from './handlers/chat.js';
import { handleOpenAIRequest } from './handlers/openai.js'; // For /v1/chat/completions

// A simple CORS handler
function handleCors(request) {
    const headers = {
        'Access-Control-Allow-Origin': '*',
        'Access-Control-Allow-Methods': 'GET, POST, OPTIONS',
        'Access-Control-Allow-Headers': 'Content-Type, Authorization',
    };
    if (request.method === 'OPTIONS') {
        return new Response(null, { headers });
    }
    return headers;
}

export default {
    /**
     * Main fetch handler for the Cloudflare Worker.
     * @param {Request} request The incoming request.
     * @param {object} env Environment variables.
     * @returns {Promise<Response>}
     */
    async fetch(request, env) {
        const corsHeaders = handleCors(request);
        if (request.method === 'OPTIONS') {
            return corsHeaders;
        }

        const url = new URL(request.url);
        const path = url.pathname;
        let response;

        try {
            // --- Main Router ---
            if (path === '/health') {
                // *** UPDATED VERSION IDENTIFIER FOR DEPLOYMENT VERIFICATION ***
                const healthInfo = {
                    status: 'ok',
                    version: '2.1.0-formdata-fix', // This confirms the new fix is live
                    message: 'Deployment successful. FormData parsing is now more robust.'
                };
                response = new Response(JSON.stringify(healthInfo), {
                    status: 200,
                    headers: { 'Content-Type': 'application/json' }
                });
            }
            // *** CORE FIX: Ensure /chat is handled correctly ***
            // Both /chat (for Android client) and /api/v1/chat are routed to the main handler.
            else if (request.method === 'POST' && (path === '/chat' || path === '/api/v1/chat')) {
                response = await handleChatRequest(request, env);
            }
            // Handle the standard OpenAI compatibility endpoint
            else if (request.method === 'POST' && path === '/v1/chat/completions') {
                const requestData = await request.json();
                const adaptedRequest = {
                    ...requestData,
                    api_key: request.headers.get('Authorization')?.replace('Bearer ', ''),
                    api_address: 'https://api.openai.com/v1/chat/completions',
                };
                response = await handleOpenAIRequest(adaptedRequest, env, crypto.randomUUID());
            }
            // Default 404 Not Found
            else {
                const availableEndpoints = [
                    'GET /health',
                    'POST /chat',
                    'POST /api/v1/chat',
                    'POST /v1/chat/completions',
                ];
                const errorResponse = {
                    error: "Not Found",
                    message: `Endpoint ${request.method} ${path} not found.`,
                    available_endpoints: availableEndpoints,
                };
                response = new Response(JSON.stringify(errorResponse), { status: 404 });
            }
        } catch (error) {
            console.error(`Unhandled error in main fetch handler for ${path}:`, error);
            response = new Response(JSON.stringify({ error: 'Internal Server Error', message: error.message }), { status: 500 });
        }

        // Attach CORS headers to the final response
        const finalHeaders = new Headers(response.headers);
        Object.entries(corsHeaders).forEach(([key, value]) => {
            finalHeaders.set(key, value);
        });
        // Ensure JSON content type for all responses from this router
        finalHeaders.set('Content-Type', 'application/json');

        return new Response(response.body, {
            status: response.status,
            statusText: response.statusText,
            headers: finalHeaders,
        });
    }
};