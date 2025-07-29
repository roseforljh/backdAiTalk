/**
 * Router for handling different API endpoints
 */

import { ChatHandler } from './handlers/chat';
import { logger } from './utils/logger';

export class Router {
  constructor() {
    this.chatHandler = new ChatHandler();
  }

  async handle(request, env, ctx) {
    const url = new URL(request.url);
    const path = url.pathname;
    const method = request.method;

    try {
      // Chat endpoints
      if (path.startsWith('/api/v1/chat')) {
        return await this.chatHandler.handle(request, env, ctx);
      }

      // OpenAI compatibility endpoints
      if (path.startsWith('/v1/chat/completions')) {
        return await this.chatHandler.handleOpenAICompat(request, env, ctx);
      }

      // Gemini endpoints
      if (path.startsWith('/api/gemini')) {
        return await this.chatHandler.handleGemini(request, env, ctx);
      }

      // Default 404 response
      return new Response(JSON.stringify({
        error: 'Not Found',
        message: `Endpoint ${method} ${path} not found`,
        available_endpoints: [
          '/health',
          '/api/v1/chat/*',
          '/v1/chat/completions',
          '/api/gemini/*'
        ]
      }), {
        status: 404,
        headers: {
          'Content-Type': 'application/json'
        }
      });

    } catch (error) {
      logger.error(`Router error for ${method} ${path}:`, error);
      
      return new Response(JSON.stringify({
        error: 'Internal Server Error',
        message: error.message,
        path: path,
        method: method
      }), {
        status: 500,
        headers: {
          'Content-Type': 'application/json'
        }
      });
    }
  }
}