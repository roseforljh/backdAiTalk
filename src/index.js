/**
 * EzTalk Proxy for Cloudflare Workers
 * Main entry point
 */

import { Router } from './router';
import { corsHeaders, handleCORS } from './utils/cors';
import { logger } from './utils/logger';

const router = new Router();

export default {
  async fetch(request, env, ctx) {
    try {
      // Handle CORS preflight requests
      if (request.method === 'OPTIONS') {
        return handleCORS(request);
      }

      const url = new URL(request.url);
      logger.info(`${request.method} ${url.pathname}`);

      // Health check endpoint
      if (url.pathname === '/health') {
        return new Response(JSON.stringify({
          status: 'ok',
          detail: 'EzTalk Proxy Worker is running',
          app_version: env.APP_VERSION || '1.0.0',
          timestamp: new Date().toISOString()
        }), {
          headers: {
            'Content-Type': 'application/json',
            ...corsHeaders
          }
        });
      }

      // Route the request
      const response = await router.handle(request, env, ctx);
      
      // Add CORS headers to all responses
      const corsResponse = new Response(response.body, {
        status: response.status,
        statusText: response.statusText,
        headers: {
          ...Object.fromEntries(response.headers),
          ...corsHeaders
        }
      });

      return corsResponse;

    } catch (error) {
      logger.error('Unhandled error in main handler:', error);
      
      return new Response(JSON.stringify({
        error: 'Internal Server Error',
        message: error.message,
        timestamp: new Date().toISOString()
      }), {
        status: 500,
        headers: {
          'Content-Type': 'application/json',
          ...corsHeaders
        }
      });
    }
  }
};