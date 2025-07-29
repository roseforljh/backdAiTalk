/**
 * Chat handler for processing AI chat requests
 */

import { logger } from '../utils/logger';
import { HTTPClient } from '../utils/http';
import { OpenAIHandler } from './openai';
import { GeminiHandler } from './gemini';

export class ChatHandler {
  constructor() {
    this.openaiHandler = new OpenAIHandler();
    this.geminiHandler = new GeminiHandler();
  }

  /**
   * Main chat endpoint handler
   */
  async handle(request, env, ctx) {
    const url = new URL(request.url);
    const path = url.pathname;
    const method = request.method;

    if (method !== 'POST') {
      return new Response(JSON.stringify({
        error: 'Method Not Allowed',
        message: 'Only POST method is supported for chat endpoints'
      }), {
        status: 405,
        headers: { 'Content-Type': 'application/json' }
      });
    }

    try {
      // Parse request body
      const requestData = await this.parseRequestBody(request);
      const requestId = this.generateRequestId();
      
      logger.info(`RID-${requestId}: Received chat request for provider '${requestData.provider}' and model '${requestData.model}'`);

      // Determine which handler to use based on API address
      const useGeminiFormat = this.isGoogleOfficialAPI(requestData.api_address || requestData.apiAddress);
      
      if (useGeminiFormat) {
        logger.info(`RID-${requestId}: Using Gemini handler for ${requestData.model}`);
        return await this.geminiHandler.handle(requestData, env, ctx, requestId);
      } else {
        logger.info(`RID-${requestId}: Using OpenAI compatible handler for ${requestData.model}`);
        return await this.openaiHandler.handle(requestData, env, ctx, requestId);
      }

    } catch (error) {
      logger.error('Chat handler error:', error);
      
      return new Response(JSON.stringify({
        error: 'Bad Request',
        message: error.message,
        timestamp: new Date().toISOString()
      }), {
        status: 400,
        headers: { 'Content-Type': 'application/json' }
      });
    }
  }

  /**
   * OpenAI compatibility endpoint
   */
  async handleOpenAICompat(request, env, ctx) {
    try {
      const requestData = await request.json();
      const requestId = this.generateRequestId();
      
      logger.info(`RID-${requestId}: OpenAI compatibility request for model '${requestData.model}'`);
      
      // Convert OpenAI format to internal format
      const internalFormat = this.convertOpenAIToInternal(requestData, env);
      
      return await this.openaiHandler.handle(internalFormat, env, ctx, requestId);

    } catch (error) {
      logger.error('OpenAI compatibility handler error:', error);
      
      return new Response(JSON.stringify({
        error: 'Bad Request',
        message: error.message
      }), {
        status: 400,
        headers: { 'Content-Type': 'application/json' }
      });
    }
  }

  /**
   * Gemini direct endpoint
   */
  async handleGemini(request, env, ctx) {
    try {
      const requestData = await request.json();
      const requestId = this.generateRequestId();
      
      logger.info(`RID-${requestId}: Direct Gemini request`);
      
      return await this.geminiHandler.handle(requestData, env, ctx, requestId);

    } catch (error) {
      logger.error('Gemini handler error:', error);
      
      return new Response(JSON.stringify({
        error: 'Bad Request',
        message: error.message
      }), {
        status: 400,
        headers: { 'Content-Type': 'application/json' }
      });
    }
  }

  /**
   * Parse request body (handles both JSON and form data)
   */
  async parseRequestBody(request) {
    const contentType = request.headers.get('content-type') || '';
    
    if (contentType.includes('application/json')) {
      return await request.json();
    } else if (contentType.includes('multipart/form-data') || contentType.includes('application/x-www-form-urlencoded')) {
      const formData = await request.formData();
      const chatRequestJson = formData.get('chat_request_json');
      
      if (!chatRequestJson) {
        throw new Error('Missing chat_request_json in form data');
      }
      
      const requestData = JSON.parse(chatRequestJson);
      
      // Handle uploaded files if any
      const uploadedFiles = [];
      for (const [key, value] of formData.entries()) {
        if (key !== 'chat_request_json' && value instanceof File) {
          uploadedFiles.push(value);
        }
      }
      
      if (uploadedFiles.length > 0) {
        requestData.uploadedFiles = uploadedFiles;
        logger.info(`Received ${uploadedFiles.length} uploaded files`);
      }
      
      return requestData;
    } else {
      throw new Error('Unsupported content type');
    }
  }

  /**
   * Check if API address is Google official
   */
  isGoogleOfficialAPI(apiAddress) {
    if (!apiAddress) {
      return false;
    }

    try {
      const url = new URL(apiAddress);
      const domain = url.hostname.toLowerCase();
      
      const googleDomains = [
        'generativelanguage.googleapis.com',
        'aiplatform.googleapis.com',
        'googleapis.com',
        'ai.google.dev'
      ];
      
      return googleDomains.some(googleDomain => 
        domain === googleDomain || domain.endsWith('.' + googleDomain)
      );
    } catch (error) {
      logger.warn(`Failed to parse API address '${apiAddress}':`, error);
      return false;
    }
  }

  /**
   * Convert OpenAI format to internal format
   */
  convertOpenAIToInternal(openaiRequest, env) {
    return {
      provider: 'openai',
      model: openaiRequest.model,
      messages: openaiRequest.messages.map(msg => ({
        role: msg.role,
        type: 'simple_text_message',
        content: msg.content
      })),
      api_key: env.OPENAI_API_KEY,
      api_address: 'https://api.openai.com/v1/chat/completions',
      temperature: openaiRequest.temperature,
      max_tokens: openaiRequest.max_tokens,
      top_p: openaiRequest.top_p,
      tools: openaiRequest.tools,
      tool_choice: openaiRequest.tool_choice
    };
  }

  /**
   * Generate unique request ID
   */
  generateRequestId() {
    return crypto.randomUUID();
  }
}