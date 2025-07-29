/**
 * Chat handler for processing AI chat requests
 */

import { logger } from '../utils/logger';
import { HTTPClient } from '../utils/http';
import { OpenAIHandler } from './openai';
import { GeminiHandler } from './gemini';
import { WebSearchService } from '../services/web-search';

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

      // Initialize web search service
      const webSearchService = new WebSearchService(env);

      // Perform web search if needed
      if (webSearchService.shouldPerformWebSearch(requestData)) {
        await this.performWebSearchAndUpdateMessages(requestData, webSearchService, requestId);
      }

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
   * Parse request body (handles both JSON and form data, matching backend-docker)
   */
  async parseRequestBody(request) {
    const contentType = request.headers.get('content-type') || '';
    
    if (contentType.includes('application/json')) {
      return await request.json();
    } else if (contentType.includes('multipart/form-data')) {
      const formData = await request.formData();
      const chatRequestJson = formData.get('chat_request_json');
      
      if (!chatRequestJson) {
        throw new Error('Missing chat_request_json in form data');
      }
      
      const requestData = JSON.parse(chatRequestJson);
      
      // Handle uploaded files (matching backend-docker behavior)
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
    } else if (contentType.includes('application/x-www-form-urlencoded')) {
      const formData = await request.formData();
      const chatRequestJson = formData.get('chat_request_json');
      
      if (!chatRequestJson) {
        throw new Error('Missing chat_request_json in form data');
      }
      
      return JSON.parse(chatRequestJson);
    } else {
      throw new Error(`Unsupported content type: ${contentType}`);
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
   * Perform web search and update messages
   */
  async performWebSearchAndUpdateMessages(requestData, webSearchService, requestId) {
    const logPrefix = `RID-${requestId}`;
    
    try {
      // Extract search query from the last user message
      const lastMessage = requestData.messages[requestData.messages.length - 1];
      if (!lastMessage || lastMessage.role !== 'user') {
        return;
      }

      const searchQuery = lastMessage.content || 
                         (lastMessage.parts && lastMessage.parts.find(p => p.type === 'text_content')?.text) || '';

      if (!searchQuery.trim()) {
        return;
      }

      logger.info(`${logPrefix}: Performing web search for query: "${searchQuery.substring(0, 100)}"`);

      // Perform web search
      const searchResults = await webSearchService.performWebSearch(searchQuery, requestId);

      if (searchResults.length > 0) {
        // Generate search context
        const searchContext = webSearchService.generateSearchContextMessageContent(
          searchResults, 
          searchQuery, 
          requestId
        );

        // Add search context to the user message
        if (lastMessage.type === 'simple_text_message') {
          lastMessage.content = searchContext + lastMessage.content;
        } else if (lastMessage.type === 'parts_message') {
          // Find text part and prepend search context
          const textPart = lastMessage.parts.find(p => p.type === 'text_content');
          if (textPart) {
            textPart.text = searchContext + textPart.text;
          } else {
            // Add search context as first part
            lastMessage.parts.unshift({
              type: 'text_content',
              text: searchContext
            });
          }
        }

        logger.info(`${logPrefix}: Added ${searchResults.length} search results to message context`);
      }

    } catch (error) {
      logger.error(`${logPrefix}: Web search error:`, error);
      // Continue without web search if it fails
    }
  }

  /**
   * Generate unique request ID
   */
  generateRequestId() {
    return crypto.randomUUID();
  }
}