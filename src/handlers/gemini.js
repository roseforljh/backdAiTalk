/**
 * Gemini handler for processing Google AI requests
 */

import { logger } from '../utils/logger';
import { HTTPClient } from '../utils/http';

export class GeminiHandler {
  constructor() {
    this.imageMimeTypes = ['image/png', 'image/jpeg', 'image/webp', 'image/heic', 'image/heif'];
    this.documentMimeTypes = [
      'application/pdf',
      'application/x-javascript', 'text/javascript',
      'application/x-python', 'text/x-python',
      'text/plain',
      'text/html',
      'text/css',
      'text/md',
      'text/markdown',
      'text/csv',
      'text/xml',
      'text/rtf'
    ];
    this.videoMimeTypes = [
      'video/mp4', 'video/mpeg', 'video/quicktime', 'video/x-msvideo', 'video/x-flv',
      'video/x-matroska', 'video/webm', 'video/x-ms-wmv', 'video/3gpp', 'video/x-m4v'
    ];
    this.audioMimeTypes = [
      'audio/wav', 'audio/mpeg', 'audio/aac', 'audio/ogg', 'audio/opus', 'audio/flac', 'audio/3gpp'
    ];
  }

  /**
   * Handle Gemini chat request
   */
  async handle(requestData, env, ctx, requestId) {
    const logPrefix = `RID-${requestId}`;
    
    try {
      logger.info(`${logPrefix}: Processing Gemini request for model ${requestData.model}`);

      // Initialize HTTP client
      const httpClient = new HTTPClient(env);

      // Process uploaded files if any
      if (requestData.uploadedFiles && requestData.uploadedFiles.length > 0) {
        await this.processUploadedFiles(requestData, logPrefix);
      }

      // Prepare the request for Gemini API
      const geminiRequest = await this.prepareGeminiRequest(requestData, env, logPrefix);

      // Determine API endpoint
      const apiUrl = this.buildGeminiApiUrl(requestData, env);
      
      logger.debug(`${logPrefix}: Making request to ${apiUrl}`);

      const response = await httpClient.post(apiUrl, geminiRequest, {
        'Content-Type': 'application/json',
        'x-goog-api-key': requestData.api_key || requestData.apiKey || env.GEMINI_API_KEY
      });

      // Handle streaming response
      return this.handleStreamingResponse(response, logPrefix);

    } catch (error) {
      logger.error(`${logPrefix}: Gemini handler error:`, error);
      
      return new Response(JSON.stringify({
        error: 'Internal Server Error',
        message: error.message,
        request_id: requestId
      }), {
        status: 500,
        headers: {
          'Content-Type': 'application/json'
        }
      });
    }
  }

  /**
   * Build Gemini API URL
   */
  buildGeminiApiUrl(requestData, env) {
    const baseUrl = requestData.api_address || requestData.apiAddress || 
                   'https://generativelanguage.googleapis.com/v1beta';
    
    const model = requestData.model || 'gemini-pro';
    
    // Handle different Gemini API endpoints
    if (baseUrl.includes('googleapis.com')) {
      return `${baseUrl}/models/${model}:streamGenerateContent`;
    } else {
      // Custom endpoint
      return `${baseUrl}/chat/completions`;
    }
  }

  /**
   * Prepare Gemini request format
   */
  async prepareGeminiRequest(requestData, env, logPrefix) {
    const contents = await this.convertMessagesToGeminiFormat(requestData.messages, logPrefix);
    
    const request = {
      contents: contents,
      generationConfig: {
        temperature: requestData.temperature,
        maxOutputTokens: requestData.max_tokens || requestData.maxTokens,
        topP: requestData.top_p || requestData.topP
      }
    };

    // Add generation config if provided
    if (requestData.generation_config || requestData.generationConfig) {
      Object.assign(request.generationConfig, requestData.generation_config || requestData.generationConfig);
    }

    // Add tools if provided
    if (requestData.tools && requestData.tools.length > 0) {
      request.tools = this.convertToolsToGeminiFormat(requestData.tools);
    }

    logger.debug(`${logPrefix}: Prepared Gemini request with ${contents.length} content items`);
    
    return request;
  }

  /**
   * Convert messages to Gemini format
   */
  async convertMessagesToGeminiFormat(messages, logPrefix) {
    const contents = [];

    for (const message of messages) {
      const content = {
        role: this.mapRoleToGemini(message.role),
        parts: []
      };

      if (message.type === 'simple_text_message' || message.message_type === 'simple_text_message') {
        content.parts.push({
          text: message.content
        });
      } else if (message.type === 'parts_message' || message.message_type === 'parts_message') {
        for (const part of message.parts) {
          if (part.type === 'text_content') {
            content.parts.push({
              text: part.text
            });
          } else if (part.type === 'inline_data_content') {
            content.parts.push({
              inlineData: {
                mimeType: part.mime_type || part.mimeType,
                data: part.base64_data || part.base64Data
              }
            });
          } else if (part.type === 'file_uri_content') {
            content.parts.push({
              fileData: {
                mimeType: part.mime_type || part.mimeType,
                fileUri: part.uri
              }
            });
          }
        }
      }

      if (content.parts.length > 0) {
        contents.push(content);
      }
    }

    logger.debug(`${logPrefix}: Converted ${messages.length} messages to ${contents.length} Gemini contents`);
    return contents;
  }

  /**
   * Map role to Gemini format
   */
  mapRoleToGemini(role) {
    const roleMap = {
      'user': 'user',
      'assistant': 'model',
      'system': 'user', // Gemini doesn't have system role, convert to user
      'function': 'function',
      'tool': 'function'
    };
    return roleMap[role] || 'user';
  }

  /**
   * Convert tools to Gemini format
   */
  convertToolsToGeminiFormat(tools) {
    return tools.map(tool => {
      if (tool.type === 'function') {
        return {
          functionDeclarations: [{
            name: tool.function.name,
            description: tool.function.description,
            parameters: tool.function.parameters
          }]
        };
      }
      return tool;
    });
  }

  /**
   * Process uploaded files
   */
  async processUploadedFiles(requestData, logPrefix) {
    logger.info(`${logPrefix}: Processing ${requestData.uploadedFiles.length} uploaded files`);

    for (const file of requestData.uploadedFiles) {
      try {
        const arrayBuffer = await file.arrayBuffer();
        const base64Data = this.arrayBufferToBase64(arrayBuffer);
        
        // Add file content to the last user message
        const lastMessage = requestData.messages[requestData.messages.length - 1];
        if (lastMessage && lastMessage.role === 'user') {
          // Convert to parts message if it's not already
          if (lastMessage.type === 'simple_text_message') {
            lastMessage.type = 'parts_message';
            lastMessage.message_type = 'parts_message';
            lastMessage.parts = [
              {
                type: 'text_content',
                text: lastMessage.content
              }
            ];
            delete lastMessage.content;
          }

          // Add file part
          lastMessage.parts.push({
            type: 'inline_data_content',
            base64_data: base64Data,
            mime_type: file.type
          });
        }

        logger.debug(`${logPrefix}: Processed file ${file.name} (${file.type})`);
      } catch (error) {
        logger.error(`${logPrefix}: Error processing file ${file.name}:`, error);
      }
    }
  }

  /**
   * Handle streaming response from Gemini
   */
  handleStreamingResponse(response, logPrefix) {
    const readable = new ReadableStream({
      async start(controller) {
        try {
          const reader = response.body.getReader();
          const decoder = new TextDecoder();

          while (true) {
            const { done, value } = await reader.read();
            
            if (done) {
              break;
            }

            const chunk = decoder.decode(value, { stream: true });
            const lines = chunk.split('\n');

            for (const line of lines) {
              if (line.trim() === '') continue;
              
              if (line.startsWith('data: ')) {
                const data = line.slice(6);
                
                if (data === '[DONE]') {
                  controller.enqueue(new TextEncoder().encode('data: [DONE]\n\n'));
                  continue;
                }

                try {
                  // Parse Gemini response and convert to OpenAI format
                  const geminiData = JSON.parse(data);
                  const openaiData = this.convertGeminiToOpenAIFormat(geminiData);
                  const modifiedData = JSON.stringify(openaiData);
                  controller.enqueue(new TextEncoder().encode(`data: ${modifiedData}\n\n`));
                } catch (error) {
                  // If parsing fails, pass through as-is
                  controller.enqueue(new TextEncoder().encode(`${line}\n`));
                }
              } else {
                controller.enqueue(new TextEncoder().encode(`${line}\n`));
              }
            }
          }
        } catch (error) {
          logger.error(`${logPrefix}: Streaming error:`, error);
          controller.error(error);
        } finally {
          controller.close();
        }
      }
    });

    return new Response(readable, {
      status: response.status,
      headers: {
        'Content-Type': 'text/event-stream',
        'Cache-Control': 'no-cache',
        'Connection': 'keep-alive'
      }
    });
  }

  /**
   * Convert Gemini response to OpenAI format
   */
  convertGeminiToOpenAIFormat(geminiData) {
    // Basic conversion from Gemini streaming format to OpenAI format
    if (geminiData.candidates && geminiData.candidates.length > 0) {
      const candidate = geminiData.candidates[0];
      
      if (candidate.content && candidate.content.parts) {
        const text = candidate.content.parts
          .filter(part => part.text)
          .map(part => part.text)
          .join('');

        return {
          id: `chatcmpl-${Date.now()}`,
          object: 'chat.completion.chunk',
          created: Math.floor(Date.now() / 1000),
          model: 'gemini-pro',
          choices: [{
            index: 0,
            delta: {
              content: text
            },
            finish_reason: candidate.finishReason === 'STOP' ? 'stop' : null
          }]
        };
      }
    }

    // Return original data if conversion fails
    return geminiData;
  }

  /**
   * Utility functions
   */
  arrayBufferToBase64(buffer) {
    const bytes = new Uint8Array(buffer);
    let binary = '';
    for (let i = 0; i < bytes.byteLength; i++) {
      binary += String.fromCharCode(bytes[i]);
    }
    return btoa(binary);
  }

  cleanupMarkdown(text) {
    if (typeof text !== 'string') {
      return '';
    }

    // Replace escaped newlines with actual newlines
    text = text.replace(/\\n/g, '\n');

    // Clean up consecutive empty lines
    text = text.replace(/\n\s*\n\s*\n+/g, '\n\n');

    // Remove lines that contain only whitespace
    const lines = text.split('\n');
    const cleanedLines = [];
    
    for (const line of lines) {
      if (line.trim() || (cleanedLines.length > 0 && cleanedLines[cleanedLines.length - 1].trim())) {
        cleanedLines.append(line);
      }
    }

    // Remove trailing empty lines
    while (cleanedLines.length > 0 && !cleanedLines[cleanedLines.length - 1].trim()) {
      cleanedLines.pop();
    }

    return cleanedLines.join('\n');
  }
}