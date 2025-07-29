/**
 * OpenAI compatible handler for processing chat requests
 */

import { logger } from '../utils/logger';
import { HTTPClient } from '../utils/http';

export class OpenAIHandler {
  constructor() {
    this.supportedImageTypes = ['image/jpeg', 'image/png', 'image/gif', 'image/webp'];
    this.audioMimeTypes = [
      'audio/wav', 'audio/x-wav', 'audio/mpeg', 'audio/mp3', 'audio/aac', 'audio/ogg',
      'audio/opus', 'audio/flac', 'audio/3gpp', 'audio/amr', 'audio/aiff', 'audio/x-m4a',
      'audio/midi', 'audio/webm'
    ];
    this.videoMimeTypes = [
      'video/mp4', 'video/mpeg', 'video/quicktime', 'video/x-msvideo', 'video/x-flv',
      'video/x-matroska', 'video/webm', 'video/x-ms-wmv', 'video/3gpp', 'video/x-m4v'
    ];
  }

  /**
   * Handle OpenAI compatible chat request
   */
  async handle(requestData, env, ctx, requestId) {
    const logPrefix = `RID-${requestId}`;
    
    try {
      logger.info(`${logPrefix}: Processing OpenAI compatible request for model ${requestData.model}`);

      // Initialize HTTP client
      const httpClient = new HTTPClient(env);

      // Process uploaded files if any
      if (requestData.uploadedFiles && requestData.uploadedFiles.length > 0) {
        await this.processUploadedFiles(requestData, logPrefix);
      }

      // Prepare the request for the upstream API
      const upstreamRequest = await this.prepareOpenAIRequest(requestData, env, logPrefix);

      // Make the request to upstream API
      const apiUrl = requestData.api_address || requestData.apiAddress || 'https://api.openai.com/v1/chat/completions';
      
      logger.debug(`${logPrefix}: Making request to ${apiUrl}`);

      const response = await httpClient.post(apiUrl, upstreamRequest, {
        'Authorization': `Bearer ${requestData.api_key || requestData.apiKey}`,
        'Content-Type': 'application/json'
      });

      // Handle streaming response
      if (upstreamRequest.stream) {
        return this.handleStreamingResponse(response, logPrefix);
      } else {
        // Handle non-streaming response
        const responseData = await response.json();
        return new Response(JSON.stringify(responseData), {
          status: response.status,
          headers: {
            'Content-Type': 'application/json'
          }
        });
      }

    } catch (error) {
      logger.error(`${logPrefix}: OpenAI handler error:`, error);
      
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
   * Prepare OpenAI compatible request
   */
  async prepareOpenAIRequest(requestData, env, logPrefix) {
    const request = {
      model: requestData.model,
      messages: await this.processMessages(requestData.messages, logPrefix),
      stream: true, // Default to streaming
      temperature: requestData.temperature,
      max_tokens: requestData.max_tokens || requestData.maxTokens,
      top_p: requestData.top_p || requestData.topP
    };

    // Add tools if provided
    if (requestData.tools && requestData.tools.length > 0) {
      request.tools = requestData.tools;
      if (requestData.tool_choice || requestData.toolChoice) {
        request.tool_choice = requestData.tool_choice || requestData.toolChoice;
      }
    }

    // Add custom parameters if provided
    if (requestData.custom_extra_body || requestData.customExtraBody) {
      Object.assign(request, requestData.custom_extra_body || requestData.customExtraBody);
    }

    logger.debug(`${logPrefix}: Prepared OpenAI request with ${request.messages.length} messages`);
    
    return request;
  }

  /**
   * Process messages for OpenAI format
   */
  async processMessages(messages, logPrefix) {
    const processedMessages = [];

    for (const message of messages) {
      if (message.type === 'simple_text_message' || message.message_type === 'simple_text_message') {
        processedMessages.push({
          role: message.role,
          content: message.content
        });
      } else if (message.type === 'parts_message' || message.message_type === 'parts_message') {
        // Handle multimodal messages
        const content = [];
        
        for (const part of message.parts) {
          if (part.type === 'text_content') {
            content.push({
              type: 'text',
              text: part.text
            });
          } else if (part.type === 'inline_data_content') {
            if (this.supportedImageTypes.includes(part.mime_type || part.mimeType)) {
              content.push({
                type: 'image_url',
                image_url: {
                  url: `data:${part.mime_type || part.mimeType};base64,${part.base64_data || part.base64Data}`
                }
              });
            }
          } else if (part.type === 'input_audio_content') {
            // Handle audio content for models that support it
            content.push({
              type: 'input_audio',
              input_audio: {
                data: part.data,
                format: part.format
              }
            });
          }
        }

        processedMessages.push({
          role: message.role,
          content: content.length === 1 && content[0].type === 'text' ? content[0].text : content
        });
      }
    }

    logger.debug(`${logPrefix}: Processed ${processedMessages.length} messages`);
    return processedMessages;
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
          if (this.supportedImageTypes.includes(file.type)) {
            lastMessage.parts.push({
              type: 'inline_data_content',
              base64_data: base64Data,
              mime_type: file.type
            });
          } else if (this.audioMimeTypes.includes(file.type)) {
            lastMessage.parts.push({
              type: 'input_audio_content',
              data: base64Data,
              format: this.getAudioFormatFromMimeType(file.type)
            });
          }
        }

        logger.debug(`${logPrefix}: Processed file ${file.name} (${file.type})`);
      } catch (error) {
        logger.error(`${logPrefix}: Error processing file ${file.name}:`, error);
      }
    }
  }

  /**
   * Handle streaming response
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
                  // Parse and potentially modify the SSE data
                  const parsed = JSON.parse(data);
                  const modifiedData = JSON.stringify(parsed);
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

  getAudioFormatFromMimeType(mimeType) {
    const mimeToFormat = {
      'audio/wav': 'wav',
      'audio/x-wav': 'wav',
      'audio/mpeg': 'mp3',
      'audio/mp3': 'mp3',
      'audio/aac': 'aac',
      'audio/ogg': 'ogg',
      'audio/opus': 'opus',
      'audio/flac': 'flac',
      'audio/3gpp': '3gp',
      'audio/amr': 'amr',
      'audio/aiff': 'aiff',
      'audio/x-m4a': 'm4a',
      'audio/midi': 'midi',
      'audio/webm': 'webm'
    };
    return mimeToFormat[mimeType.toLowerCase()] || mimeType.split('/')[1];
  }

  isGeminiModel(modelName) {
    return modelName && modelName.toLowerCase().includes('gemini');
  }

  supportsMultimodalContent(modelName) {
    return this.isGeminiModel(modelName) || 
           (modelName && (modelName.includes('gpt-4') || modelName.includes('claude')));
  }
}