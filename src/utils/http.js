/**
 * HTTP utilities for making external API calls
 */

import { logger } from './logger';

export class HTTPClient {
  constructor(env) {
    this.env = env;
    this.timeout = parseInt(env.API_TIMEOUT || '30000');
    this.readTimeout = parseInt(env.READ_TIMEOUT || '25000');
  }

  async request(url, options = {}) {
    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), this.timeout);

    try {
      const defaultOptions = {
        method: 'GET',
        headers: {
          'User-Agent': 'EzTalk-Proxy-Worker/1.0.0',
          'Accept': 'application/json',
          'Content-Type': 'application/json'
        },
        signal: controller.signal
      };

      const mergedOptions = {
        ...defaultOptions,
        ...options,
        headers: {
          ...defaultOptions.headers,
          ...options.headers
        }
      };

      logger.debug(`Making ${mergedOptions.method} request to ${url}`);
      
      const response = await fetch(url, mergedOptions);
      
      clearTimeout(timeoutId);
      
      if (!response.ok) {
        logger.warn(`HTTP ${response.status} from ${url}: ${response.statusText}`);
      }
      
      return response;

    } catch (error) {
      clearTimeout(timeoutId);
      
      if (error.name === 'AbortError') {
        logger.error(`Request timeout after ${this.timeout}ms for ${url}`);
        throw new Error(`Request timeout after ${this.timeout}ms`);
      }
      
      logger.error(`HTTP request failed for ${url}:`, error);
      throw error;
    }
  }

  async get(url, headers = {}) {
    return this.request(url, { method: 'GET', headers });
  }

  async post(url, body, headers = {}) {
    return this.request(url, {
      method: 'POST',
      headers,
      body: typeof body === 'string' ? body : JSON.stringify(body)
    });
  }

  async put(url, body, headers = {}) {
    return this.request(url, {
      method: 'PUT',
      headers,
      body: typeof body === 'string' ? body : JSON.stringify(body)
    });
  }

  async delete(url, headers = {}) {
    return this.request(url, { method: 'DELETE', headers });
  }
}