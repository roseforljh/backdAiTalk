/**
 * Web search service for Cloudflare Workers
 * Matches backend-docker web search functionality
 */

import { logger } from '../utils/logger';

export class WebSearchService {
  constructor(env) {
    this.env = env;
    this.googleApiKey = env.GOOGLE_API_KEY;
    this.googleCseId = env.GOOGLE_CSE_ID;
    this.searchResultCount = parseInt(env.SEARCH_RESULT_COUNT || '5');
    this.snippetMaxLength = parseInt(env.SEARCH_SNIPPET_MAX_LENGTH || '200');
  }

  /**
   * Perform web search using Google Custom Search API
   */
  async performWebSearch(query, requestId) {
    const logPrefix = `RID-${requestId}`;
    logger.info(`${logPrefix}: perform_web_search called. Query: '${query}'. API Key set: ${!!this.googleApiKey}, CSE ID set: ${!!this.googleCseId}`);

    const results = [];

    if (!this.googleApiKey || !this.googleCseId) {
      logger.warn(`${logPrefix}: Web search skipped. GOOGLE_API_KEY or GOOGLE_CSE_ID not set in environment variables.`);
      logger.warn(`${logPrefix}: To enable web search, please set GOOGLE_API_KEY and GOOGLE_CSE_ID in Cloudflare Workers environment variables.`);
      return results;
    }

    if (!query || query.trim() === '') {
      logger.warn(`${logPrefix}: Web search skipped, query is empty.`);
      return results;
    }

    try {
      const searchUrl = new URL('https://www.googleapis.com/customsearch/v1');
      searchUrl.searchParams.set('key', this.googleApiKey);
      searchUrl.searchParams.set('cx', this.googleCseId);
      searchUrl.searchParams.set('q', query);
      searchUrl.searchParams.set('num', Math.min(this.searchResultCount, 10).toString());

      logger.info(`${logPrefix}: Performing web search for query: '${query.substring(0, 100)}'`);

      const response = await fetch(searchUrl.toString());
      
      if (!response.ok) {
        const errorText = await response.text();
        logger.error(`${logPrefix}: Google Custom Search API error: ${response.status} - ${errorText}`);
        return results;
      }

      const data = await response.json();
      const items = data.items || [];

      for (let i = 0; i < items.length; i++) {
        const item = items[i];
        let snippet = (item.snippet || 'N/A').replace(/\n/g, ' ').trim();
        
        if (snippet.length > this.snippetMaxLength) {
          snippet = snippet.substring(0, this.snippetMaxLength) + '...';
        }

        results.push({
          index: i + 1,
          title: (item.title || 'N/A').trim(),
          href: item.link || 'N/A',
          snippet: snippet
        });
      }

      logger.info(`${logPrefix}: Web search completed, found ${results.length} results.`);

    } catch (error) {
      logger.error(`${logPrefix}: Web search error:`, error);
    }

    return results;
  }

  /**
   * Generate search context message content
   */
  generateSearchContextMessageContent(searchResults, query, requestId) {
    const logPrefix = `RID-${requestId}`;
    
    if (!searchResults || searchResults.length === 0) {
      logger.info(`${logPrefix}: No search results to include in context.`);
      return '';
    }

    logger.info(`${logPrefix}: Generating search context with ${searchResults.length} results.`);

    let contextContent = `\n\n--- Web Search Results for "${query}" ---\n`;
    
    for (const result of searchResults) {
      contextContent += `\n${result.index}. **${result.title}**\n`;
      contextContent += `   URL: ${result.href}\n`;
      contextContent += `   Summary: ${result.snippet}\n`;
    }
    
    contextContent += '\n--- End of Web Search Results ---\n\n';
    contextContent += 'Please use the above search results to provide accurate and up-to-date information in your response. ';
    contextContent += 'Cite the sources when appropriate using the provided URLs.\n\n';

    return contextContent;
  }

  /**
   * Check if web search should be performed
   */
  shouldPerformWebSearch(requestData) {
    // Check if web search is explicitly enabled
    if (requestData.use_web_search === true || requestData.qwen_enable_search === true) {
      return true;
    }

    // Check if the query contains search-related keywords
    const lastMessage = requestData.messages[requestData.messages.length - 1];
    if (!lastMessage || lastMessage.role !== 'user') {
      return false;
    }

    const content = lastMessage.content || 
                   (lastMessage.parts && lastMessage.parts.find(p => p.type === 'text_content')?.text) || '';
    
    const searchKeywords = [
      'search for', 'find information', 'latest news', 'current events',
      'what happened', 'recent developments', 'up to date', 'latest update'
    ];

    return searchKeywords.some(keyword => 
      content.toLowerCase().includes(keyword.toLowerCase())
    );
  }
}