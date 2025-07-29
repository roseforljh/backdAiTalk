/**
 * CORS utilities for Cloudflare Workers
 */

export const corsHeaders = {
  'Access-Control-Allow-Origin': '*',
  'Access-Control-Allow-Methods': 'GET, POST, PUT, DELETE, OPTIONS',
  'Access-Control-Allow-Headers': 'Content-Type, Authorization, X-Requested-With, Accept, Origin, User-Agent',
  'Access-Control-Expose-Headers': '*',
  'Access-Control-Allow-Credentials': 'true',
  'Access-Control-Max-Age': '86400'
};

export function handleCORS(request) {
  // Handle CORS preflight requests
  return new Response(null, {
    status: 200,
    headers: corsHeaders
  });
}

export function addCORSHeaders(response) {
  // Add CORS headers to existing response
  const newHeaders = new Headers(response.headers);
  
  Object.entries(corsHeaders).forEach(([key, value]) => {
    newHeaders.set(key, value);
  });

  return new Response(response.body, {
    status: response.status,
    statusText: response.statusText,
    headers: newHeaders
  });
}