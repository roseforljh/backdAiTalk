import asyncio
import logging
import orjson
from typing import List, Dict

from googleapiclient.discovery import build
from googleapiclient.errors import HttpError

from ..core.config import GOOGLE_API_KEY_ENV, GOOGLE_CSE_ID, SEARCH_RESULT_COUNT, SEARCH_SNIPPET_MAX_LENGTH

logger = logging.getLogger("EzTalkProxy.WebSearch")

async def perform_web_search(query: str, rid: str) -> List[Dict[str, str]]:
    logger.info(f"RID-{rid}: perform_web_search called. Query: '{query}'. GOOGLE_API_KEY_ENV is set: {bool(GOOGLE_API_KEY_ENV)}, GOOGLE_CSE_ID is set: {bool(GOOGLE_CSE_ID)}")
    results = []
    actual_google_api_key = GOOGLE_API_KEY_ENV
    if not actual_google_api_key or not GOOGLE_CSE_ID:
        logger.warning(f"RID-{rid}: Web search skipped. GOOGLE_API_KEY or GOOGLE_CSE_ID not set in environment variables.")
        logger.warning(f"RID-{rid}: To enable web search, please create a .env file in the root directory of the 'backdAiTalk' project and add your Google API Key and Custom Search Engine ID.")
        logger.warning(f"RID-{rid}: Example .env file content:\n# Google Custom Search API Key\nGOOGLE_API_KEY=\"YOUR_GOOGLE_API_KEY\"\n# Google Custom Search Engine ID\nGOOGLE_CSE_ID=\"YOUR_GOOGLE_CSE_ID\"")
        return results
    if not query:
        logger.warning(f"RID-{rid}: Web search skipped, query is empty.")
        return results

    try:
        def search_sync():
            service = build("customsearch", "v1", developerKey=actual_google_api_key, cache_discovery=False)
            res = service.cse().list(q=query, cx=GOOGLE_CSE_ID, num=min(SEARCH_RESULT_COUNT, 10)).execute()
            return res.get('items', [])

        logger.info(f"RID-{rid}: Performing web search for query: '{query[:100]}'")
        search_items = await asyncio.to_thread(search_sync)

        for i, item in enumerate(search_items):
            snippet = item.get('snippet', 'N/A').replace('\n', ' ').strip()
            if len(snippet) > SEARCH_SNIPPET_MAX_LENGTH:
                snippet = snippet[:SEARCH_SNIPPET_MAX_LENGTH] + "..."
            results.append({
                "index": i + 1,
                "title": item.get('title', 'N/A').strip(),
                "href": item.get('link', 'N/A'),
                "snippet": snippet
            })
        logger.info(f"RID-{rid}: Web search completed, found {len(results)} results.")

    except HttpError as e:
        err_content = "Unknown Google API error"
        status_code = "N/A"
        if hasattr(e, 'resp') and hasattr(e.resp, 'status'):
            status_code = e.resp.status
        try:
            content_json = orjson.loads(e.content)
            err_detail = content_json.get("error", {})
            err_message = err_detail.get("message", str(e.content))
            err_content = f"{err_message} (Code: {err_detail.get('code', 'N/A')}, Status: {err_detail.get('status', 'N/A')})"
        except:
            err_content = e._get_reason() if hasattr(e, '_get_reason') else e.content.decode(errors='ignore')[:200]
        logger.error(f"RID-{rid}: Google Web Search HttpError (Status: {status_code}): {err_content}")
    except Exception as search_exc:
        logger.error(f"RID-{rid}: Google Web Search failed for query '{query[:50]}': {search_exc}", exc_info=True)
    return results

def generate_search_context_message_content(query: str, search_results: List[Dict[str, str]]) -> str:
    if not search_results:
        return ""
    
    prompt_parts = [
        f"You have been provided with the following web search results for the user's query: '{query}'. "
        "Your task is to synthesize this information, along with your general knowledge, to construct a comprehensive and natural-sounding answer. "
        "It is crucial that you DO NOT include any inline citation marks like [1], [2], [Source 1], etc., directly in your response text. "
        "The user will have a separate way to view the sources if they wish."
    ]

    for i, res in enumerate(search_results):
        source_identifier = res.get('index', i + 1)
        prompt_parts.append(
            f"\nSource {source_identifier}:\n"
            f"  Title: {res.get('title', 'N/A')}\n"
            f"  Snippet: {res.get('snippet', 'N/A')}\n"
            f"  URL: {res.get('href', 'N/A')} (This URL is for your reference only and should not be included in the response)"
        )
    
    prompt_parts.append(
        "\n\nBased on the information from these sources and your existing knowledge, please formulate your answer. "
        "Focus on delivering a clear, accurate, and well-integrated response to the user's query. "
        "Remember, do not insert any citation markers (e.g., [1], [Source 2]) into the body of your answer."
    )
    
    return "\n\n".join(prompt_parts)