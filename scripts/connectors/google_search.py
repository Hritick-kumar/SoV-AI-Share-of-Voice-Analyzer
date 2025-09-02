import requests
from typing import List, Dict, Any, Optional


def google_search(
    query: str,
    api_key: str,
    n_results: int = 20,
    hl: str = "en",
    gl: str = "in",
) -> List[Dict[str, Any]]:
    """
    Use SerpAPI to fetch Google Web search results.
    Returns a list of normalized dicts:
    - platform='google'
    - title, url, snippet, position
    """
    # SerpAPI: https://serpapi.com/search
    url = "https://serpapi.com/search.json"
    params = {
        "engine": "google",
        "q": query,
        "num": min(n_results, 100),
        "api_key": api_key,
        "hl": hl,
        "gl": gl,
    }
    resp = requests.get(url, params=params, timeout=30)
    resp.raise_for_status()
    data = resp.json()

    results = []
    organic = data.get("organic_results", []) or []
    for item in organic[:n_results]:
        results.append(
            {
                "platform": "google",
                "title": item.get("title"),
                "url": item.get("link"),
                "snippet": item.get("snippet"),
                "position": item.get("position") or item.get("rank") or 0,
            }
        )

    return results
