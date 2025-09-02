import requests
from typing import List, Dict, Any


def youtube_search(
    query: str,
    api_key: str,
    max_results: int = 20,
    region_code: str = "IN",
    comments_limit: int = 20,
    language: str = "en",
) -> List[Dict[str, Any]]:
    """
    Fetch YouTube videos for a query + statistics + top-level comments.
    Returns normalized dicts with platform='youtube'
    """
    search_url = "https://www.googleapis.com/youtube/v3/search"
    search_params = {
        "part": "snippet",
        "q": query,
        "type": "video",
        "maxResults": min(max_results, 50),
        "key": api_key,
        "regionCode": region_code,
        "relevanceLanguage": language,
        "safeSearch": "none",
    }
    s_resp = requests.get(search_url, params=search_params, timeout=30)
    s_resp.raise_for_status()
    s_data = s_resp.json()

    video_ids = [item["id"]["videoId"] for item in s_data.get("items", []) if item.get("id", {}).get("videoId")]
    if not video_ids:
        return []

    # Fetch statistics and details
    videos_url = "https://www.googleapis.com/youtube/v3/videos"
    v_params = {
        "part": "snippet,statistics",
        "id": ",".join(video_ids),
        "key": api_key,
    }
    v_resp = requests.get(videos_url, params=v_params, timeout=30)
    v_resp.raise_for_status()
    v_data = v_resp.json()

    results: List[Dict[str, Any]] = []

    for v in v_data.get("items", []):
        vid = v["id"]
        snippet = v.get("snippet", {})
        stats = v.get("statistics", {})

        # Comments (top-level)
        comments_text = ""
        if comments_limit > 0:
            comments_url = "https://www.googleapis.com/youtube/v3/commentThreads"
            c_params = {
                "part": "snippet",
                "videoId": vid,
                "maxResults": min(comments_limit, 100),
                "key": api_key,
                "textFormat": "plainText",
                "order": "relevance",
            }
            try:
                c_resp = requests.get(comments_url, params=c_params, timeout=30)
                if c_resp.status_code == 200:
                    c_data = c_resp.json()
                    comments = []
                    for c in c_data.get("items", []):
                        top = c.get("snippet", {}).get("topLevelComment", {}).get("snippet", {})
                        if top:
                            comments.append(top.get("textDisplay", "") or top.get("textOriginal", ""))
                    comments_text = "\n".join(comments[:comments_limit])
            except Exception:
                comments_text = ""

        results.append(
            {
                "platform": "youtube",
                "title": snippet.get("title"),
                "description": snippet.get("description"),
                "channelTitle": snippet.get("channelTitle"),
                "url": f"https://www.youtube.com/watch?v={vid}",
                "viewCount": int(stats.get("viewCount", 0)),
                "likeCount": int(stats.get("likeCount", 0)) if "likeCount" in stats else 0,
                "commentCount": int(stats.get("commentCount", 0)),
                "comments_text": comments_text,
            }
        )

    return results
