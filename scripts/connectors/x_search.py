from typing import List, Dict, Any

def x_search(query: str, limit: int = 100, lang: str = "en") -> List[Dict[str, Any]]:
    """
    Fetch recent posts from X (Twitter) using snscrape (no API key).
    Returns normalized dicts with platform='x'.

    Note: Requires 'snscrape' package. If not installed, raises ImportError.
    """
    try:
        import snscrape.modules.twitter as sntwitter
    except Exception as e:
        raise RuntimeError("snscrape not installed. Add 'snscrape' to requirements to enable X.") from e

    results: List[Dict[str, Any]] = []
    # Use a simple query; you can customize e.g. 'lang:en'
    query_full = f'{query} lang:{lang}'
    scraper = sntwitter.TwitterSearchScraper(query_full)
    count = 0
    for tweet in scraper.get_items():
        # Limit early
        if count >= limit:
            break
        # Some tweets might be long; we capture core engagement
        results.append(
            {
                "platform": "x",
                "content": tweet.content,
                "username": getattr(tweet.user, "username", None),
                "url": tweet.url,
                "likeCount": tweet.likeCount or 0,
                "retweetCount": tweet.retweetCount or 0,
                "replyCount": tweet.replyCount or 0,
            }
        )
        count += 1

    return results
