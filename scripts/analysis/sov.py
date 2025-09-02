import re
from typing import Dict, List, Any, Tuple
import numpy as np
import pandas as pd


def build_brand_patterns(brands: List[str]) -> Dict[str, re.Pattern]:
    """Build regex patterns for brand mentions incl. spacing/hyphen variants."""
    patterns: Dict[str, re.Pattern] = {}
    for b in brands:
        # Normalize brand to tokens and allow optional spaces/hyphens between tokens
        tokens = re.split(r"\s+", b.strip())
        if len(tokens) == 1:
            # e.g., "Atomberg" -> allow "atom-berg" by injecting optional hyphen/space variants
            word = tokens[0]
            # Split camel-case variations? Keep simple and case-insensitive
            patt = rf"\b{re.escape(word)}\b|{re.escape(word[:5])}\s*-\s*{re.escape(word[5:])}" if len(word) > 5 else rf"\b{re.escape(word)}\b"
        else:
            join = r"[\s\-]*".join([re.escape(t) for t in tokens])
            patt = rf"\b{join}\b"
        patterns[b] = re.compile(patt, flags=re.IGNORECASE)
    return patterns


def extract_brand_mentions(text: str, patterns: Dict[str, re.Pattern]) -> Dict[str, int]:
    counts: Dict[str, int] = {b: 0 for b in patterns.keys()}
    if not text:
        return counts
    for brand, patt in patterns.items():
        matches = patt.findall(text)
        counts[brand] = len(matches) if matches else 0
    return counts


def compute_google_ctr_weight(position: int) -> float:
    """Approx CTR curve weight for Google organic positions 1..10; decay afterward."""
    ctr = {
        1: 0.31, 2: 0.15, 3: 0.10, 4: 0.07, 5: 0.05,
        6: 0.04, 7: 0.03, 8: 0.02, 9: 0.015, 10: 0.01
    }
    if position in ctr:
        return ctr[position]
    # Mild decay for >10
    return max(0.005, 0.01 * (0.9 ** (position - 10)))


def compute_youtube_engagement_weight(views: int, likes: int, comments: int) -> float:
    # Log-scaled to reduce outlier influence
    return np.log10(max(1, views) + 1) + 0.5 * np.log10(max(1, likes) + 1) + 0.5 * np.log10(max(1, comments) + 1)


def compute_x_engagement_weight(likes: int, retweets: int, replies: int) -> float:
    return np.log10(max(1, likes + retweets + replies) + 1)


def _safe_div(a: float, b: float) -> float:
    return float(a) / float(b) if b else 0.0


def aggregate_sov_metrics(df_mentions: pd.DataFrame, brands: List[str]) -> Dict[str, Any]:
    """
    Compute:
      - mention_share: share of raw mentions
      - weighted_sov: share of weighted attention (weight * mentions)
      - positive_voice_share: share of positive weighted attention (compound >= threshold assumed in df_mentions)
    Returns dict with 'overall' and 'by_platform' groupings.
    """
    if df_mentions.empty:
        return {}

    # Overall aggregations
    overall = {}
    # Raw mentions
    total_mentions = df_mentions["mentions"].sum()
    overall_mentions = {
        b: df_mentions.loc[df_mentions["brand"] == b, "mentions"].sum() for b in brands
    }
    overall["mention_share"] = {b: _safe_div(overall_mentions[b], total_mentions) for b in brands}

    # Weighted SoV
    df_mentions["weighted_mentions"] = df_mentions["mentions"] * df_mentions["weight"]
    total_weighted = df_mentions["weighted_mentions"].sum()
    weighted_sov = {
        b: df_mentions.loc[df_mentions["brand"] == b, "weighted_mentions"].sum() for b in brands
    }
    overall["weighted_sov"] = {b: _safe_div(weighted_sov[b], total_weighted) for b in brands}

    # Positive Voice Share
    df_positive = df_mentions[df_mentions["positive_flag"] == 1].copy()
    df_positive["weighted_mentions_pos"] = df_positive["mentions"] * df_positive["weight"]
    total_weighted_pos = df_positive["weighted_mentions_pos"].sum()
    pos_sov = {
        b: df_positive.loc[df_positive["brand"] == b, "weighted_mentions_pos"].sum() for b in brands
    }
    overall["positive_voice_share"] = {b: _safe_div(pos_sov[b], total_weighted_pos) for b in brands}

    # By platform
    by_platform = {}
    for plat, g in df_mentions.groupby("platform"):
        res = {}
        tot_m = g["mentions"].sum()
        res["mention_share"] = {b: _safe_div(g.loc[g["brand"] == b, "mentions"].sum(), tot_m) for b in brands}
        g = g.copy()
        g["weighted_mentions"] = g["mentions"] * g["weight"]
        tot_w = g["weighted_mentions"].sum()
        res["weighted_sov"] = {b: _safe_div(g.loc[g["brand"] == b, "weighted_mentions"].sum(), tot_w) for b in brands}
        gp = g[g["positive_flag"] == 1].copy()
        gp["weighted_mentions_pos"] = gp["mentions"] * gp["weight"]
        tot_wp = gp["weighted_mentions_pos"].sum()
        res["positive_voice_share"] = {
            b: _safe_div(gp.loc[gp["brand"] == b, "weighted_mentions_pos"].sum(), tot_wp) for b in brands
        }
        by_platform[plat] = res

    return {"overall": overall, "by_platform": by_platform}


def summarize_insights(
    df_items: pd.DataFrame,
    df_mentions: pd.DataFrame,
    focus_brand: str,
    competitors: List[str],
) -> str:
    """Generate simple, actionable insights for content and marketing."""
    if df_items.empty or df_mentions.empty:
        return "No insights available — try increasing N, adding platforms, or verifying API keys."

    lines: List[str] = []

    # Top platforms for focus brand by weighted attention
    dfm = df_mentions.copy()
    dfm["weighted_mentions"] = dfm["mentions"] * dfm["weight"]
    focus = dfm[dfm["brand"].str.lower() == focus_brand.lower()]
    plat_scores = focus.groupby("platform")["weighted_mentions"].sum().sort_values(ascending=False)
    if not plat_scores.empty:
        top_plat = plat_scores.index.tolist()[0]
        lines.append(f"- Atomberg gains the most weighted attention on {top_plat}. Double-down with targeted content cadence there.")

    # Content formats (YouTube channel titles as proxy)
    if "youtube" in df_items["platform"].unique():
        yt = df_items[(df_items["platform"] == "youtube") & (df_items.get(f"mentions_{focus_brand}", 0) > 0)]
        if not yt.empty and "channelTitle" in yt.columns:
            top_channels = yt.groupby("channelTitle")["weight"].sum().sort_values(ascending=False).head(5)
            if not top_channels.empty:
                ch_list = ", ".join([f"{c}" for c in top_channels.index.tolist()])
                lines.append(f"- YouTube channels driving Atomberg mentions: {ch_list}. Consider outreach for collaborations and reviews.")

        # Topic cues from video titles
        titles = (yt["title"].dropna().tolist()) if not yt.empty else []
        if titles:
            keywords = _top_keywords_from_texts(titles, top_k=10)
            if keywords:
                lines.append(f"- High-performing YouTube title cues: {', '.join(keywords)}. Try A/B testing titles around these themes.")

    # Competitive gaps: which competitor dominates where Atomberg is weak
    comp = dfm[dfm["brand"].str.lower().isin([c.lower() for c in competitors])]
    comp_plat = comp.groupby(["platform", "brand"])["weighted_mentions"].sum().reset_index()
    if not comp_plat.empty:
        top_comp = comp_plat.sort_values("weighted_mentions", ascending=False).head(3)
        bullets = [f"{r['brand']} on {r['platform']}" for _, r in top_comp.iterrows()]
        lines.append(f"- Competitive hotspots: {', '.join(bullets)}. Consider responsive content and SEO to reclaim visibility.")

    # Positive voice – where to amplify
    df_pos = dfm[(dfm["brand"].str.lower() == focus_brand.lower()) & (dfm["positive_flag"] == 1)]
    if not df_pos.empty:
        pos_plat = df_pos.groupby("platform")["weighted_mentions"].sum().sort_values(ascending=False)
        if not pos_plat.empty:
            best = pos_plat.index.tolist()[0]
            lines.append(f"- Share of Positive Voice is strongest on {best}. Amplify UGC and testimonials there.")

    if not lines:
        return "Directional insight: Even distribution across platforms. Increase content volume and diversify formats to find breakout channels."
    return "\n".join(lines)


def _top_keywords_from_texts(texts: List[str], top_k: int = 10) -> List[str]:
    import re
    from collections import Counter

    tokens = []
    for t in texts:
        toks = re.findall(r"[A-Za-z]{3,}", t.lower())
        tokens.extend(toks)
    # Lightweight stopword list
    stops = set("""the a and for with from your this that are was were has have into over under about into smart fan fans bldc wifi energy efficient
                   atomberg havells crompton orient electric usha bajaj panasonic luminous dyson review reviews""".split())
    toks = [t for t in tokens if t not in stops]
    common = Counter(toks).most_common(top_k)
    return [w for w, _ in common]
