import os
import time
import math
import re
from typing import List, Dict, Any, Tuple
import streamlit as st
import pandas as pd
import numpy as np
import altair as alt

from analysis.sentiment import get_sentiment_analyzer, score_text
from analysis.sov import (
    build_brand_patterns,
    extract_brand_mentions,
    compute_google_ctr_weight,
    compute_youtube_engagement_weight,
    compute_x_engagement_weight,
    aggregate_sov_metrics,
    summarize_insights,
)
from connectors.google_search import google_search
from connectors.youtube_search import youtube_search
from connectors.x_search import x_search


# -------------- Streamlit Page Config --------------
st.set_page_config(
    page_title="Smart Fan SoV Intelligence",
    page_icon="🧠",
    layout="wide",
)

st.title("🧠 Smart Fan Share of Voice Intelligence")
st.caption("Python backend + Streamlit UI | Platforms: Google Web, YouTube, X (optional)")

# -------------- Sidebar Controls --------------
with st.sidebar:
    st.header("Settings")

    # API keys
    st.subheader("API Keys")
    serpapi_key = st.text_input("SerpAPI API Key (Google Web)", type="password")
    youtube_api_key = st.text_input("YouTube Data API Key", type="password")
    st.caption("Note: X runs via snscrape and does not require an API key.")

    st.subheader("Platforms")
    use_google = st.checkbox("Google Web (SerpAPI)", value=True)
    use_youtube = st.checkbox("YouTube", value=True)
    use_x = st.checkbox("X (Twitter) via snscrape", value=False)

    st.subheader("Keywords")
    default_keywords = "smart fan, smart ceiling fan, BLDC fan, wifi fan, energy efficient fan"
    keywords_raw = st.text_area(
        "Enter keywords (comma separated)",
        value=default_keywords,
        height=100,
    )
    keywords = [k.strip() for k in keywords_raw.split(",") if k.strip()]

    st.subheader("Brands")
    focus_brand = st.text_input("Focus brand", value="Atomberg")
    competitors_raw = st.text_area(
        "Competitors (comma separated)",
        value="Havells, Crompton, Orient Electric, Usha, Bajaj, Panasonic, Luminous, Dyson",
        height=100,
    )
    competitors = [c.strip() for c in competitors_raw.split(",") if c.strip()]
    brands = [focus_brand] + [b for b in competitors if b.lower() != focus_brand.lower()]

    st.subheader("Limits")
    n_results = st.slider("Top N results per platform per keyword", min_value=5, max_value=50, value=20, step=5)
    yt_comments_per_video = st.slider("YouTube: comments per video", min_value=0, max_value=100, value=20, step=10)
    x_posts_limit = st.slider("X: posts per keyword", min_value=10, max_value=200, value=100, step=10)

    st.subheader("Filters")
    yt_region = st.selectbox("YouTube region code", options=["IN", "US", "GB", "AE", "SG", "AU", "CA"], index=0)
    lang = st.selectbox("Search language", options=["en", "hi", "bn", "ta", "te"], index=0)
    positive_threshold = st.slider("Positive sentiment threshold (VADER compound)", 0.0, 1.0, 0.3, 0.05)

    run_btn = st.button("Run Analysis", type="primary")


# -------------- Helpers --------------
@st.cache_data(show_spinner=False)
def _brand_patterns_cached(brands: List[str]) -> Dict[str, re.Pattern]:
    return build_brand_patterns(brands)

@st.cache_data(show_spinner=False)
def _get_vader():
    return get_sentiment_analyzer()


def fetch_platform_data_for_keyword(
    keyword: str,
    brands: List[str],
    serpapi_key: str,
    youtube_api_key: str,
    use_google: bool,
    use_youtube: bool,
    use_x: bool,
    n_results: int,
    yt_comments_per_video: int,
    yt_region: str,
    x_posts_limit: int,
    lang: str,
) -> List[Dict[str, Any]]:
    """Fetch results from selected platforms for a single keyword with safe fallbacks."""
    results: List[Dict[str, Any]] = []

    if use_google:
        if not serpapi_key:
            st.warning("Google Web selected but SerpAPI API key missing. Skipping Google.")
        else:
            try:
                g = google_search(
                    query=keyword,
                    api_key=serpapi_key,
                    n_results=n_results,
                    hl=lang,
                    gl="in",
                )
                results.extend(g)
            except Exception as e:
                st.error(f"Google fetch error for '{keyword}': {e}")

    if use_youtube:
        if not youtube_api_key:
            st.warning("YouTube selected but API key missing. Skipping YouTube.")
        else:
            try:
                y = youtube_search(
                    query=keyword,
                    api_key=youtube_api_key,
                    max_results=n_results,
                    region_code=yt_region,
                    comments_limit=yt_comments_per_video,
                    language=lang,
                )
                results.extend(y)
            except Exception as e:
                st.error(f"YouTube fetch error for '{keyword}': {e}")

    if use_x:
        try:
            xres = x_search(
                query=keyword,
                limit=x_posts_limit,
                lang=lang,
            )
            results.extend(xres)
        except Exception as e:
            st.error(f"X fetch error for '{keyword}': {e}")

    return results


def analyze_results(
    rows: List[Dict[str, Any]],
    brands: List[str],
    positive_threshold: float,
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    """Annotate rows with mentions, sentiment, weights; return annotated df, per-item mentions df, and aggregate report."""
    if not rows:
        return pd.DataFrame(), pd.DataFrame(), {}

    patterns = _brand_patterns_cached(brands)
    vader = _get_vader()

    enriched: List[Dict[str, Any]] = []
    mentions_records: List[Dict[str, Any]] = []

    for r in rows:
        text_chunks = [
            r.get("title", ""),
            r.get("snippet", ""),
            r.get("description", ""),
            r.get("comments_text", ""),
            r.get("content", ""),
        ]
        full_text = "\n".join([t for t in text_chunks if t])

        # Brand mention counts
        brand_counts = extract_brand_mentions(full_text, patterns)

        # Sentiment
        sent = score_text(vader, full_text)
        compound = sent.get("compound", 0.0)

        # Platform-specific weight
        platform = r.get("platform")
        weight = 1.0
        if platform == "google":
            pos_idx = r.get("position", 1)
            weight = compute_google_ctr_weight(pos_idx)
        elif platform == "youtube":
            views = int(r.get("viewCount", 0))
            likes = int(r.get("likeCount", 0))
            comments = int(r.get("commentCount", 0))
            weight = compute_youtube_engagement_weight(views, likes, comments)
        elif platform == "x":
            likes = int(r.get("likeCount", 0))
            retweets = int(r.get("retweetCount", 0))
            replies = int(r.get("replyCount", 0))
            weight = compute_x_engagement_weight(likes, retweets, replies)

        row_out = dict(r)
        row_out.update(
            {
                "full_text": full_text,
                "sentiment_compound": compound,
                "sentiment_pos": sent.get("pos", 0.0),
                "sentiment_neg": sent.get("neg", 0.0),
                "weight": weight,
            }
        )
        # Add brand counts as columns
        for b in brands:
            row_out[f"mentions_{b}"] = brand_counts.get(b, 0)
        enriched.append(row_out)

        # Mentions record per brand for tidy analysis
        total_mentions = sum(brand_counts.values())
        for b, cnt in brand_counts.items():
            positive_flag = 1 if compound >= positive_threshold else 0
            mentions_records.append(
                {
                    "platform": platform,
                    "keyword": r.get("keyword"),
                    "brand": b,
                    "mentions": cnt,
                    "total_mentions_in_item": total_mentions,
                    "weight": weight,
                    "compound": compound,
                    "positive_flag": positive_flag,
                }
            )

    df_items = pd.DataFrame(enriched)
    df_mentions = pd.DataFrame(mentions_records)

    agg = aggregate_sov_metrics(df_mentions, brands)
    return df_items, df_mentions, agg


# -------------- Run Pipeline --------------
if run_btn:
    if not keywords:
        st.error("Please add at least one keyword.")
        st.stop()
    if not (use_google or use_youtube or use_x):
        st.error("Please select at least one platform.")
        st.stop()

    prog = st.progress(0.0, text="Fetching data...")
    all_rows: List[Dict[str, Any]] = []
    total_steps = len(keywords)
    for i, kw in enumerate(keywords, 1):
        rows = fetch_platform_data_for_keyword(
            keyword=kw,
            brands=brands,
            serpapi_key=serpapi_key,
            youtube_api_key=youtube_api_key,
            use_google=use_google,
            use_youtube=use_youtube,
            use_x=use_x,
            n_results=n_results,
            yt_comments_per_video=yt_comments_per_video,
            yt_region=yt_region,
            x_posts_limit=x_posts_limit,
            lang=lang,
        )
        # Add keyword onto each row
        for r in rows:
            r["keyword"] = kw
        all_rows.extend(rows)
        prog.progress(i / total_steps, text=f"Fetched {i}/{total_steps} keywords")

    st.success(f"Fetched {len(all_rows)} items across {len(keywords)} keywords.")

    df_items, df_mentions, agg = analyze_results(all_rows, brands, positive_threshold)

    if df_items.empty:
        st.warning("No data fetched. Check API keys and platform selections.")
        st.stop()

    # -------------- KPIs --------------
    st.subheader("Key Metrics")

    if agg:
        kpi_cols = st.columns(3)
        atomberg_name = focus_brand
        atomberg_sov = agg["overall"]["weighted_sov"].get(atomberg_name, 0.0) * 100
        atomberg_pos = agg["overall"]["positive_voice_share"].get(atomberg_name, 0.0) * 100
        atomberg_mentions = agg["overall"]["mention_share"].get(atomberg_name, 0.0) * 100

        kpi_cols[0].metric("Weighted SoV (Atomberg)", f"{atomberg_sov:.1f}%")
        kpi_cols[1].metric("Share of Positive Voice (Atomberg)", f"{atomberg_pos:.1f}%")
        kpi_cols[2].metric("Mention Share (Atomberg)", f"{atomberg_mentions:.1f}%")
    else:
        st.info("Not enough mention data to compute SoV metrics.")

    # -------------- Charts --------------
    def dict_to_df(d: Dict[str, float], metric: str) -> pd.DataFrame:
        return pd.DataFrame({"brand": list(d.keys()), metric: list(d.values())})

    if agg:
        st.subheader("Overall SoV Comparison")

        overall = agg["overall"]
        sov_df = dict_to_df(overall["weighted_sov"], "weighted_sov")
        sov_df["weighted_sov_pct"] = sov_df["weighted_sov"] * 100

        pos_df = dict_to_df(overall["positive_voice_share"], "positive_voice_share")
        pos_df["positive_voice_share_pct"] = pos_df["positive_voice_share"] * 100

        mention_df = dict_to_df(overall["mention_share"], "mention_share")
        mention_df["mention_share_pct"] = mention_df["mention_share"] * 100

        chart_cols = st.columns(3)

        c1 = (
            alt.Chart(sov_df)
            .mark_bar()
            .encode(
                x=alt.X("brand:N", sort="-y"),
                y=alt.Y("weighted_sov_pct:Q", title="Weighted SoV (%)"),
                color=alt.condition(
                    alt.FieldOneOfPredicate(field="brand", oneOf=[focus_brand]),
                    alt.value("#1f77b4"),
                    alt.value("#999999"),
                ),
                tooltip=["brand", alt.Tooltip("weighted_sov_pct:Q", format=".1f")],
            )
            .properties(height=300)
        )
        chart_cols[0].altair_chart(c1, use_container_width=True)

        c2 = (
            alt.Chart(pos_df)
            .mark_bar()
            .encode(
                x=alt.X("brand:N", sort="-y"),
                y=alt.Y("positive_voice_share_pct:Q", title="Share of Positive Voice (%)"),
                color=alt.condition(
                    alt.FieldOneOfPredicate(field="brand", oneOf=[focus_brand]),
                    alt.value("#1f77b4"),
                    alt.value("#999999"),
                ),
                tooltip=["brand", alt.Tooltip("positive_voice_share_pct:Q", format=".1f")],
            )
            .properties(height=300)
        )
        chart_cols[1].altair_chart(c2, use_container_width=True)

        c3 = (
            alt.Chart(mention_df)
            .mark_bar()
            .encode(
                x=alt.X("brand:N", sort="-y"),
                y=alt.Y("mention_share_pct:Q", title="Mention Share (%)"),
                color=alt.condition(
                    alt.FieldOneOfPredicate(field="brand", oneOf=[focus_brand]),
                    alt.value("#1f77b4"),
                    alt.value("#999999"),
                ),
                tooltip=["brand", alt.Tooltip("mention_share_pct:Q", format=".1f")],
            )
            .properties(height=300)
        )
        chart_cols[2].altair_chart(c3, use_container_width=True)

    # -------------- Tables --------------
    st.subheader("Fetched Items (Annotated)")
    display_cols = [
        "platform", "keyword", "title", "url", "channelTitle", "username", "position",
        "viewCount", "likeCount", "commentCount", "retweetCount", "replyCount",
        "sentiment_compound", "weight",
    ]
    for b in brands:
        display_cols.append(f"mentions_{b}")

    st.dataframe(df_items[ [c for c in display_cols if c in df_items.columns] ].sort_values("weight", ascending=False), use_container_width=True)

    # -------------- Drilldowns --------------
    st.subheader(f"Top Atomberg Posts by Attention")
    atomberg_mask = df_items.get(f"mentions_{focus_brand}", pd.Series([0]*len(df_items))) > 0
    df_atomberg = df_items[atomberg_mask].copy()
    if not df_atomberg.empty:
        df_atomberg["attention"] = df_atomberg["weight"]
        st.dataframe(
            df_atomberg[
                [c for c in ["platform", "keyword", "title", "url", "channelTitle", "username", "attention", "sentiment_compound", "viewCount", "likeCount", "commentCount"] if c in df_atomberg.columns]
            ].sort_values("attention", ascending=False).head(30),
            use_container_width=True,
        )
    else:
        st.info("No Atomberg mentions found in fetched items.")

    # -------------- Insights & Recommendations --------------
    st.subheader("Insights & Recommendations")
    insights = summarize_insights(df_items, df_mentions, focus_brand, competitors)
    st.write(insights)

    # -------------- Export --------------
    st.subheader("Export")
    exp1, exp2 = st.columns(2)
    with exp1:
        st.download_button(
            label="Download Items CSV",
            data=df_items.to_csv(index=False).encode("utf-8"),
            file_name="sov_items.csv",
            mime="text/csv",
        )
    with exp2:
        st.download_button(
            label="Download Mentions CSV",
            data=df_mentions.to_csv(index=False).encode("utf-8"),
            file_name="sov_mentions.csv",
            mime="text/csv",
        )

else:
    st.info("Configure your settings in the sidebar and click 'Run Analysis' to begin.")
