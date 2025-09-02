from typing import Dict
import nltk

# Ensure the VADER lexicon is available at runtime
def get_sentiment_analyzer():
    try:
        from nltk.sentiment import SentimentIntensityAnalyzer
    except Exception:
        nltk.download("vader_lexicon")
        from nltk.sentiment import SentimentIntensityAnalyzer

    # Ensure lexicon is present
    try:
        nltk.data.find("sentiment/vader_lexicon.zip")
    except LookupError:
        nltk.download("vader_lexicon")
    return SentimentIntensityAnalyzer()


def score_text(analyzer, text: str) -> Dict[str, float]:
    if not text:
        return {"neg": 0.0, "neu": 0.0, "pos": 0.0, "compound": 0.0}
    return analyzer.polarity_scores(text)
