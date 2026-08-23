from typing import List, Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from keyword_extraction import extract_top_keywords
from preprocessing import load_config
from sentiment_model import load_sentiment_model
from topic_clustering import cluster_reviews

app = FastAPI(
    title="Text Analytics Suite API",
    description="Sentiment analysis, keyword extraction, and topic clustering for user reviews.",
    version="1.0.0",
)

_model = None


def get_model():
    global _model
    if _model is None:
        try:
            _model = load_sentiment_model()
        except FileNotFoundError:
            raise HTTPException(
                status_code=503,
                detail="Sentiment model not found. Run `python train_sentiment.py` first.",
            )
    return _model


class ReviewIn(BaseModel):
    text: str = Field(..., min_length=1, description="Raw review text to analyze.")


class SentimentOut(BaseModel):
    sentiment: str
    confidence: float


class BatchIn(BaseModel):
    texts: List[str] = Field(..., min_length=1)
    top_n_keywords: Optional[int] = None
    n_topics: Optional[int] = None


class BatchItem(BaseModel):
    text: str
    sentiment: str
    confidence: float


class BatchOut(BaseModel):
    results: List[BatchItem]
    sentiment_stats: dict
    top_keywords: List[List]
    topic_clusters: dict


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/analyze", response_model=SentimentOut)
def analyze(review: ReviewIn):
    clf = get_model()
    probs = clf.predict_proba([review.text])[0]
    labels = clf.classes_
    idx = probs.argmax()
    return SentimentOut(sentiment=labels[idx], confidence=float(probs[idx]))


@app.post("/analyze/batch", response_model=BatchOut)
def analyze_batch(payload: BatchIn):
    clf = get_model()
    config = load_config()

    probs = clf.predict_proba(payload.texts)
    labels = clf.classes_
    pred_indices = probs.argmax(axis=1)
    sentiments = [labels[i] for i in pred_indices]
    confidences = probs.max(axis=1)

    results = [
        BatchItem(text=text, sentiment=sentiment, confidence=float(confidence))
        for text, sentiment, confidence in zip(payload.texts, sentiments, confidences)
    ]

    counts = {}
    for sentiment in sentiments:
        counts[sentiment] = counts.get(sentiment, 0) + 1
    total = len(sentiments)
    sentiment_stats = {"total_reviews": total, **counts}
    for label, count in counts.items():
        sentiment_stats[f"{label}_pct"] = round(100 * count / total, 2)

    top_n = payload.top_n_keywords or config["analysis"]["top_n_keywords"]
    top_keywords = extract_top_keywords(payload.texts, top_n=top_n)

    n_topics = payload.n_topics or config["analysis"]["n_topics"]
    n_topics = max(1, min(n_topics, len(payload.texts)))
    topic_clusters = cluster_reviews(payload.texts, n_topics=n_topics)

    return BatchOut(
        results=results,
        sentiment_stats=sentiment_stats,
        top_keywords=[list(item) for item in top_keywords],
        topic_clusters={str(topic_id): reviews for topic_id, reviews in topic_clusters.items()},
    )
