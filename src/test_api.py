import pytest
from fastapi.testclient import TestClient

from sentiment_model import train_sentiment_model


@pytest.fixture(scope="session", autouse=True)
def trained_model():
    train_sentiment_model()


@pytest.fixture
def client():
    from api import app
    return TestClient(app)


def test_health(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_analyze_single_review(client):
    response = client.post("/analyze", json={"text": "This product is amazing and works great!"})
    assert response.status_code == 200
    data = response.json()
    assert data["sentiment"] in {"positive", "neutral", "negative"}
    assert 0 <= data["confidence"] <= 1


def test_analyze_rejects_empty_text(client):
    response = client.post("/analyze", json={"text": ""})
    assert response.status_code == 422


def test_analyze_batch(client):
    response = client.post("/analyze/batch", json={
        "texts": [
            "The product quality is excellent and I love it.",
            "Terrible experience, item arrived broken.",
        ],
        "n_topics": 2,
    })
    assert response.status_code == 200
    data = response.json()
    assert len(data["results"]) == 2
    assert data["sentiment_stats"]["total_reviews"] == 2
    assert len(data["top_keywords"]) > 0
    assert len(data["topic_clusters"]) == 2


def test_analyze_batch_caps_topics_to_input_size(client):
    response = client.post("/analyze/batch", json={
        "texts": ["Just one review here."],
        "n_topics": 5,
    })
    assert response.status_code == 200
    assert len(response.json()["topic_clusters"]) == 1
