import os
import joblib
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.pipeline import Pipeline

from preprocessing import TextPreprocessor, load_labeled_data, load_config

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def train_sentiment_model():
    config = load_config()
    model_path = os.path.join(BASE_DIR, config["models"]["sentiment_model_path"])

    df = load_labeled_data()
    # Expect columns: "review_text" and "sentiment" (e.g., positive/neutral/negative)
    X = df["review_text"]
    y = df["sentiment"]

    preprocessor = TextPreprocessor()

    def preprocess_series(series):
        return series.apply(preprocessor.clean_text)

    X_clean = preprocess_series(X)

    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer()),
        ("clf", LogisticRegression(max_iter=1000))  # <-- multi_class removed
    ])

    X_train, X_test, y_train, y_test = train_test_split(
        X_clean, y, test_size=0.2, random_state=42, stratify=y
    )

    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)

    print("Sentiment model report:\n")
    print(classification_report(y_test, y_pred))

    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    joblib.dump(pipeline, model_path)
    print(f"✅ Sentiment model saved to {model_path}")


def load_sentiment_model():
    config = load_config()
    model_path = os.path.join(BASE_DIR, config["models"]["sentiment_model_path"])
    return joblib.load(model_path)
