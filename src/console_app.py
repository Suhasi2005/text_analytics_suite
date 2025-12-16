import os
from collections import Counter

import pandas as pd
import joblib

from preprocessing import load_config
from sentiment_model import load_sentiment_model
from keyword_extraction import extract_top_keywords
from topic_clustering import cluster_reviews
from reporting import generate_summary_report, export_detailed_csv

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Global variable to keep last batch results in memory
LAST_BATCH_ROWS = None
LAST_SENTIMENT_STATS = None
LAST_TOP_KEYWORDS = None
LAST_TOPIC_CLUSTERS = None

def analyze_single_review():
    clf = load_sentiment_model()
    text = input("\nEnter a review: ").strip()
    if not text:
        print("❌ Empty input.")
        return

    probs = clf.predict_proba([text])[0]
    labels = clf.classes_
    sentiment = labels[probs.argmax()]
    confidence = probs.max()

    print(f"\nSentiment: {sentiment}")
    print(f"Confidence: {confidence:.2%}")

def analyze_reviews_from_file():
    global LAST_BATCH_ROWS, LAST_SENTIMENT_STATS, LAST_TOP_KEYWORDS, LAST_TOPIC_CLUSTERS

    path = input("\nEnter path to reviews CSV (with 'review_text' column): ").strip()
    if not os.path.exists(path):
        print("❌ File not found.")
        return

    df = pd.read_csv(path)
    if "review_text" not in df.columns:
        print("❌ CSV must contain 'review_text' column.")
        return

    clf = load_sentiment_model()

    probs = clf.predict_proba(df["review_text"])
    labels = clf.classes_
    pred_indices = probs.argmax(axis=1)
    sentiments = [labels[i] for i in pred_indices]
    confidences = probs.max(axis=1)

    df["sentiment"] = sentiments
    df["confidence"] = confidences

    # Sentiment stats
    counts = Counter(sentiments)
    total = len(sentiments)
    sentiment_stats = {
        "total_reviews": total,
        "positive": counts.get("positive", 0),
        "neutral": counts.get("neutral", 0),
        "negative": counts.get("negative", 0),
    }
    if total > 0:
        sentiment_stats["positive_pct"] = round(100 * sentiment_stats["positive"] / total, 2)
        sentiment_stats["neutral_pct"] = round(100 * sentiment_stats["neutral"] / total, 2)
        sentiment_stats["negative_pct"] = round(100 * sentiment_stats["negative"] / total, 2)

    # Keywords
    config = load_config()
    top_n = config["analysis"]["top_n_keywords"]
    top_keywords = extract_top_keywords(df["review_text"], top_n=top_n)

    # Topic clustering
    n_topics = config["analysis"]["n_topics"]
    topic_clusters = cluster_reviews(df["review_text"], n_topics=n_topics)

    # Save for reporting
    LAST_BATCH_ROWS = df.to_dict(orient="records")
    LAST_SENTIMENT_STATS = sentiment_stats
    LAST_TOP_KEYWORDS = top_keywords
    LAST_TOPIC_CLUSTERS = topic_clusters

    print("\nBatch analysis complete.")
    print(f"Total reviews: {total}")
    print(f"Positive: {sentiment_stats['positive']} | Neutral: {sentiment_stats['neutral']} | Negative: {sentiment_stats['negative']}")

def generate_report_from_last_analysis():
    if LAST_BATCH_ROWS is None:
        print("❌ No previous batch analysis found. Run 'Analyze reviews from file' first.")
        return

    summary_path = generate_summary_report(
        LAST_SENTIMENT_STATS,
        LAST_TOP_KEYWORDS,
        LAST_TOPIC_CLUSTERS
    )
    detailed_path = export_detailed_csv(LAST_BATCH_ROWS)

    print(f"\n✅ Summary report saved to: {summary_path}")
    print(f"✅ Detailed CSV saved to: {detailed_path}")

def view_statistics_in_console():
    if LAST_BATCH_ROWS is None:
        print("❌ No previous batch analysis found.")
        return

    print("\nLast batch sentiment stats:")
    for k, v in LAST_SENTIMENT_STATS.items():
        print(f"- {k}: {v}")

    print("\nTop keywords:")
    for word, freq in LAST_TOP_KEYWORDS:
        print(f"- {word}: {freq}")

    print("\nTopic clusters (showing first 2 reviews each):")
    for topic_id, reviews in LAST_TOPIC_CLUSTERS.items():
        print(f"\nTopic {topic_id}:")
        for r in reviews[:2]:
            print(f"  • {r}")

def main_menu():
    while True:
        print("\n=== TEXT ANALYTICS SUITE ===")
        print("1) Analyze single review")
        print("2) Analyze reviews from file")
        print("3) Generate report from last analysis")
        print("4) View statistics in console")
        print("5) Exit")
        choice = input("Enter your choice: ").strip()

        if choice == "1":
            analyze_single_review()
        elif choice == "2":
            analyze_reviews_from_file()
        elif choice == "3":
            generate_report_from_last_analysis()
        elif choice == "4":
            view_statistics_in_console()
        elif choice == "5":
            print("Goodbye!")
            break
        else:
            print("❌ Invalid choice. Try again.")

if __name__ == "__main__":
    main_menu()
