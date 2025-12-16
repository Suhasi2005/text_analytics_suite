import os
from datetime import datetime
import csv

from preprocessing import load_config

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def generate_summary_report(sentiment_stats, top_keywords, topic_clusters):
    """
    sentiment_stats: dict with counts & percentages
    top_keywords: list of (word, freq)
    topic_clusters: dict {topic_id: [reviews]}
    """
    config = load_config()
    reports_dir = os.path.join(BASE_DIR, config["reports"]["reports_dir"])
    os.makedirs(reports_dir, exist_ok=True)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    summary_path = os.path.join(reports_dir, f"summary_{ts}.txt")

    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("TEXT ANALYTICS SUITE – SUMMARY REPORT\n")
        f.write("=" * 50 + "\n\n")

        f.write("Sentiment Statistics:\n")
        for k, v in sentiment_stats.items():
            f.write(f"- {k}: {v}\n")
        f.write("\n")

        f.write("Top Keywords:\n")
        for word, freq in top_keywords:
            f.write(f"- {word}: {freq}\n")
        f.write("\n")

        f.write("Topic Clusters (sample reviews):\n")
        for topic_id, reviews in topic_clusters.items():
            f.write(f"\nTopic {topic_id}:\n")
            for r in reviews[:3]:  # first 3 reviews as examples
                f.write(f"  • {r}\n")

    return summary_path

def export_detailed_csv(rows):
    """
    rows: list of dicts, each like:
    {
      "review_text": ...,
      "sentiment": ...,
      "confidence": ...,
      "topic": ...
    }
    """
    config = load_config()
    reports_dir = os.path.join(BASE_DIR, config["reports"]["reports_dir"])
    os.makedirs(reports_dir, exist_ok=True)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = os.path.join(reports_dir, f"detailed_{ts}.csv")

    if not rows:
        return csv_path

    fieldnames = list(rows[0].keys())
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    return csv_path
