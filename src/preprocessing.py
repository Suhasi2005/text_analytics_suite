import json
import os
import re
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Path to config.json at project root
CONFIG_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "config.json")


# Make sure you download these once in a setup script or interactively:
# nltk.download('punkt')
# nltk.download('stopwords')
# nltk.download('wordnet')


class TextPreprocessor:
    def __init__(self):
        self.stop_words = set(stopwords.words("english"))
        self.lemmatizer = WordNetLemmatizer()

    def clean_text(self, text: str) -> str:
        text = text.lower()
        text = re.sub(r"http\S+", "", text)
        text = re.sub(r"[^a-z\s]", " ", text)
        tokens = text.split()
        tokens = [t for t in tokens if t not in self.stop_words]
        tokens = [self.lemmatizer.lemmatize(t) for t in tokens]
        return " ".join(tokens)


def load_config():
    with open(CONFIG_PATH, "r") as f:
        return json.load(f)


def _abs_path(rel_path: str) -> str:
    """Convert a relative path from config.json to an absolute path from project root."""
    base_dir = os.path.dirname(os.path.dirname(__file__))  # project root
    return os.path.join(base_dir, rel_path)


def load_labeled_data():
    config = load_config()
    rel_path = config["data"]["labeled_reviews_path"]
    path = _abs_path(rel_path)
    return pd.read_csv(path)


def load_raw_reviews():
    config = load_config()
    rel_path = config["data"]["raw_reviews_path"]
    path = _abs_path(rel_path)
    return pd.read_csv(path)
