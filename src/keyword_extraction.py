from collections import Counter
from sklearn.feature_extraction.text import CountVectorizer

from preprocessing import TextPreprocessor

def extract_top_keywords(texts, top_n=10):
    """
    texts: list/Series of raw text
    """
    preprocessor = TextPreprocessor()
    cleaned = [preprocessor.clean_text(t) for t in texts]

    vectorizer = CountVectorizer(ngram_range=(1, 2), min_df=1)
    X = vectorizer.fit_transform(cleaned)
    sum_words = X.sum(axis=0)

    freq = []
    for word, idx in vectorizer.vocabulary_.items():
        freq.append((word, int(sum_words[0, idx])))

    freq_sorted = sorted(freq, key=lambda x: x[1], reverse=True)
    return freq_sorted[:top_n]
