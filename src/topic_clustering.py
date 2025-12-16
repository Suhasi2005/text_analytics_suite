from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans

from preprocessing import TextPreprocessor

def cluster_reviews(texts, n_topics=3):
    preprocessor = TextPreprocessor()
    cleaned = [preprocessor.clean_text(t) for t in texts]

    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(cleaned)

    km = KMeans(n_clusters=n_topics, random_state=42, n_init="auto")
    labels = km.fit_predict(X)

    clusters = {i: [] for i in range(n_topics)}
    for text, label in zip(texts, labels):
        clusters[label].append(text)

    return clusters
