"""
Knowledge-base retrieval: TF-IDF -> TruncatedSVD (LSA) -> FAISS cosine search.

KnowledgeBase.build()   reads and validates the FAQ file and builds an index
KnowledgeBase.save()    writes ONE file (models/kb_index.joblib)
KnowledgeBase.load()    reads it back
KnowledgeBase.search()  returns the top articles with similarity scores

Similarity measures how related the text is. It is NOT proof that an article
answers the question; that decision belongs to src/answerability.py.
"""
import hashlib
import json
from pathlib import Path

import faiss
import joblib
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer

ROOT = Path(__file__).parent.parent
DEFAULT_KB_PATH = ROOT / "data" / "faq_knowledge_base.json"
DEFAULT_INDEX_PATH = ROOT / "models" / "kb_index.joblib"
REQUIRED_FIELDS = ("id", "question", "answer")


def file_fingerprint(path) -> str:
    """SHA-256 of a file's bytes. Changes whenever the FAQ file is edited."""
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_articles(kb_path) -> list:
    """Read the FAQ file and check it is well formed."""
    with open(kb_path, encoding="utf-8") as f:
        articles = json.load(f)
    if not isinstance(articles, list) or not articles:
        raise ValueError(f"{kb_path} must contain a non-empty JSON list of articles")
    seen = set()
    for pos, article in enumerate(articles):
        missing = [k for k in REQUIRED_FIELDS if not article.get(k)]
        if missing:
            raise ValueError(f"Article #{pos} is missing required fields: {missing}")
        if article["id"] in seen:
            raise ValueError(f"Duplicate article id: {article['id']}")
        seen.add(article["id"])
    return articles


def _article_text(article: dict) -> str:
    # question + answer, so both question-style and keyword-style queries can match
    return f"{article['question']} {article['answer']}"


class KnowledgeBase:
    def __init__(self, vectorizer, svd, vectors, articles, fingerprint):
        self.vectorizer = vectorizer
        self.svd = svd
        self.articles = articles
        self.fingerprint = fingerprint
        self._vectors = vectors  # already L2-normalised, shape (n_articles, n_components)
        self._index = faiss.IndexFlatIP(vectors.shape[1])
        self._index.add(vectors)

    @classmethod
    def build(cls, kb_path=DEFAULT_KB_PATH, n_components: int = 100):
        articles = load_articles(kb_path)
        texts = [_article_text(a) for a in articles]

        vectorizer = TfidfVectorizer(ngram_range=(1, 2), stop_words="english")
        tfidf = vectorizer.fit_transform(texts)

        # SVD cannot use more components than the data allows
        k = min(n_components, tfidf.shape[0] - 1, tfidf.shape[1] - 1)
        svd = TruncatedSVD(n_components=k, random_state=42)
        vectors = svd.fit_transform(tfidf).astype("float32")
        faiss.normalize_L2(vectors)  # inner product of unit vectors == cosine similarity

        return cls(vectorizer, svd, vectors, articles, file_fingerprint(kb_path))

    def save(self, path=DEFAULT_INDEX_PATH):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(
            {
                "vectorizer": self.vectorizer,
                "svd": self.svd,
                "vectors": self._vectors,
                "articles": self.articles,
                "fingerprint": self.fingerprint,
            },
            path,
        )

    @classmethod
    def load(cls, path=DEFAULT_INDEX_PATH):
        bundle = joblib.load(path)
        return cls(
            bundle["vectorizer"], bundle["svd"], bundle["vectors"],
            bundle["articles"], bundle["fingerprint"],
        )

    def is_stale(self, kb_path=DEFAULT_KB_PATH) -> bool:
        """True if the FAQ file changed since this index was built (rebuild needed)."""
        return file_fingerprint(kb_path) != self.fingerprint

    def search(self, query: str, top_k: int = 3) -> list:
        top_k = max(1, min(int(top_k), len(self.articles)))
        text = query if isinstance(query, str) else ""
        q = self.svd.transform(self.vectorizer.transform([text])).astype("float32")
        faiss.normalize_L2(q)
        scores, idxs = self._index.search(q, top_k)

        results = []
        for rank, (score, idx) in enumerate(zip(scores[0], idxs[0]), start=1):
            if idx == -1:
                continue
            results.append({**self.articles[idx], "similarity": round(float(score), 3), "rank": rank})
        return results