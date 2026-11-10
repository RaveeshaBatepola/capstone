import pickle
from pathlib import Path

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

from src.indexer import MODEL_NAME, PROC, read_chunks, tokenize


class Retriever:
    def __init__(self, split: str, model=None):
        self.split = split
        self.chunks = read_chunks(split)
        self.ids = [c["chunk_id"] for c in self.chunks]
        self.doc_ids = np.array([c["doc_id"] for c in self.chunks])
        self.n = len(self.chunks)

        with open(PROC / f"bm25_{split}.pkl", "rb") as f:
            self.bm25 = pickle.load(f)
        self.dense = faiss.read_index(str(PROC / f"dense_{split}.faiss"))
        self.model = model or SentenceTransformer(MODEL_NAME)

    # ---- full score arrays (also used later for rank fusion) ----
    def bm25_scores(self, query: str):
        return np.asarray(self.bm25.get_scores(tokenize(query)), dtype="float32")

    def dense_scores(self, query: str):
        q = self.model.encode([query], normalize_embeddings=True).astype("float32")
        D, I = self.dense.search(q, self.n)
        scores = np.zeros(self.n, dtype="float32")
        scores[I[0]] = D[0]
        return scores

    # ---- top-k helpers ----
    def _top(self, scores, k, doc_id=None):
        if doc_id is not None:                       # restrict to one report page
            scores = np.where(self.doc_ids == doc_id, scores, -np.inf)
        order = np.argsort(-scores)[:k]
        return [(self.ids[i], float(scores[i])) for i in order
                if np.isfinite(scores[i])]

    def search_bm25(self, query, k=10, doc_id=None):
        return self._top(self.bm25_scores(query), k, doc_id)

    def search_dense(self, query, k=10, doc_id=None):
        return self._top(self.dense_scores(query), k, doc_id)

    def text_of(self, chunk_id):
        if not hasattr(self, "_lookup"):
            self._lookup = {c["chunk_id"]: c["text"] for c in self.chunks}
        return self._lookup[chunk_id]