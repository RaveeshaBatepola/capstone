import pickle
import re
import sys
import time
import json
from pathlib import Path

import faiss
import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

PROC = Path("data/processed")
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


def tokenize(text: str):
    """Lowercase words and numbers; keeps values like 3.8 or 1,200 whole."""
    return re.findall(r"[a-z0-9]+(?:[.,][0-9]+)*", text.lower())


def read_chunks(split):
    with open(PROC / f"chunks_{split}.jsonl", encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def build_bm25(chunks, split):
    corpus = [tokenize(c["text"]) for c in chunks]
    bm25 = BM25Okapi(corpus)
    with open(PROC / f"bm25_{split}.pkl", "wb") as f:
        pickle.dump(bm25, f)
    print(f"  BM25 index saved ({len(corpus)} chunks)")


def build_dense(chunks, split, model):
    texts = [c["text"] for c in chunks]
    emb = model.encode(texts, batch_size=64, show_progress_bar=True,
                       normalize_embeddings=True, convert_to_numpy=True)
    emb = emb.astype("float32")
    index = faiss.IndexFlatIP(emb.shape[1])   # inner product = cosine (normalized)
    index.add(emb)
    faiss.write_index(index, str(PROC / f"dense_{split}.faiss"))
    print(f"  Dense index saved: {index.ntotal} vectors, dim {emb.shape[1]}")


if __name__ == "__main__":
    splits = sys.argv[1:] or ["dev", "test"]
    model = SentenceTransformer(MODEL_NAME)
    for split in splits:
        print(f"\n=== {split} ===")
        t0 = time.time()
        chunks = read_chunks(split)
        build_bm25(chunks, split)
        build_dense(chunks, split, model)
        print(f"  done in {time.time() - t0:.0f}s")