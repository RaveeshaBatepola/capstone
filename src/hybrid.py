import numpy as np

RRF_K = 60


def _rank_positions(scores):
    order = np.argsort(-scores)
    pos = np.empty(len(scores), dtype=int)
    pos[order] = np.arange(1, len(scores) + 1)
    return pos


def hybrid_search(r, query, k=10, doc_id=None):
    """BM25 + dense fused with Reciprocal Rank Fusion.

    Returns (hits, evidence_score). evidence_score is the highest dense
    similarity among the top-k fused chunks, used for the abstention check.
    """
    b_raw = r.bm25_scores(query)
    d_raw = r.dense_scores(query)
    mask = np.ones(r.n, dtype=bool) if doc_id is None else (r.doc_ids == doc_id)

    b = np.where(mask, b_raw, -np.inf)
    d = np.where(mask, d_raw, -np.inf)
    fused = 1.0 / (RRF_K + _rank_positions(b)) + 1.0 / (RRF_K + _rank_positions(d))
    fused = np.where(mask, fused, -np.inf)

    top = [i for i in np.argsort(-fused)[:k] if np.isfinite(fused[i])]
    hits = [{"chunk_id": r.ids[i],
             "rrf": float(fused[i]),
             "dense": float(d_raw[i]),
             "bm25": float(b_raw[i])} for i in top]
    evidence_score = float(d_raw[top].max()) if top else 0.0
    return hits, evidence_score