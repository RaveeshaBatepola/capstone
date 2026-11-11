import json
import random

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm

from src.retriever import Retriever

RRF_K = 60
KS = [1, 3, 5, 10, 20]
THRESHOLDS = [0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50]
split = "dev"

r = Retriever(split)
with open(f"data/processed/questions_{split}.jsonl", encoding="utf-8") as f:
    questions = [json.loads(line) for line in f]

id_to_idx = {cid: i for i, cid in enumerate(r.ids)}
n = r.n
all_docs = sorted(set(r.doc_ids.tolist()))
rng = random.Random(0)


def rank_positions(scores):
    """1-based rank of every chunk under the given scores."""
    order = np.argsort(-scores)
    pos = np.empty(n, dtype=int)
    pos[order] = np.arange(1, n + 1)
    return pos


def restrict(scores, mask):
    return np.where(mask, scores, -np.inf)


def rrf(b, d):
    """Reciprocal Rank Fusion of two score arrays."""
    return 1.0 / (RRF_K + rank_positions(b)) + 1.0 / (RRF_K + rank_positions(d))


def top10_max_dense(h, d):
    top = np.argsort(-h)[:10]
    return float(d[top].max())


results = {}
good_scores, bad_scores = [], []

for q in tqdm(questions, desc="evaluating"):
    gold_idx = np.array([id_to_idx[g] for g in q["gold_chunk_ids"] if g in id_to_idx])
    if len(gold_idx) == 0:
        continue

    b = r.bm25_scores(q["question"])
    d = r.dense_scores(q["question"])
    mask = r.doc_ids == q["doc_id"]

    # open retrieval: all chunks
    open_scores = {"BM25": b, "Dense": d, "Hybrid (RRF)": rrf(b, d)}
    # same-doc retrieval: only the correct report page
    bm, dm = restrict(b, mask), restrict(d, mask)
    hm = restrict(rrf(bm, dm), mask)
    doc_scores = {"BM25": bm, "Dense": dm, "Hybrid (RRF)": hm}

    for method in open_scores:
        results.setdefault((method, "open"), []).append(
            rank_positions(open_scores[method])[gold_idx])
        results.setdefault((method, "same-doc"), []).append(
            rank_positions(doc_scores[method])[gold_idx])

    # threshold calibration: right report vs a random wrong report
    good_scores.append(top10_max_dense(hm, d))
    wrong = rng.choice([x for x in all_docs if x != q["doc_id"]])
    wmask = r.doc_ids == wrong
    wb, wd = restrict(b, wmask), restrict(d, wmask)
    wh = restrict(rrf(wb, wd), wmask)
    bad_scores.append(top10_max_dense(wh, d))

# ---------- retrieval table ----------
rows = []
for (method, setting), lst in results.items():
    row = {"method": method, "setting": setting}
    for k in KS:
        row[f"R@{k}"] = np.mean([(rk <= k).mean() for rk in lst])
    row["MRR"] = np.mean([1.0 / rk.min() for rk in lst])
    row["Full@5"] = np.mean([(rk <= 5).all() for rk in lst])
    row["Full@10"] = np.mean([(rk <= 10).all() for rk in lst])
    rows.append(row)

df = pd.DataFrame(rows).round(3)
df["setting"] = pd.Categorical(df["setting"], ["same-doc", "open"])
df["method"] = pd.Categorical(df["method"], ["BM25", "Dense", "Hybrid (RRF)"])
df = df.sort_values(["setting", "method"])
print("\n=== Retrieval: BM25 vs Dense vs Hybrid (dev) ===")
print(df.to_string(index=False))
df.to_csv("results/hybrid_retrieval_dev.csv", index=False)

# ---------- threshold table ----------
good, bad = np.array(good_scores), np.array(bad_scores)
print("\nMax dense score in top-10 fused chunks:")
print(f"  right report : mean {good.mean():.3f}  median {np.median(good):.3f}  "
      f"5th pct {np.percentile(good, 5):.3f}")
print(f"  wrong report : mean {bad.mean():.3f}  median {np.median(bad):.3f}  "
      f"95th pct {np.percentile(bad, 95):.3f}")

trows = []
for t in THRESHOLDS:
    trows.append({
        "threshold": t,
        "wrongly_abstains_on_real_%": round(100 * (good < t).mean(), 1),
        "correctly_abstains_on_wrong_doc_%": round(100 * (bad < t).mean(), 1),
    })
tdf = pd.DataFrame(trows)
print("\n=== Threshold calibration ===")
print(tdf.to_string(index=False))
tdf.to_csv("results/threshold_calibration_dev.csv", index=False)

plt.figure(figsize=(7, 4))
plt.hist(good, bins=40, alpha=0.6, label="right report")
plt.hist(bad, bins=40, alpha=0.6, label="wrong report (simulated unanswerable)")
plt.xlabel("max dense similarity in top-10 fused chunks")
plt.ylabel("questions")
plt.title("Evidence-score distribution (dev)")
plt.legend()
plt.tight_layout()
plt.savefig("results/figures/08_threshold_calibration.png", dpi=150)
print("\nsaved results/figures/08_threshold_calibration.png")