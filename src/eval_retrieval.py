import json
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm

from src.retriever import Retriever

KS = [1, 3, 5, 10, 20]
split = sys.argv[1] if len(sys.argv) > 1 else "dev"

r = Retriever(split)
with open(f"data/processed/questions_{split}.jsonl", encoding="utf-8") as f:
    questions = [json.loads(line) for line in f]

id_to_idx = {cid: i for i, cid in enumerate(r.ids)}
n = r.n


def gold_ranks(scores, gold_idx):
    """1-based rank of each gold chunk under the given scores."""
    order = np.argsort(-scores)
    pos = np.empty(n, dtype=int)
    pos[order] = np.arange(1, n + 1)
    return pos[gold_idx]


results = {}  # (method, setting) -> list of rank arrays
used = 0

for q in tqdm(questions, desc=f"evaluating {split}"):
    gold_idx = np.array([id_to_idx[g] for g in q["gold_chunk_ids"]
                         if g in id_to_idx])
    if len(gold_idx) == 0:
        continue
    used += 1

    score_sets = {"BM25": r.bm25_scores(q["question"]),
                  "Dense": r.dense_scores(q["question"])}
    in_doc = r.doc_ids == q["doc_id"]

    for method, scores in score_sets.items():
        open_ranks = gold_ranks(scores, gold_idx)
        doc_scores = np.where(in_doc, scores, -np.inf)
        doc_ranks = gold_ranks(doc_scores, gold_idx)
        results.setdefault((method, "open"), []).append(open_ranks)
        results.setdefault((method, "same-doc"), []).append(doc_ranks)

print(f"\nQuestions evaluated: {used} of {len(questions)}")

recall_rows, full_rows = [], []
for (method, setting), lst in results.items():
    rr = {"method": method, "setting": setting}
    ff = {"method": method, "setting": setting}
    for k in KS:
        rr[f"R@{k}"] = np.mean([(rk <= k).mean() for rk in lst])
        ff[f"Full@{k}"] = np.mean([(rk <= k).all() for rk in lst])
    rr["MRR"] = np.mean([1.0 / rk.min() for rk in lst])
    recall_rows.append(rr)
    full_rows.append(ff)

recall_df = pd.DataFrame(recall_rows).round(3)
full_df = pd.DataFrame(full_rows).round(3)

print("\n=== Recall@K and MRR ===")
print(recall_df.to_string(index=False))
print("\n=== Full@K (all gold chunks in top K) ===")
print(full_df.to_string(index=False))

recall_df.to_csv(f"results/retrieval_{split}.csv", index=False)
full_df.to_csv(f"results/retrieval_full_{split}.csv", index=False)

# chart: Recall@5 and MRR
plot_df = recall_df.copy()
plot_df["config"] = plot_df["method"] + "\n" + plot_df["setting"]
plot_df.set_index("config")[["R@5", "R@10", "MRR"]].plot(
    kind="bar", figsize=(8, 4.5), rot=0)
plt.title(f"Retrieval baselines on {split}")
plt.ylim(0, 1)
plt.tight_layout()
plt.savefig(f"results/figures/07_retrieval_{split}.png", dpi=150)
print(f"\nsaved results/figures/07_retrieval_{split}.png")