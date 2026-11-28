import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

FIG = Path("results/figures")
FIG.mkdir(parents=True, exist_ok=True)
BLUE, LIGHT, GREY, ORANGE = "#2F5496", "#8FAADC", "#A5A5A5", "#ED7D31"
plt.rcParams.update({"font.size": 9, "axes.spines.top": False,
                     "axes.spines.right": False})


def read_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f]


# ---------- Figure 1: EDA overview (train split) ----------
q = pd.DataFrame(read_jsonl("data/processed/questions_train.jsonl"))
q["program"] = q["program"].fillna("")
q["ops"] = q["program"].apply(lambda s: re.findall(r"([a-z_]+)\(", s))
q["n_steps"] = q["ops"].apply(len)


def evidence_kind(ids):
    t = sum("::table_" in g for g in ids)
    x = len(ids) - t
    return "Text and\ntable" if t and x else "Table rows\nonly" if t else "Text\nonly"


kinds = q["gold_chunk_ids"].apply(evidence_kind).value_counts(normalize=True) * 100
kinds = kinds.reindex(["Table rows\nonly", "Text\nonly", "Text and\ntable"])
steps = q["n_steps"].value_counts().reindex(range(1, 7), fill_value=0)
ops = pd.Series([o for lst in q["ops"] for o in lst]).value_counts()

fig, ax = plt.subplots(1, 3, figsize=(10, 3.3))
ax[0].bar(kinds.index, kinds.values, color=[BLUE, LIGHT, GREY])
for i, v in enumerate(kinds.values):
    ax[0].text(i, v + 1, f"{v:.1f}%", ha="center")
ax[0].set_title("Where the gold evidence comes from")
ax[0].set_ylabel("% of questions")
ax[0].set_ylim(0, 75)

ax[1].bar([str(s) for s in steps.index], steps.values, color=BLUE)
for i, v in enumerate(steps.values):
    ax[1].text(i, v + 60, str(v), ha="center", fontsize=8)
ax[1].set_title("Reasoning steps per question")
ax[1].set_xlabel("steps in the gold program")
ax[1].set_ylabel("questions")

ax[2].barh(ops.index[::-1], ops.values[::-1], color=BLUE)
ax[2].set_title("Operations in the gold programs")
ax[2].set_xlabel("count")
plt.tight_layout()
plt.savefig(FIG / "10_eda_overview.png", dpi=200)
plt.close()

# ---------- Figure 2: retrieval MRR (dev) ----------
r = pd.read_csv("results/hybrid_retrieval_dev.csv")
methods = ["BM25", "Dense", "Hybrid (RRF)"]
same = [r[(r.method == m) & (r.setting == "same-doc")]["MRR"].iloc[0] for m in methods]
opn = [r[(r.method == m) & (r.setting == "open")]["MRR"].iloc[0] for m in methods]
x, w = np.arange(3), 0.36
fig, ax = plt.subplots(figsize=(6.4, 3.3))
b1 = ax.bar(x - w / 2, same, w, label="Report page given", color=BLUE)
b2 = ax.bar(x + w / 2, opn, w, label="Report page not given (open)", color=LIGHT)
for b in list(b1) + list(b2):
    ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.01,
            f"{b.get_height():.3f}", ha="center", fontsize=8)
ax.set_xticks(x)
ax.set_xticklabels(methods)
ax.set_ylim(0, 1)
ax.set_ylabel("MRR")
ax.set_title("Retrieval quality on the 883 dev questions")
ax.legend(frameon=False, loc="upper right")
plt.tight_layout()
plt.savefig(FIG / "11_retrieval_mrr_dev.png", dpi=200)
plt.close()

# ---------- Figure 3: answer accuracy (dev) ----------
a = pd.read_csv("results/answer_scores_dev.csv")
fig, ax = plt.subplots(figsize=(6.4, 3.3))
b1 = ax.bar(x - w / 2, a["EM %"], w, label="Exact Match", color=BLUE)
b2 = ax.bar(x + w / 2, a["Rounded match %"], w, label="Rounded match", color=ORANGE)
for b in list(b1) + list(b2):
    ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 1,
            f"{b.get_height():.1f}", ha="center", fontsize=8)
ax.set_xticks(x)
ax.set_xticklabels(a["config"])
ax.set_ylim(0, 75)
ax.set_ylabel("% of questions correct")
ax.set_title("Answer accuracy on 150 dev questions")
ax.legend(frameon=False, loc="upper left")
plt.tight_layout()
plt.savefig(FIG / "12_answer_accuracy_dev.png", dpi=200)
plt.close()

print("Saved: 10_eda_overview.png, 11_retrieval_mrr_dev.png, 12_answer_accuracy_dev.png")