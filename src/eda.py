import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from src.data_loader import load_split

PROC = Path("data/processed")
FIG = Path("results/figures")
FIG.mkdir(parents=True, exist_ok=True)
sns.set_theme(style="whitegrid")


def read_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def save(name):
    plt.tight_layout()
    plt.savefig(FIG / name, dpi=150)
    plt.close()
    print("saved", FIG / name)


# ---------- load processed data ----------
splits = ["train", "dev", "test"]
chunks = {s: pd.DataFrame(read_jsonl(PROC / f"chunks_{s}.jsonl")) for s in splits}
qs = {s: pd.DataFrame(read_jsonl(PROC / f"questions_{s}.jsonl")) for s in splits}

# ---------- 1. dataset sizes ----------
print("\n=== 1. DATASET SIZE ===")
rows = []
for s in splits:
    rows.append({
        "split": s,
        "questions": len(qs[s]),
        "documents": qs[s]["doc_id"].nunique(),
        "chunks": len(chunks[s]),
    })
size_df = pd.DataFrame(rows)
print(size_df.to_string(index=False))

size_df.set_index("split")[["questions", "documents", "chunks"]].plot(
    kind="bar", figsize=(7, 4), rot=0)
plt.title("Dataset size per split")
plt.ylabel("count")
save("01_dataset_size.png")

# ---------- 2. chunks ----------
print("\n=== 2. CHUNKS ===")
train_chunks = chunks["train"].copy()
train_chunks["n_words"] = train_chunks["text"].str.split().str.len()
print(train_chunks["type"].value_counts().to_string())
print("\nWords per chunk (by type):")
print(train_chunks.groupby("type")["n_words"].describe().round(1).to_string())

per_doc = train_chunks.groupby("doc_id").size()
print("\nChunks per document:")
print(per_doc.describe().round(1).to_string())

plt.figure(figsize=(7, 4))
sns.histplot(data=train_chunks[train_chunks["n_words"] <= 150],
             x="n_words", hue="type", bins=40, element="step")
plt.title("Chunk length in words (train, capped at 150)")
save("02_chunk_length.png")

# ---------- 3. questions ----------
print("\n=== 3. QUESTIONS ===")
q = qs["train"].copy()
q["q_words"] = q["question"].str.split().str.len()
print("Question length (words):")
print(q["q_words"].describe().round(1).to_string())

plt.figure(figsize=(7, 4))
sns.histplot(q["q_words"], bins=30)
plt.title("Question length in words (train)")
save("03_question_length.png")

# ---------- 4. reasoning programs ----------
print("\n=== 4. REASONING PROGRAMS ===")
q["program"] = q["program"].fillna("")
q["ops"] = q["program"].apply(lambda p: re.findall(r"([a-z_]+)\(", p))
q["n_steps"] = q["ops"].apply(len)

print("Number of reasoning steps:")
print(q["n_steps"].value_counts().sort_index().to_string())

all_ops = pd.Series([op for ops in q["ops"] for op in ops])
print("\nOperation counts:")
print(all_ops.value_counts().to_string())

plt.figure(figsize=(7, 4))
all_ops.value_counts().plot(kind="bar")
plt.title("Operations used in gold programs (train)")
plt.ylabel("count")
save("04_operations.png")

plt.figure(figsize=(7, 4))
sns.countplot(x=q["n_steps"].clip(upper=6))
plt.title("Reasoning steps per question (6 means 6 or more)")
plt.xlabel("steps")
save("05_reasoning_steps.png")

# ---------- 5. gold evidence ----------
print("\n=== 5. GOLD EVIDENCE ===")
q["n_gold"] = q["gold_chunk_ids"].apply(len)
q["n_gold_table"] = q["gold_chunk_ids"].apply(
    lambda ids: sum("::table_" in g for g in ids))
q["n_gold_text"] = q["n_gold"] - q["n_gold_table"]


def kind(r):
    if r["n_gold_table"] and r["n_gold_text"]:
        return "text + table"
    if r["n_gold_table"]:
        return "table only"
    if r["n_gold_text"]:
        return "text only"
    return "none"


q["evidence_kind"] = q.apply(kind, axis=1)
print("Gold chunks per question:")
print(q["n_gold"].describe().round(2).to_string())
print("\nEvidence type needed:")
print(q["evidence_kind"].value_counts().to_string())
print((100 * q["evidence_kind"].value_counts(normalize=True)).round(1).to_string())

plt.figure(figsize=(6, 4))
q["evidence_kind"].value_counts().plot(kind="bar", rot=0)
plt.title("Where the gold evidence comes from (train)")
plt.ylabel("questions")
save("06_evidence_type.png")

# ---------- 6. table sizes ----------
print("\n=== 6. TABLE SIZES ===")
raw = load_split("train")
seen = {}
for ex in raw:
    if ex["filename"] not in seen:
        t = ex["table"]
        seen[ex["filename"]] = (len(t), max((len(r) for r in t), default=0))
tdf = pd.DataFrame(seen.values(), columns=["rows", "cols"])
print(tdf.describe().round(1).to_string())

# ---------- save summary ----------
summary = {
    "sizes": size_df.to_dict(orient="records"),
    "chunks_per_doc_mean": float(per_doc.mean()),
    "median_question_words": float(q["q_words"].median()),
    "multi_step_pct": float(100 * (q["n_steps"] > 1).mean()),
    "evidence_kind_pct": (100 * q["evidence_kind"].value_counts(normalize=True)).round(1).to_dict(),
}
with open("results/eda_summary.json", "w") as f:
    json.dump(summary, f, indent=2)
print("\nsaved results/eda_summary.json")