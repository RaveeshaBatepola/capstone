import csv
import json
import random
from pathlib import Path

N = 45
MARKS = Path("results/review_marks.csv")

if MARKS.exists():
    raise SystemExit("results/review_marks.csv already exists. Delete it only if "
                     "you really want to start the review over.")


def read_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f]


chunks = {c["chunk_id"]: c["text"]
          for c in read_jsonl("data/processed/chunks_test.jsonl")}
preds = read_jsonl("results/preds_hybrid_rag_test.jsonl")
sample = random.Random(1).sample(preds, N)

with open("results/review_sheet.txt", "w", encoding="utf-8") as f:
    for n, r in enumerate(sample, 1):
        f.write("=" * 80 + "\n")
        f.write(f"ITEM {n}    id: {r['id']}\n")
        f.write(f"QUESTION: {r['question']}\n")
        f.write(f"ANSWER GIVEN: {r['pred']}\n")
        if r["abstained_by_threshold"]:
            f.write("(refused by the evidence threshold, no LLM call)\n")
        if r["raw"]:
            f.write("MODEL'S OWN REASONING (judge against the passages, not this):\n")
            f.write(r["raw"] + "\n")
        f.write("PASSAGES THE ANSWER WAS BASED ON:\n")
        for i, h in enumerate(r["retrieved"], 1):
            f.write(f"  [{i}] {chunks[h['chunk_id']]}\n")
        f.write("\n")

with open(MARKS, "w", newline="", encoding="utf-8-sig") as f:
    w = csv.writer(f)
    w.writerow(["item", "id", "label", "note"])
    for n, r in enumerate(sample, 1):
        w.writerow([n, r["id"], "", ""])

print(f"Wrote results/review_sheet.txt and {MARKS} with {N} items.")