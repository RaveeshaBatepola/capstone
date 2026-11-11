import argparse
import json
import random
import re
import time
from pathlib import Path

from tqdm import tqdm

import src.llm as llm_mod
from src.llm import generate
from src.prompts import LLM_ONLY_PROMPT, RAG_PROMPT, format_evidence

parser = argparse.ArgumentParser()
parser.add_argument("mode", choices=["llm_only", "dense_rag"])
parser.add_argument("split", choices=["dev", "test"])
parser.add_argument("--n", type=int, default=20, help="number of questions")
parser.add_argument("--k", type=int, default=10, help="passages for RAG")
parser.add_argument("--sleep", type=float, default=5.0, help="seconds between calls")
args = parser.parse_args()


def parse_final(text):
    m = re.findall(r"FINAL ANSWER:\s*(.+)", text, flags=re.IGNORECASE)
    if m:
        return m[-1].strip().strip("*").strip()
    lines = [l for l in text.splitlines() if l.strip()]
    return lines[-1].strip() if lines else ""


# fixed sample, identical for every configuration
with open(f"data/processed/questions_{args.split}.jsonl", encoding="utf-8") as f:
    questions = [json.loads(line) for line in f]
random.Random(42).shuffle(questions)
questions = questions[: args.n]

out_path = Path(f"results/preds_{args.mode}_{args.split}.jsonl")
done = set()
if out_path.exists():
    with open(out_path, encoding="utf-8") as f:
        done = {json.loads(line)["id"] for line in f}
print(f"{len(done)} already done, {len(questions) - len(done)} to go")

retriever = None
if args.mode == "dense_rag":
    from src.retriever import Retriever
    retriever = Retriever(args.split)

with open(out_path, "a", encoding="utf-8") as out:
    for q in tqdm(questions, desc=args.mode):
        if q["id"] in done:
            continue

        retrieved = []
        if args.mode == "llm_only":
            prompt = LLM_ONLY_PROMPT.format(question=q["question"])
        else:
            hits = retriever.search_dense(q["question"], k=args.k, doc_id=q["doc_id"])
            retrieved = [(cid, retriever.text_of(cid), score) for cid, score in hits]
            prompt = RAG_PROMPT.format(
                evidence=format_evidence([(c, t) for c, t, _ in retrieved]),
                question=q["question"])

        raw = generate(prompt)
        rec = {
            "id": q["id"],
            "question": q["question"],
            "gold_answer": q["answer"],
            "gold_exe_ans": q["exe_ans"],
            "pred": parse_final(raw),
            "raw": raw,
            "retrieved": [{"chunk_id": c, "score": s} for c, _, s in retrieved],
            "gold_chunk_ids": q["gold_chunk_ids"],
        }
        out.write(json.dumps(rec) + "\n")
        out.flush()
        time.sleep(args.sleep)

# show a few examples
with open(out_path, encoding="utf-8") as f:
    recs = [json.loads(line) for line in f][:5]
print("\n--- first examples ---")
for r in recs:
    print("Q    :", r["question"])
    print("GOLD :", r["gold_answer"], "| exe_ans:", r["gold_exe_ans"])
    print("PRED :", r["pred"])
    print()

    n_new = len(questions) - len(done)
    print(f"Tokens used this run: {llm_mod.USED_TOKENS} for {n_new} new questions"
      f" ({llm_mod.USED_TOKENS / max(n_new, 1):.0f} per question)")