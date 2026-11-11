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
parser.add_argument("mode", choices=["llm_only", "dense_rag", "hybrid_rag"])
parser.add_argument("split", choices=["dev", "test"])
parser.add_argument("--n", type=int, default=20, help="number of questions")
parser.add_argument("--k", type=int, default=10, help="passages for RAG")
parser.add_argument("--sleep", type=float, default=5.0, help="seconds between LLM calls")
parser.add_argument("--threshold", type=float, default=0.45,
                    help="hybrid_rag: abstain below this evidence score")
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
if args.mode != "llm_only":
    from src.retriever import Retriever
    retriever = Retriever(args.split)
if args.mode == "hybrid_rag":
    from src.hybrid import hybrid_search

n_abstained = 0
with open(out_path, "a", encoding="utf-8") as out:
    for q in tqdm(questions, desc=args.mode):
        if q["id"] in done:
            continue

        retrieved, evidence_score, abstained, raw = [], None, False, ""
        called_llm = True

        if args.mode == "llm_only":
            raw = generate(LLM_ONLY_PROMPT.format(question=q["question"]))
            pred = parse_final(raw)

        else:
            if args.mode == "dense_rag":
                hits = retriever.search_dense(q["question"], k=args.k, doc_id=q["doc_id"])
                retrieved = [{"chunk_id": c, "score": s} for c, s in hits]
            else:
                retrieved, evidence_score = hybrid_search(
                    retriever, q["question"], k=args.k, doc_id=q["doc_id"])

            if args.mode == "hybrid_rag" and evidence_score < args.threshold:
                abstained, called_llm = True, False
                n_abstained += 1
                pred = "insufficient evidence"
            else:
                passages = [(h["chunk_id"], retriever.text_of(h["chunk_id"]))
                            for h in retrieved]
                prompt = RAG_PROMPT.format(evidence=format_evidence(passages),
                                           question=q["question"])
                raw = generate(prompt)
                pred = parse_final(raw)

        rec = {
            "id": q["id"],
            "question": q["question"],
            "gold_answer": q["answer"],
            "gold_exe_ans": q["exe_ans"],
            "pred": pred,
            "raw": raw,
            "retrieved": retrieved,
            "evidence_score": evidence_score,
            "abstained_by_threshold": abstained,
            "gold_chunk_ids": q["gold_chunk_ids"],
        }
        out.write(json.dumps(rec) + "\n")
        out.flush()
        if called_llm:
            time.sleep(args.sleep)

# ---- summary ----
with open(out_path, encoding="utf-8") as f:
    recs = [json.loads(line) for line in f]

print("\n--- first examples ---")
for r in recs[:5]:
    print("Q    :", r["question"])
    print("GOLD :", r["gold_answer"], "| exe_ans:", r["gold_exe_ans"])
    print("PRED :", r["pred"])
    print()

n_new = len(questions) - len(done)
print(f"Tokens used this run: {llm_mod.USED_TOKENS} for {n_new} new questions "
      f"({llm_mod.USED_TOKENS / max(n_new, 1):.0f} per question)")
if args.mode == "hybrid_rag":
    total_abs = sum(r["abstained_by_threshold"] for r in recs)
    print(f"Abstained by threshold {args.threshold}: {total_abs} of {len(recs)} questions")