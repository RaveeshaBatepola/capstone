import argparse
import json
import random
import re
import time
from pathlib import Path

import src.llm as llm_mod
from src.llm import generate
from src.prompts import RAG_PROMPT, format_evidence

THRESHOLD = 0.45
NUM = re.compile(r"-?\d+(?:\.\d+)?")

parser = argparse.ArgumentParser()
parser.add_argument("action", choices=["hybrid", "dense", "report"])
parser.add_argument("--n", type=int, default=40)
parser.add_argument("--sleep", type=float, default=8.0)
args = parser.parse_args()


def read_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def parse_final(text):
    m = re.findall(r"FINAL ANSWER:\s*(.+)", text, flags=re.IGNORECASE)
    if m:
        return m[-1].strip().strip("*").strip()
    lines = [l for l in text.splitlines() if l.strip()]
    return lines[-1].strip() if lines else ""


def parse_num(s):
    t = str(s).replace(",", "").replace("$", "")
    m = NUM.search(t)
    if not m:
        return None
    txt = m.group(0)
    dec = len(txt.split(".")[1]) if "." in txt else 0
    return float(txt), dec, "%" in str(s)


def matches_gold(pred, gold):
    """Strict match at the gold answer's precision (same rule as eval_answers)."""
    p, g = parse_num(pred), parse_num(gold)
    if p is None or g is None:
        return str(pred).strip().lower() == str(gold).strip().lower()
    pv, _, pp = p
    gv, gd, gp = g
    tol = 0.5 * 10 ** (-gd) + 1e-9
    if abs(pv - gv) <= tol:
        return True
    return bool(gp and not pp and abs(pv * 100 - gv) <= tol)


def is_abstain(p):
    return "insufficient evidence" in str(p).lower()


def build_pairs(n):
    """The first n questions of the 150-question sample, each paired with a
    report from a DIFFERENT company."""
    qs = read_jsonl("data/processed/questions_dev.jsonl")
    random.Random(42).shuffle(qs)
    qs = qs[:n]
    docs = sorted({c["doc_id"] for c in read_jsonl("data/processed/chunks_dev.jsonl")})
    rng = random.Random(7)
    pairs = []
    for q in qs:
        company = q["doc_id"].split("/")[0]
        wrong = rng.choice([d for d in docs if d.split("/")[0] != company])
        pairs.append({**q, "wrong_doc_id": wrong})
    return pairs


pairs = build_pairs(args.n)

# ---------------------------------------------------------------- run
if args.action in ("hybrid", "dense"):
    from src.hybrid import hybrid_search
    from src.retriever import Retriever

    r = Retriever("dev")
    out_path = Path(f"results/stress_{args.action}_dev.jsonl")
    done = set()
    if out_path.exists():
        done = {x["id"] for x in read_jsonl(out_path)}
    print(f"{len(done)} already done, {len(pairs) - len(done)} to go")

    with open(out_path, "a", encoding="utf-8") as out:
        for q in pairs:
            if q["id"] in done:
                continue
            if args.action == "dense":
                hits = r.search_dense(q["question"], k=10, doc_id=q["wrong_doc_id"])
                retrieved = [{"chunk_id": c, "score": s} for c, s in hits]
                ev = max((s for _, s in hits), default=0.0)
            else:
                retrieved, ev = hybrid_search(r, q["question"], k=10,
                                              doc_id=q["wrong_doc_id"])
            passages = [(h["chunk_id"], r.text_of(h["chunk_id"])) for h in retrieved]
            raw = generate(RAG_PROMPT.format(evidence=format_evidence(passages),
                                             question=q["question"]))
            out.write(json.dumps({
                "id": q["id"], "question": q["question"],
                "gold_answer": q["answer"], "right_doc_id": q["doc_id"],
                "wrong_doc_id": q["wrong_doc_id"],
                "pred": parse_final(raw), "raw": raw,
                "evidence_score": ev, "retrieved": retrieved,
            }) + "\n")
            out.flush()
            print(f"  done {q['id']}  evidence score {ev:.3f}")
            time.sleep(args.sleep)

    n_new = len(pairs) - len(done)
    print(f"Tokens used this run: {llm_mod.USED_TOKENS} for {n_new} new questions")

# ------------------------------------------------------------- report
else:
    H = {x["id"]: x for x in read_jsonl("results/stress_hybrid_dev.jsonl")}
    D = {x["id"]: x for x in read_jsonl("results/stress_dense_dev.jsonl")}
    ids = [q["id"] for q in pairs if q["id"] in H and q["id"] in D]
    n = len(ids)
    print(f"Questions with both runs: {n} (of {len(pairs)} requested)")

    ev = [H[i]["evidence_score"] for i in ids]
    print(f"Mean evidence score on the wrong report: {sum(ev) / n:.3f}")

    configs = {
        "Dense RAG (prompt abstention only)":
            (lambda i: is_abstain(D[i]["pred"]), lambda i: D[i]["pred"]),
        "Hybrid RAG, no threshold":
            (lambda i: is_abstain(H[i]["pred"]), lambda i: H[i]["pred"]),
        f"Hybrid RAG + threshold {THRESHOLD}":
            (lambda i: H[i]["evidence_score"] < THRESHOLD or is_abstain(H[i]["pred"]),
             lambda i: H[i]["pred"]),
    }

    print(f"\n{'configuration':40s} {'refused':>8s} {'answered':>9s} "
          f"{'wrong answer':>13s} {'matched gold':>13s}")
    for name, (refuses, pred_of) in configs.items():
        refused = sum(refuses(i) for i in ids)
        answered = [i for i in ids if not refuses(i)]
        lucky = sum(matches_gold(pred_of(i), H[i]["gold_answer"]) for i in answered)
        print(f"{name:40s} {refused:>5d}/{n:<2d} {len(answered):>6d}/{n:<2d} "
              f"{len(answered) - lucky:>10d}/{n:<2d} {lucky:>10d}/{n:<2d}")

    thr = [i for i in ids if H[i]["evidence_score"] < THRESHOLD]
    prevented = [i for i in thr if not is_abstain(H[i]["pred"])]
    print(f"\nThreshold refused {len(thr)} of {n} questions.")
    print(f"Of those, the LLM would otherwise have answered {len(prevented)} "
          f"(answers the threshold prevented).")
    print(f"Answers that got past the threshold: {n - len(thr)}; "
          f"the LLM then refused {sum(is_abstain(H[i]['pred']) for i in ids if i not in thr)} of them.")

    print("\n=== Hybrid + threshold: answers given on an unrelated report ===")
    shown = 0
    for i in ids:
        if H[i]["evidence_score"] >= THRESHOLD and not is_abstain(H[i]["pred"]):
            ok = matches_gold(H[i]["pred"], H[i]["gold_answer"])
            print("Q    :", H[i]["question"])
            print("GOLD :", H[i]["gold_answer"], "| PRED:", H[i]["pred"],
                  "| matches gold:", ok)
            print("REPORT:", H[i]["wrong_doc_id"],
                  f"| evidence score {H[i]['evidence_score']:.3f}\n")
            shown += 1
            if shown == 6:
                break
    if shown == 0:
        print("(none: every answer was refused)")