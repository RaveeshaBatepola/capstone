import json
import re
from collections import Counter

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

SPLIT = "dev"
MODES = ["llm_only", "dense_rag", "hybrid_rag"]
NAMES = {"llm_only": "LLM only", "dense_rag": "Dense RAG", "hybrid_rag": "Hybrid RAG"}
NUM = re.compile(r"-?\d+(?:\.\d+)?")


def read(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f]


# ---------- normalization and scoring ----------
def norm(s):
    s = str(s).lower().strip()
    s = s.replace("$", "").replace(",", "").replace("%", "")
    s = re.sub(r"\b(million|millions|billion|billions|thousand|usd)\b", "", s)
    s = re.sub(r"\s+", " ", s).strip()
    if re.fullmatch(r"-?\d+(\.\d+)?", s) and "." in s:
        s = s.rstrip("0").rstrip(".")
    return s


def parse_num(s):
    """(value, decimals, has_percent) from the first number in s, or None."""
    t = str(s).replace(",", "").replace("$", "")
    m = NUM.search(t)
    if not m:
        return None
    txt = m.group(0)
    dec = len(txt.split(".")[1]) if "." in txt else 0
    return float(txt), dec, "%" in str(s)


def numeric_match(pred, gold_ans, exe_ans):
    """Pred, rounded to the gold answer's own precision, equals the gold answer.
    A percentage written as a fraction (0.0474 for 4.74%) is also accepted."""
    p, g = parse_num(pred), parse_num(gold_ans)
    if p is None or g is None:
        return norm(pred) == norm(gold_ans)       # non-numeric gold, e.g. yes/no
    pv, _, pp = p
    gv, gd, gp = g
    tol = 0.5 * 10 ** (-gd) + 1e-9
    if abs(pv - gv) <= tol:
        return True
    if gp and not pp and abs(pv * 100 - gv) <= tol:
        return True
    return False


def rounded_match(pred, gold_ans, exe_ans):
    """Strict match, OR pred has >= 1 decimal and equals gold rounded to pred's precision."""
    if numeric_match(pred, gold_ans, exe_ans):
        return True
    p, g = parse_num(pred), parse_num(gold_ans)
    if p is None or g is None:
        return False
    pv, pdec, _ = p
    gv, gd, _ = g
    if pdec < 1 or pdec >= gd:
        return False
    return abs(round(gv, pdec) - pv) < 1e-9



def token_f1(pred, gold):
    a, b = norm(pred).split(), norm(gold).split()
    if not a or not b:
        return float(a == b)
    common = sum((Counter(a) & Counter(b)).values())
    if common == 0:
        return 0.0
    pr, rc = common / len(a), common / len(b)
    return 2 * pr * rc / (pr + rc)


def is_abstain(p):
    return "insufficient evidence" in str(p).lower()


# ---------- load predictions ----------
data = {m: {r["id"]: r for r in read(f"results/preds_{m}_{SPLIT}.jsonl")} for m in MODES}
ids = sorted(set.intersection(*[set(d) for d in data.values()]))
print(f"Questions with predictions from all three configurations: {len(ids)}")

rows = []
for i in ids:
    base = data["llm_only"][i]
    row = {"id": i, "question": base["question"], "gold": base["gold_answer"]}
    for m in MODES:
        rec = data[m][i]
        pred = rec["pred"]
        row[f"{m}_pred"] = pred
        row[f"{m}_em"] = int(norm(pred) == norm(rec["gold_answer"]))
        row[f"{m}_num"] = int(numeric_match(pred, rec["gold_answer"], rec["gold_exe_ans"]))
        row[f"{m}_rnd"] = int(rounded_match(pred, rec["gold_answer"], rec["gold_exe_ans"]))
        row[f"{m}_f1"] = token_f1(pred, rec["gold_answer"])
        row[f"{m}_abstain"] = int(is_abstain(pred))
        if m != "llm_only":
            got = {h["chunk_id"] for h in rec["retrieved"]}
            row[f"{m}_full_hit"] = int(all(g in got for g in rec["gold_chunk_ids"]))
    rows.append(row)
df = pd.DataFrame(rows)
df.to_csv(f"results/answer_detail_{SPLIT}.csv", index=False)

# ---------- main table ----------
summary = []
for m in MODES:
    summary.append({
        "config": NAMES[m],
        "EM %": round(100 * df[f"{m}_em"].mean(), 1),
        "Numeric match %": round(100 * df[f"{m}_num"].mean(), 1),
        "Rounded match %": round(100 * df[f"{m}_rnd"].mean(), 1),
        "F1 %": round(100 * df[f"{m}_f1"].mean(), 1),
        "abstained": int(df[f"{m}_abstain"].sum()),
    })
sdf = pd.DataFrame(summary)
print("\n=== Answer accuracy (150 dev questions) ===")
print(sdf.to_string(index=False))
sdf.to_csv(f"results/answer_scores_{SPLIT}.csv", index=False)

# ---------- retrieval vs generation errors ----------
print("\n=== Did the retrieved top 10 contain ALL gold evidence? ===")
for m in ["dense_rag", "hybrid_rag"]:
    hit = df[df[f"{m}_full_hit"] == 1]
    miss = df[df[f"{m}_full_hit"] == 0]
    print(f"{NAMES[m]}: full evidence retrieved for {len(hit)}/{len(df)} questions")
    print(f"   numeric match when evidence retrieved : {100 * hit[f'{m}_num'].mean():.1f}%  (n={len(hit)})")
    if len(miss):
        print(f"   numeric match when evidence missing   : {100 * miss[f'{m}_num'].mean():.1f}%  (n={len(miss)})")

# ---------- paired bootstrap ----------
rng = np.random.default_rng(0)


def compare(a, b, label):
    x, y = df[f"{a}_num"].to_numpy(), df[f"{b}_num"].to_numpy()
    diff = y - x
    boots = [diff[rng.integers(0, len(diff), len(diff))].mean() for _ in range(5000)]
    lo, hi = np.percentile(boots, [2.5, 97.5])
    print(f"\n{label}: {100 * diff.mean():+.1f} points "
          f"(95% CI {100 * lo:+.1f} to {100 * hi:+.1f})")
    print(f"   both right {int(((x == 1) & (y == 1)).sum())} | "
          f"only {NAMES[b]} right {int(((x == 0) & (y == 1)).sum())} | "
          f"only {NAMES[a]} right {int(((x == 1) & (y == 0)).sum())} | "
          f"both wrong {int(((x == 0) & (y == 0)).sum())}")


print("\n=== Paired comparison on numeric match ===")
compare("llm_only", "dense_rag", "Dense RAG minus LLM only")
compare("dense_rag", "hybrid_rag", "Hybrid RAG minus Dense RAG")
compare("llm_only", "hybrid_rag", "Hybrid RAG minus LLM only")

# ---------- look at failures ----------
print("\n=== 8 random Hybrid RAG failures ===")
bad = df[df["hybrid_rag_num"] == 0].sample(n=min(8, (df["hybrid_rag_num"] == 0).sum()),
                                           random_state=1)
for _, r in bad.iterrows():
    print("Q   :", r["question"])
    print("GOLD:", r["gold"], "| PRED:", r["hybrid_rag_pred"],
          "| evidence fully retrieved:", bool(r["hybrid_rag_full_hit"]))
    print()

# ---------- abstention breakdown ----------
print("\n=== Abstention breakdown (RAG configurations) ===")
for m in ["dense_rag", "hybrid_rag"]:
    ans = df[df[f"{m}_abstain"] == 0]
    print(f"{NAMES[m]}: answered {len(ans)}/{len(df)}; "
          f"numeric match among answered: {100 * ans[f'{m}_num'].mean():.1f}%")
    miss = df[df[f"{m}_full_hit"] == 0]
    if len(miss):
        abst = int(miss[f"{m}_abstain"].sum())
        wrong = int(((miss[f"{m}_abstain"] == 0) & (miss[f"{m}_num"] == 0)).sum())
        right = int(miss[f"{m}_num"].sum())
        print(f"   evidence missing (n={len(miss)}): abstained {abst}, "
              f"answered wrong {wrong}, answered right {right}")




# ---------- sanity check: LLM-only 'matches' that EM does not count ----------
lenient = df[(df["llm_only_num"] == 1) & (df["llm_only_em"] == 0)]
print(f"\n=== LLM-only: {len(lenient)} numeric matches that EM rejects (up to 12 shown) ===")
for _, r in lenient.head(12).iterrows():
    print(f"   gold {r['gold']!s:>10}   pred {r['llm_only_pred']!s}")

# ---------- false abstentions: refused although all gold evidence was retrieved ----------
print("\n=== False abstentions (evidence fully retrieved but system refused) ===")
for m in ["dense_rag", "hybrid_rag"]:
    fa = int(((df[f"{m}_abstain"] == 1) & (df[f"{m}_full_hit"] == 1)).sum())
    print(f"{NAMES[m]}: {fa} of {int(df[f'{m}_full_hit'].sum())} questions with full evidence")




# ---------- chart ----------
ax = sdf.set_index("config")[["EM %", "Numeric match %", "F1 %"]].plot(
    kind="bar", figsize=(7.5, 4.5), rot=0)
ax.set_ylim(0, 100)
ax.set_ylabel("%")
ax.set_title(f"Answer accuracy on {len(df)} {SPLIT} questions")
plt.tight_layout()
plt.savefig(f"results/figures/09_answer_accuracy_{SPLIT}.png", dpi=150)
print(f"saved results/figures/09_answer_accuracy_{SPLIT}.png")