import math

import pandas as pd

LABELS = {"S", "N", "A", "F"}

m = pd.read_csv("results/review_marks.csv", dtype=str, keep_default_na=False)
m["label"] = m["label"].str.strip().str.upper()
bad = m[~m["label"].isin(LABELS)]
if len(bad):
    print(f"{len(bad)} rows still have a missing or invalid label (use S, N, A or F).")
    print("Items to fix:", bad["item"].tolist()[:20])
    raise SystemExit

d = pd.read_csv("results/answer_detail_test.csv")[["id", "hybrid_rag_rnd"]]
m = m.merge(d, on="id", how="left")


def wilson(k, n, z=1.96):
    if n == 0:
        return float("nan"), float("nan")
    p = k / n
    den = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / den
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return centre - half, centre + half


print(f"Reviewed items: {len(m)}")
print("\nLabel counts:")
print(m["label"].value_counts().reindex(["S", "N", "A", "F"], fill_value=0).to_string())

ans = m[m["label"].isin(["S", "N"])]
k = int((ans["label"] == "S").sum())
lo, hi = wilson(k, len(ans))
print(f"\nAnswered questions: {len(ans)}")
print(f"Faithful (supported by the passages): {k}/{len(ans)} = {100 * k / len(ans):.1f}% "
      f"(95% interval {100 * lo:.0f}% to {100 * hi:.0f}%)")

ab = m[m["label"].isin(["A", "F"])]
if len(ab):
    print(f"\nRefusals: {len(ab)} "
          f"({int((ab['label'] == 'A').sum())} appropriate, "
          f"{int((ab['label'] == 'F').sum())} false)")

ans = ans.assign(outcome=ans["hybrid_rag_rnd"].map({1: "correct", 0: "wrong"}))
print("\nFaithfulness vs correctness (answered questions):")
print(pd.crosstab(ans["label"], ans["outcome"]).to_string())

wrong_n = ans[(ans["label"] == "N") & (ans["outcome"] == "wrong")]
wrong_s = ans[(ans["label"] == "S") & (ans["outcome"] == "wrong")]
print(f"\nWrong answers that were NOT supported by the passages: {len(wrong_n)}")
print(f"Wrong answers that WERE supported (reasoning or interpretation errors): {len(wrong_s)}")