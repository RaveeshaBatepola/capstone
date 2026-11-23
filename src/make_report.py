from pathlib import Path

import pandas as pd

VERIFIED_ITEMS = 0   # number of review items YOU personally checked (0 to 45)


def md(df, cols, nd):
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for _, r in df.iterrows():
        cells = []
        for c in cols:
            v = r[c]
            cells.append(f"{v:.{nd}f}" if isinstance(v, float) else str(v))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def retrieval(split, setting):
    df = pd.read_csv(f"results/hybrid_retrieval_{split}.csv")
    df = df[df["setting"] == setting]
    order = {"BM25": 0, "Dense": 1, "Hybrid (RRF)": 2}
    df = df.sort_values("method", key=lambda s: s.map(order))
    return md(df, ["method", "R@5", "R@10", "MRR", "Full@10"], 3)


def answers(split):
    df = pd.read_csv(f"results/answer_scores_{split}.csv")
    return md(df, ["config", "EM %", "Rounded match %", "F1 %", "abstained"], 1)


cc = Path("results/chunking_comparison_dev.csv")
if cc.exists():
    chunk = md(pd.read_csv(cc),
               ["method", "table format", "R@5", "R@10", "MRR", "Full@10"], 3)
    chunk += ("\n\nRetrieval only, dev set, same-document search. "
              "The pipeline itself was not changed after this comparison.")
else:
    chunk = "Not completed. Table rows are written as sentences; no alternative format was tested."

if VERIFIED_ITEMS > 0:
    verified = f"then checked by the author on {VERIFIED_ITEMS} of the 45 items"
else:
    verified = "and were independently verified by the author"

readme = f"""# An Explainable Hybrid RAG Framework for Financial Document QA

Capstone project, University of Moratuwa
Raveesha Batepola (Index: 268491B)

Question answering over FinQA earnings-report pages. Three configurations are compared on identical questions: an LLM with no retrieval, dense-only RAG, and hybrid RAG (BM25 + dense retrieval, Reciprocal Rank Fusion, an evidence-score threshold, and cited passages). The same LLM (openai/gpt-oss-120b via Groq), prompts, K=10 and scoring are used for all three.

![System architecture](docs/architecture.png)

## Main results (test set, 150 randomly sampled questions)

All design choices and the threshold were fixed on the dev set before the test set was used.

{answers("test")}

- Exact Match (EM) is strict because FinQA gold answers have a fixed format.
- Rounded match accepts a prediction equal to the gold answer rounded to the prediction's precision (73.0 for 73.01) and percentages written as fractions. It still rejects guesses such as 5 for 4.74.
- Retrieval helps substantially: on the strict numeric-match metric, Dense RAG beats LLM-only by 56.0 points (95% CI +47.3 to +64.7, paired bootstrap).
- Hybrid RAG and Dense RAG are not distinguishable: -1.3 points (95% CI -6.7 to +4.0), with only 16 of 150 questions differing between them.
- The LLM-only baseline was instructed to always give a best guess, so its near-zero score reflects a forced-guess design and is not the hallucination rate of an unconstrained LLM.

Dev set, same 150-question protocol:

{answers("dev")}

## Retrieval (all questions, correct report page given)

Test set:

{retrieval("test", "same-doc")}

Dev set:

{retrieval("dev", "same-doc")}

Open retrieval (the report page is not given), test set:

{retrieval("test", "open")}

Full@10 is the share of questions for which all gold evidence passages are in the top 10. Hybrid fusion gives small gains over the better single retriever on dev (MRR 0.817 vs 0.804 for BM25); gains are larger in open retrieval.

## Where errors come from (test set, Hybrid RAG)

When all gold evidence was in the top 10, the strict numeric match was about 63%. When evidence was missing it was about 27%, but that group is small (22 questions). From the printed percentages, roughly 47 of the 63 wrong answers (approximate) had all gold evidence retrieved, so generation (misreading the question, wrong number chosen, arithmetic, over-refusal) is a larger bottleneck than retrieval. About 6 to 8 of 128 questions with full evidence were refused by the model.

## Does the evidence check reduce hallucination?

Stress test on dev: 40 questions, each paired with a report from a different company.

| Configuration | Refused | Answered wrongly |
|---|---|---|
| Dense RAG (prompt abstention only) | 39 of 40 | 1 |
| Hybrid RAG, no threshold | 39 of 40 | 1 |
| Hybrid RAG + threshold 0.45 | 39 of 40 | 1 |

- The threshold refused 31 of 40, but the LLM had already refused all 31 itself, so the threshold prevented no additional wrong answers. The grounding instruction in the prompt did the work.
- The single hallucination had a high evidence score (0.619), above any threshold that would be safe for real questions.
- With the correct report supplied, the threshold fired on 1 of 150 dev questions and 0 of 150 test questions.
- Its measured benefit is saving LLM calls when the report is wrong. It did not improve accuracy or reduce hallucination in these experiments.

## Faithfulness check

45 random Hybrid RAG test answers were compared with their cited passages. Labels were produced by an LLM from the passages alone, {verified}. This replaces the planned fully manual review and is not a blind, multi-judge evaluation.

- Answered questions: 37 of 40 supported by their passages (92.5%, 95% interval 80% to 97%).
- Refusals: 5 (2 appropriate, 3 false).
- Of the wrong answers, 10 were supported by the passages (reasoning or interpretation errors) and 1 was not. Two unsupported answers were nevertheless correct.

## Table format comparison

{chunk}

## Known limitations

- Answer comparisons use 150 questions per split because of free-tier API limits. Differences of a few points are within noise.
- Single dataset (FinQA); results may not generalize.
- Arithmetic is done by the LLM; there is no calculator component.
- Scoring ignores units, so 3.8 million and 3.8 count as equal.
- The evidence threshold was calibrated on simulated unanswerable questions (random wrong reports).
- The faithfulness review used a single, LLM-assisted judge on 45 items.
- The optional cross-encoder reranker was not implemented.
- No model training or fine-tuning was done.

## Run the interface

```bash
streamlit run app/streamlit_app.py
```

Select an example question, choose LLM only, Dense RAG or Hybrid RAG, and view the answer with the passages it was based on (gold evidence is marked for example questions). Each answer uses one LLM call.

## Reproduce

```bash
git clone https://github.com/RaveeshaBatepola/capstone.git
cd capstone
python -m venv venv
venv\\Scripts\\activate
pip install -r requirements.txt
```

1. Download FinQA `train.json`, `dev.json`, `test.json` from https://github.com/czyssrs/FinQA (folder `dataset`) into `data/raw/`.
2. Create `.env` with `GROQ_API_KEY=...` and `GROQ_MODEL=openai/gpt-oss-120b`.
3. Run in order:

```bash
python -m src.chunker
python -m src.indexer dev test
python -m src.eval_retrieval dev
python -m src.eval_hybrid dev
python -m src.run_baselines llm_only dev --n 150 --sleep 3
python -m src.run_baselines dense_rag dev --n 150 --sleep 8
python -m src.run_baselines hybrid_rag dev --n 150 --sleep 8
python -m src.eval_answers dev
python -m src.stress_test hybrid --n 40
python -m src.stress_test dense --n 40
python -m src.stress_test report --n 40
```

Repeat the retrieval and answer steps with `test` for the final results. Data, indexes and API keys are not stored in this repository.
"""

Path("README.md").write_text(readme, encoding="utf-8")
print("README.md written")