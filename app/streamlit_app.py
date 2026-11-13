import os
import json
import random
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))

import streamlit as st

from src.hybrid import hybrid_search
from src.llm import generate
from src.prompts import LLM_ONLY_PROMPT, RAG_PROMPT, format_evidence
from src.retriever import Retriever

st.set_page_config(page_title="Financial QA with evidence", layout="wide")


@st.cache_resource(show_spinner="Loading indexes...")
def load_retriever():
    r = Retriever("dev")
    meta = {c["chunk_id"]: c for c in r.chunks}
    docs = sorted(set(c["doc_id"] for c in r.chunks))
    return r, meta, docs


@st.cache_data
def load_examples():
    with open("data/processed/questions_dev.jsonl", encoding="utf-8") as f:
        qs = [json.loads(line) for line in f]
    random.Random(42).shuffle(qs)
    return qs[:150]


def parse_final(text):
    m = re.findall(r"FINAL ANSWER:\s*(.+)", text, flags=re.IGNORECASE)
    if m:
        return m[-1].strip().strip("*").strip()
    lines = [l for l in text.splitlines() if l.strip()]
    return lines[-1].strip() if lines else ""


retriever, meta, docs = load_retriever()
examples = load_examples()

st.title("Financial document QA with evidence")
st.caption("FinQA dev reports | Hybrid RAG: BM25 + dense retrieval, rank fusion, "
           "evidence check, cited passages")

with st.sidebar:
    mode = st.radio("System", ["Hybrid RAG", "Dense RAG", "LLM only"])
    k = st.slider("Passages given to the LLM (K)", 3, 15, 10)
    threshold = st.slider("Evidence threshold (Hybrid only)", 0.0, 0.8, 0.45, 0.05)
    st.info("Each answer uses one LLM call, about 800 tokens of the free quota.")

ex_idx = st.selectbox(
    "Example question (from the 150-question evaluation sample)",
    range(len(examples) + 1),
    format_func=lambda i: "(write my own question)" if i == 0
    else examples[i - 1]["question"][:110],
)
ex = examples[ex_idx - 1] if ex_idx else None
tag = ex["id"] if ex else "own"

question = st.text_area("Question", value=ex["question"] if ex else "", key=f"q_{tag}")
doc_id = st.selectbox("Report page to search", docs,
                      index=docs.index(ex["doc_id"]) if ex else 0, key=f"d_{tag}")

if st.button("Answer", type="primary") and question.strip():
    gold_ids = set(ex["gold_chunk_ids"]) if ex else set()
    hits, ev_score, abstained, raw, answer = [], None, False, "", ""
    try:
        with st.spinner("Retrieving and answering..."):
            if mode == "LLM only":
                raw = generate(LLM_ONLY_PROMPT.format(question=question))
                answer = parse_final(raw)
            else:
                if mode == "Dense RAG":
                    res = retriever.search_dense(question, k=k, doc_id=doc_id)
                    hits = [{"chunk_id": c, "dense": s} for c, s in res]
                else:
                    hits, ev_score = hybrid_search(retriever, question, k=k, doc_id=doc_id)

                if mode == "Hybrid RAG" and ev_score < threshold:
                    abstained = True
                    answer = "insufficient evidence"
                else:
                    passages = [(h["chunk_id"], retriever.text_of(h["chunk_id"]))
                                for h in hits]
                    raw = generate(RAG_PROMPT.format(
                        evidence=format_evidence(passages), question=question))
                    answer = parse_final(raw)
    except RuntimeError as e:
        st.error(str(e))
        st.stop()

    st.subheader("Answer")
    if abstained or "insufficient evidence" in answer.lower():
        st.warning(answer)
    else:
        st.success(answer)
    if abstained:
        st.write(f"The evidence check refused to answer: the best passage scored "
                 f"{ev_score:.3f}, below the threshold of {threshold:.2f}.")
    if ex:
        st.caption(f"Gold answer from FinQA: {ex['answer']}")
    if ev_score is not None and not abstained:
        st.caption(f"Evidence score: {ev_score:.3f} (threshold {threshold:.2f})")

    if hits:
        st.subheader("Evidence (the passages the answer was based on)")
        for i, h in enumerate(hits, 1):
            c = meta[h["chunk_id"]]
            flag = "  | GOLD EVIDENCE" if h["chunk_id"] in gold_ids else ""
            with st.container(border=True):
                st.markdown(f"**[{i}] {h['chunk_id']}** ({c['type']}){flag}")
                st.write(c["text"])
                scores = [f"dense {h['dense']:.3f}"]
                if "bm25" in h:
                    scores += [f"bm25 {h['bm25']:.2f}", f"rrf {h['rrf']:.4f}"]
                st.caption(" | ".join(scores))

    if raw:
        with st.expander("Model reasoning"):
            st.text(raw)