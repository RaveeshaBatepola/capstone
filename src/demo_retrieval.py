import json

from src.retriever import Retriever

r = Retriever("dev")

with open("data/processed/questions_dev.jsonl", encoding="utf-8") as f:
    questions = [json.loads(line) for line in f]

q = questions[0]
print("QUESTION:", q["question"])
print("GOLD    :", q["gold_chunk_ids"])
for g in q["gold_chunk_ids"]:
    print("   ", r.text_of(g)[:120])

for name, fn in [("BM25", r.search_bm25), ("DENSE", r.search_dense)]:
    for label, doc in [("open", None), ("same-doc", q["doc_id"])]:
        print(f"\n--- {name} ({label}) top 3 ---")
        for cid, score in fn(q["question"], k=3, doc_id=doc):
            mark = "  <-- GOLD" if cid in q["gold_chunk_ids"] else ""
            print(f"{score:.3f}  {cid}{mark}")
            print("      ", r.text_of(cid)[:100])