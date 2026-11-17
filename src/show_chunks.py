import json

with open("data/processed/chunks_train.jsonl", encoding="utf-8") as f:
    chunks = [json.loads(line) for line in f]

print("Total chunks:", len(chunks))

texts = [c for c in chunks if c["type"] == "text"][:3]
tables = [c for c in chunks if c["type"] == "table"][:4]

print("\n--- TEXT CHUNKS ---")
for c in texts:
    print(c["chunk_id"], "->", c["text"][:150])

print("\n--- TABLE CHUNKS ---")
for c in tables:
    print(c["chunk_id"], "->", c["text"][:200])