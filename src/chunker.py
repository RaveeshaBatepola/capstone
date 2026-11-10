import json
from pathlib import Path

from tqdm import tqdm

from src.data_loader import load_split

PROC_DIR = Path("data/processed")
PROC_DIR.mkdir(parents=True, exist_ok=True)


def table_row_to_text(header, row):
    """Turn one table row into sentence-style text."""
    label = row[0].strip()
    parts = []
    for j in range(1, len(row)):
        val = row[j].strip()
        if not val:
            continue
        col = header[j].strip() if j < len(header) else ""
        if col:
            parts.append(f"the {label} of {col} is {val}")
        else:
            parts.append(f"the {label} is {val}")
    return " ; ".join(parts)


def build_chunks_for_example(ex):
    """Return a list of chunk dicts for one FinQA example."""
    filename = ex["filename"]            # e.g. ADI/2009/page_49.pdf
    company, year, page = (filename.split("/") + ["", "", ""])[:3]
    meta = {"doc_id": filename, "company": company,
            "year": year, "page": page.replace(".pdf", "")}

    chunks = []
    pre, post = ex["pre_text"], ex["post_text"]

    # text keys: pre_text first, then post_text continues the numbering
    for i, s in enumerate(pre):
        chunks.append((f"text_{i}", "text", s))
    for j, s in enumerate(post):
        chunks.append((f"text_{len(pre) + j}", "text", s))

    # table keys: row number
    table = ex["table"]
    header = table[0] if table else []
    for i, row in enumerate(table):
        if i == 0:
            text = " | ".join(c.strip() for c in row if c.strip())
        else:
            text = table_row_to_text(header, row)
        chunks.append((f"table_{i}", "table", text))

    out = []
    for key, ctype, text in chunks:
        text = text.strip()
        if len(text) <= 2:       # skip junk like "."
            continue
        out.append({"chunk_id": f"{filename}::{key}", "key": key,
                    "type": ctype, "text": text, **meta})
    return out


def process_split(split):
    data = load_split(split)
    chunks = {}      # dedupe: several questions share the same report page
    questions = []

    for ex in tqdm(data, desc=split):
        for c in build_chunks_for_example(ex):
            chunks[c["chunk_id"]] = c

        filename = ex["filename"]
        gold_keys = list(ex["qa"].get("gold_inds", {}).keys())
        questions.append({
            "id": ex["id"],
            "doc_id": filename,
            "question": ex["qa"]["question"],
            "answer": str(ex["qa"]["answer"]),
            "exe_ans": ex["qa"].get("exe_ans"),
            "program": ex["qa"].get("program"),
            "gold_chunk_ids": [f"{filename}::{k}" for k in gold_keys],
        })

    with open(PROC_DIR / f"chunks_{split}.jsonl", "w", encoding="utf-8") as f:
        for c in chunks.values():
            f.write(json.dumps(c) + "\n")
    with open(PROC_DIR / f"questions_{split}.jsonl", "w", encoding="utf-8") as f:
        for q in questions:
            f.write(json.dumps(q) + "\n")

    # sanity check: does every gold evidence id exist as a chunk?
    total = found = 0
    for q in questions:
        for g in q["gold_chunk_ids"]:
            total += 1
            found += g in chunks
    print(f"{split}: {len(questions)} questions, {len(chunks)} unique chunks")
    print(f"{split}: gold evidence matched to chunks: {found}/{total} "
          f"({100 * found / max(total, 1):.2f}%)")


if __name__ == "__main__":
    for split in ["train", "dev", "test"]:
        process_split(split)