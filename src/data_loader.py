import json
from pathlib import Path

RAW_DIR = Path("data/raw")


def load_split(split: str):
    """Load a FinQA split: 'train', 'dev', or 'test'."""
    with open(RAW_DIR / f"{split}.json", "r", encoding="utf-8") as f:
        return json.load(f)


if __name__ == "__main__":
    for split in ["train", "dev", "test"]:
        data = load_split(split)
        print(f"{split}: {len(data)} examples")

    ex = load_split("train")[0]
    print("\nKeys:", list(ex.keys()))
    print("QA keys:", list(ex["qa"].keys()))
    print("\nID:", ex["id"])
    print("Question:", ex["qa"]["question"])
    print("Answer:", ex["qa"]["answer"])
    print("Gold evidence:", ex["qa"].get("gold_inds"))
    print("Pre-text sentences:", len(ex["pre_text"]))
    print("Table rows:", len(ex["table"]))