from src.data_loader import load_split

ex = load_split("train")[0]

print("ID:", ex["id"])
print("Filename:", ex["filename"])

print("\n--- PRE_TEXT (first 3) ---")
for i, s in enumerate(ex["pre_text"][:3]):
    print(i, s)

print("\n--- TABLE ---")
for i, row in enumerate(ex["table"]):
    print(i, row)

print("\n--- POST_TEXT (first 3) ---")
for i, s in enumerate(ex["post_text"][:3]):
    print(i, s)

print("\n--- QA ---")
print("Question:", ex["qa"]["question"])
print("Answer:", ex["qa"]["answer"])
print("Gold evidence:", ex["qa"]["gold_inds"])
print("Program:", ex["qa"]["program"])