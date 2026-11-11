from src.llm import MODEL, generate

print("Model:", MODEL)
print("Reply:", generate("What is 12 divided by 4? Answer with just the number."))