from src.llm import MODEL, generate, _get_client

print("Model:", MODEL)
try:
    print("Reply:", generate("What is 12 divided by 4? Answer with just the number."))
except Exception as e:
    print("\nERROR:", e)
    print("\nModels your key can use:")
    for m in _get_client().models.list():
        print(" ", m.name)