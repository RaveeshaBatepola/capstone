LLM_ONLY_PROMPT = """You are a financial analyst answering a question about a company's annual report. You do not have the report. Give your single best estimate. Never refuse and never answer "unknown".

Question: {question}

Reason briefly (at most 3 short lines), then finish with one last line in exactly this form:
FINAL ANSWER: <answer>
The final answer must be one single value, not a list. For numeric answers write only the number (add % only if the answer is a percentage)."""


RAG_PROMPT = """Answer the question using ONLY the evidence below, which comes from a company's annual report. Do not use outside knowledge. If the evidence does not contain what is needed, finish with: FINAL ANSWER: insufficient evidence

Evidence:
{evidence}

Question: {question}

Do any needed arithmetic step by step (at most 5 short lines), then finish with one last line in exactly this form:
FINAL ANSWER: <answer>
The final answer must be one single value, not a list. For numeric answers write only the number (add % only if the answer is a percentage)."""


def format_evidence(passages):
    """passages: list of (chunk_id, text)."""
    return "\n".join(f"[{i + 1}] {text}" for i, (_, text) in enumerate(passages))