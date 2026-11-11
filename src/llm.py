import truststore
truststore.inject_into_ssl()

import os
import re
import time

from dotenv import load_dotenv
from groq import Groq

load_dotenv()

MODEL = os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")
_client = None
USED_TOKENS = 0


def _get_client():
    global _client
    if _client is None:
        key = os.getenv("GROQ_API_KEY")
        if not key:
            raise RuntimeError("GROQ_API_KEY not found. Check your .env file.")
        _client = Groq(api_key=key)
    return _client


def _call(prompt, with_effort=True):
    kwargs = dict(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0,
        max_completion_tokens=1200,
    )
    if with_effort:
        kwargs["reasoning_effort"] = "low"   # keeps token use down
        global USED_TOKENS
        resp = _get_client().chat.completions.create(**kwargs)
        if getattr(resp, "usage", None):
            USED_TOKENS += resp.usage.total_tokens
    return (resp.choices[0].message.content or "").strip()


def generate(prompt: str, max_retries: int = 5) -> str:
    """Call the LLM at temperature 0. Handles short rate limits, stops on daily limit."""
    with_effort = True
    for attempt in range(max_retries):
        try:
            return _call(prompt, with_effort)
        except Exception as e:
            msg = str(e)
            low = msg.lower()
            if with_effort and ("reasoning_effort" in low or "reasoning effort" in low):
                with_effort = False          # model does not support the option
                continue
            if "429" in msg or "rate limit" in low:
                if "per day" in low or "(tpd)" in low or "(rpd)" in low:
                    raise RuntimeError(
                        "Daily limit reached. Detail: " + msg[:400] +
                        "\nStop now; rerun the same command tomorrow to resume.")
                m = re.search(r"try again in ([0-9.]+)s", msg)
                wait = float(m.group(1)) + 1 if m else 15 * (attempt + 1)
                print(f"  rate limited, waiting {wait:.0f}s...")
                time.sleep(wait)
            elif "503" in msg or "500" in msg:
                time.sleep(10)
            else:
                raise
    raise RuntimeError("Too many retries (still rate limited)")