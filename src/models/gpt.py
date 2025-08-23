# src/models/openai_client.py

import os
import time
from datetime import datetime
from openai import OpenAI

import os
from dotenv import load_dotenv

# Load .env from project root (where src is sibling)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
print(BASE_DIR)
load_dotenv(dotenv_path=os.path.join(BASE_DIR, "..", ".env"))

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise RuntimeError("OPENAI_API_KEY not found in environment")

client = OpenAI(api_key=OPENAI_API_KEY)

def query_openai(prompt: str,
                 run_id: int,
                 qid: str,
                 domain: str,
                 setting: str,
                 temperature: float = 0.0,
                 top_p: float = 1.0,
                 max_tokens: int = 64,
                 seed: int | None = None,
                 timeout: int = 60) -> dict:
    """
    Query OpenAI GPT-4o with a text prompt.
    Returns dict ready for experiment logging.
    """

    start = time.time()
    ts = datetime.utcnow().isoformat()

    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
            seed=seed,
            timeout=timeout
        )
        latency_ms = int((time.time() - start) * 1000)

        choice = response.choices[0]
        text = choice.message.content.strip()

        return {
            "run_id": run_id,
            "qid": qid,
            "domain": domain,
            "model": "gpt4o",
            "setting": setting,
            "prompt": prompt,
            "output": text,
            "latency_ms": latency_ms,
            "ts": ts,
            "version": "gpt-4o",
            "finish_reason": choice.finish_reason,
            "usage": {
                "prompt_tokens": response.usage.prompt_tokens if response.usage else None,
                "completion_tokens": response.usage.completion_tokens if response.usage else None,
                "total_tokens": response.usage.total_tokens if response.usage else None,
            },
            "error": None,
        }

    except Exception as e:
        latency_ms = int((time.time() - start) * 1000)
        return {
            "run_id": run_id,
            "qid": qid,
            "domain": domain,
            "model": "gpt4o",
            "setting": setting,
            "prompt": prompt,
            "output": "",
            "latency_ms": latency_ms,
            "ts": ts,
            "version": "gpt-4o",
            "finish_reason": "error",
            "usage": None,
            "error": str(e),
        }