# src/models/openai_client.py
from ._common import clean_text, get_env_or_raise, backoff_sleep
import time
from typing import Optional

def query_openai_4o(
    prompt: str,
    model: str = "gpt-4o",
    temperature: float = 0.0,
    top_p: float = 1.0,
    max_tokens: int = 64,
    timeout: int = 30,
    retries: int = 2,
):
    """
    Returns a normalized dict:
      version, output, finish_reason, usage, error
    """
    import openai
    openai.api_key = get_env_or_raise("OPENAI_API_KEY")

    last_err: Optional[str] = None
    for attempt in range(retries + 1):
        t0 = time.time()
        try:
            client = openai.OpenAI(timeout=timeout)
            resp = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
                top_p=top_p,
                max_tokens=max_tokens,
            )
            c0 = resp.choices[0]
            text = c0.message.content or ""
            out = {
                "version": model,
                "output": clean_text(text),
                "finish_reason": getattr(c0, "finish_reason", None),
                "usage": {
                    "prompt_tokens": getattr(resp.usage, "prompt_tokens", None),
                    "completion_tokens": getattr(resp.usage, "completion_tokens", None),
                    "total_tokens": getattr(resp.usage, "total_tokens", None),
                },
                "error": None,
            }
            return out
        except Exception as e:
            last_err = str(e)
            if attempt < retries:
                backoff_sleep(attempt)
            else:
                return {
                    "version": model,
                    "output": "",
                    "finish_reason": None,
                    "usage": {"prompt_tokens": None, "completion_tokens": None, "total_tokens": None},
                    "error": last_err,
                }
        finally:
            _ = int((time.time() - t0) * 1000)