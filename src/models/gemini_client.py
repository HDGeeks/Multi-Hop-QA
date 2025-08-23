# src/models/gemini_client.py
from ._common import clean_text, get_env_or_raise, backoff_sleep
import time
from typing import Optional

def query_gemini(
    prompt: str,
    model: str = "gemini-1.5-flash",
    temperature: float = 0.0,
    top_p: float = 1.0,
    max_tokens: int = 64,
    timeout: int = 30,
    retries: int = 2,
):
    """
    Normalized dict: version, output, finish_reason, usage, error
    """
    import google.generativeai as genai
    genai.configure(api_key=get_env_or_raise("GOOGLE_API_KEY"))

    last_err: Optional[str] = None
    for attempt in range(retries + 1):
        t0 = time.time()
        try:
            model_obj = genai.GenerativeModel(model)
            resp = model_obj.generate_content(
                prompt,
                generation_config={
                    "temperature": temperature,
                    "top_p": top_p,
                    "max_output_tokens": max_tokens,
                },
                safety_settings=None,
                request_options={"timeout": timeout},
            )
            # Gemini returns text in .text
            text = getattr(resp, "text", "") or ""
            # usage fields may not be present consistently
            usage = {"prompt_tokens": None, "completion_tokens": None, "total_tokens": None}
            out = {
                "version": model,
                "output": clean_text(text),
                "finish_reason": None,  # Gemini SDK doesn’t always expose this
                "usage": usage,
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