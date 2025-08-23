import os, time
from typing import Dict, Any, Optional
from dotenv import load_dotenv
from huggingface_hub import InferenceClient

# Load .env from project root
load_dotenv()

HF_TOKEN = os.getenv("HUGGINGFACE_API_KEY")
if not HF_TOKEN:
    raise RuntimeError("HUGGINGFACE_API_KEY not found in environment")

# Default Mistral instruct model
DEFAULT_MISTRAL_MODEL = "mistralai/Mistral-7B-Instruct-v0.3"

_client = InferenceClient(api_key=HF_TOKEN)

def query_mistral(
    prompt: str,
    *,
    temperature: float = 0.0,
    top_p: float = 1.0,
    max_tokens: int = 64,
    model: Optional[str] = None
) -> Dict[str, Any]:
    """
    Query Mistral via HF Inference API (chat_completion).
    Returns a unified schema dict.
    """
    model_id = model or DEFAULT_MISTRAL_MODEL
    t0 = time.time()
    try:
        resp = _client.chat_completion(
            model=model_id,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
            stream=False,
        )
        choice = resp.choices[0] if resp and resp.choices else None
        output = (choice.message.content or "").strip() if choice else ""
        finish_reason = getattr(choice, "finish_reason", None) or "stop"

        usage = {}
        return {
            "output": output,
            "error": None,
            "latency_ms": int((time.time() - t0) * 1000),
            "version": model_id,
            "finish_reason": finish_reason,
            "usage": usage,
        }
    except Exception as e:
        return {
            "output": "",
            "error": f"{e}",
            "latency_ms": int((time.time() - t0) * 1000),
            "version": model_id,
            "finish_reason": "",
            "usage": {},
        }