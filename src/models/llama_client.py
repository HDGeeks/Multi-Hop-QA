import os, time
from typing import Dict, Any, Optional
from dotenv import load_dotenv
from huggingface_hub import InferenceClient

# Load .env from project root
load_dotenv()

HF_TOKEN = os.getenv("HUGGINGFACE_API_KEY")
if not HF_TOKEN:
    raise RuntimeError("HUGGINGFACE_API_KEY not found in environment")

# Default LLaMA instruct model; override via param if you like
DEFAULT_LLAMA_MODEL = "meta-llama/Llama-3.1-8B-Instruct"

# Single, re-usable client
_client = InferenceClient(api_key=HF_TOKEN)

def query_llama(
    prompt: str,
    *,
    temperature: float = 0.7,
    top_p: float = 0.9,
    max_tokens: int = 64,
    model: Optional[str] = None
) -> Dict[str, Any]:
    """
    Query LLaMA via HF Inference API (chat_completion).
    Returns a unified schema dict.
    """
    model_id = model or DEFAULT_LLAMA_MODEL
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
        # HF chat_completion returns choices similar to OpenAI
        choice = resp.choices[0] if resp and resp.choices else None
        output = (choice.message.content or "").strip() if choice else ""
        finish_reason = getattr(choice, "finish_reason", None) or "stop"

        # Token usage is not always available on HF endpoint
        usage = {}
        result = {
            "output": output,
            "error": None,
            "latency_ms": int((time.time() - t0) * 1000),
            "version": model_id,
            "finish_reason": finish_reason,
            "usage": usage,
        }
        return result

    except Exception as e:
        return {
            "output": "",
            "error": f"{e}",
            "latency_ms": int((time.time() - t0) * 1000),
            "version": model_id,
            "finish_reason": "",
            "usage": {},
        }