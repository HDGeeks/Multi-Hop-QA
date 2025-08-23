# src/models/qwen_client.py
import os
import time
from dotenv import load_dotenv
from huggingface_hub import InferenceClient

# Load env (so HUGGINGFACE_API_KEY is available whether you call from run.py or REPL)
load_dotenv()

HF_TOKEN = os.getenv("HUGGINGFACE_API_KEY") or os.getenv("HF_TOKEN")
if not HF_TOKEN:
    raise RuntimeError("HUGGINGFACE_API_KEY (or HF_TOKEN) not found in environment")

# Default model: Qwen 2.5 7B Instruct (chat-tuned)
HF_MODEL_QWEN = os.getenv("HF_MODEL_QWEN", "Qwen/Qwen2.5-7B-Instruct")

# Single shared client is fine (stateless, thread-safe for simple use)
_client = InferenceClient(api_key=HF_TOKEN)

def get_version() -> str:
    """Return the model identifier we’re calling (for logging)."""
    return HF_MODEL_QWEN

def query_qwen(
    prompt: str,
    *,
    max_tokens: int = 128,
    temperature: float = 0.0,
    top_p: float = 1.0,
    model: str = None,
):
    """
    Call Qwen chat completion via Hugging Face Inference API.

    Returns:
        text (str): model output (stripped)
        err (str): empty on success, otherwise the exception string
        latency_ms (int): round-trip latency in milliseconds
    """
    t0 = time.time()
    model_to_use = model or HF_MODEL_QWEN
    try:
        resp = _client.chat_completion(
            model=model_to_use,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
            stream=False,
        )
        # HF chat_completion returns an object with .choices[0].message.content
        text = (resp.choices[0].message.content or "").strip()
        err = ""
    except Exception as e:
        text, err = "", str(e)

    latency_ms = int((time.time() - t0) * 1000)
    return text, err, latency_ms