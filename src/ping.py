import os
from datetime import datetime
from pprint import PrettyPrinter

from models.openai_client import query_openai
from models.gemini_client import query_gemini
from models.llama_client import query_llama
from models.mistral_client import query_mistral
from models.openai_mini_client import query_openai_mini 
from models.gemini_pro_client import query_gemini_pro
pp = PrettyPrinter(indent=2, width=100)

def _as_structured_dict(
    output_text: str = "",
    err: str | None = None,
    version: str | None = None,
    finish_reason: str | None = "stop",
    usage: dict | None = None,
    latency_ms: int | None = None,
):
    return {
        "error": None if not err else err,
        "finish_reason": finish_reason if not err else None,
        "output": output_text or "",
        "usage": usage or {},
        "version": version,
        "latency_ms": latency_ms,
    }

def ping_all():
    prompt = "Who was the first president of the United States?"
    print(f"Ping run at {datetime.utcnow().isoformat()}Z")

    results = {}

    # --- GPT-4o (OpenAI) ---
    try:
        res = query_openai(prompt, temperature=0.0, max_tokens=64)
        # expected to already be a dict with keys: output, error, usage, version, finish_reason
        if isinstance(res, dict):
            results["gpt4o"] = res
        else:
            # fallback normalization
            results["gpt4o"] = _as_structured_dict(output_text=str(res), version="gpt-4o")
    except Exception as e:
        results["gpt4o"] = _as_structured_dict(err=str(e), version="gpt-4o")

    # --- Gemini Flash ---
    try:
        res = query_gemini(prompt, temperature=0.0, max_tokens=64)
        if isinstance(res, dict):
            results["gemini_flash"] = res
        else:
            results["gemini_flash"] = _as_structured_dict(output_text=str(res), version="gemini-1.5-flash")
    except Exception as e:
        results["gemini_flash"] = _as_structured_dict(err=str(e), version="gemini-1.5-flash")

    # --- LLaMA 3.1 8B Instruct (HF) ---
    try:
        res = query_llama(
            prompt,
            temperature=0.0,
            top_p=1.0,
            max_tokens=64,
            model="meta-llama/Llama-3.1-8B-Instruct",
        )
        if isinstance(res, dict):
            results["llama31_8b"] = res
        else:
            # If your llama client ever returns a tuple, normalize it
            if isinstance(res, tuple) and len(res) >= 3:
                text, err, latency_ms = res[0], res[1], res[2]
                results["llama31_8b"] = _as_structured_dict(
                    output_text=text, err=err, version="meta-llama/Llama-3.1-8B-Instruct", latency_ms=latency_ms
                )
            else:
                results["llama31_8b"] = _as_structured_dict(
                    output_text=str(res), version="meta-llama/Llama-3.1-8B-Instruct"
                )
    except Exception as e:
        results["llama31_8b"] = _as_structured_dict(err=str(e), version="meta-llama/Llama-3.1-8B-Instruct")

    # --- Mistral 7B Instruct (HF) ---
    try:
        res = query_mistral(
            prompt,
            temperature=0.0,
            top_p=1.0,
            max_tokens=64,
            model="mistralai/Mistral-7B-Instruct-v0.3",
        )
        if isinstance(res, dict):
            results["mistral7b"] = res
        else:
            if isinstance(res, tuple) and len(res) >= 3:
                text, err, latency_ms = res[0], res[1], res[2]
                results["mistral7b"] = _as_structured_dict(
                    output_text=text, err=err, version="mistralai/Mistral-7B-Instruct-v0.3", latency_ms=latency_ms
                )
            else:
                results["mistral7b"] = _as_structured_dict(
                    output_text=str(res), version="mistralai/Mistral-7B-Instruct-v0.3"
                )
    except Exception as e:
        results["mistral7b"] = _as_structured_dict(err=str(e), version="mistralai/Mistral-7B-Instruct-v0.3")

    # --- GPT-4o Mini (OpenAI) ---
    try:
        res = query_openai_mini(
            prompt,
            temperature=0.0,
            max_tokens=64,
        )
        if isinstance(res, dict):
            results["gpt4o_mini"] = res
        else:
            results["gpt4o_mini"] = _as_structured_dict(output_text=str(res), version="gpt-4o-mini")
    except Exception as e:
        results["gpt4o_mini"] = _as_structured_dict(err=str(e), version="gpt-4o-mini")

    # --- Gemini Pro (Google) ---
    try:
        res = query_gemini_pro(
            prompt,
            temperature=0.0,
            max_tokens=64,
        )
        if isinstance(res, dict):
            results["gemini_pro"] = res
        else:
            results["gemini_pro"] = _as_structured_dict(output_text=str(res), version="gemini-1.5-pro")
    except Exception as e:
        results["gemini_pro"] = _as_structured_dict(err=str(e), version="gemini-1.5-pro")

    # Print results consistently
    for model_id, res in results.items():
        print(f"\n=== {model_id} ===")
        pp.pprint(res)

if __name__ == "__main__":
    ping_all()