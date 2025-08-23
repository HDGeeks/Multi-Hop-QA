import os, pprint
from datetime import datetime

from models.openai_client import query_openai
from models.gemini_client import query_gemini
from models.llama_client import query_llama
from models.mistral_client import query_mistral

pp = pprint.PrettyPrinter(indent=2, width=100)

def ping_all():
    prompt = "Who was the first president of the United States?"

    results = {}

    # OpenAI GPT-4o
    results["gpt4o"] = query_openai(
        prompt,
        temperature=0.0,
        max_tokens=64,
    )

    # Gemini Flash
    results["gemini_flash"] = query_gemini(
        prompt,
        temperature=0.0,
        max_tokens=64,
    )

    # LLaMA 3.1
    results["llama31_8b"] = query_llama(
        prompt,
        temperature=0.0,
        top_p=1.0,
        max_tokens=64,
        model="meta-llama/Llama-3.1-8B-Instruct",
    )

    # Mistral 7B
    results["mistral7b"] = query_mistral(
        prompt,
        temperature=0.0,
        top_p=1.0,
        max_tokens=64,
        model="mistralai/Mistral-7B-Instruct-v0.3",
    )

    # Print results in a consistent way
    print(f"Ping run at {datetime.utcnow().isoformat()}Z")
    for model, res in results.items():
        print(f"\n=== {model} ===")
        pp.pprint(res)

if __name__ == "__main__":
    ping_all()