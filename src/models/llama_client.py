import os, time
from dotenv import load_dotenv
from huggingface_hub import InferenceClient

load_dotenv()
HF_TOKEN = os.getenv("HUGGINGFACE_API_KEY")
if not HF_TOKEN:
    raise ValueError("HUGGINGFACE_API_KEY not set in .env")

# HF_MODEL = "meta-llama/Llama-3.1-8B"
# HF_MODEL_INSTRUCT = "meta-llama/Llama-3.1-8B-Instruct"  # or your preferred LLaMA 3.x
# HF_MODEL = "mistralai/Mistral-7B-Instruct-v0.3"

client = InferenceClient(api_key=HF_TOKEN)


def query_llama(prompt: str, max_tokens, temperature, top_p, model):
    t0 = time.time()
    try:
        result = client.chat_completion(
            messages=[{"role": "user", "content": prompt}],
            model=model,
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
            stream=False,
        )
        text = result.choices[0].message.content.strip()  # ✅ this fixes the error
        err = ""
    except Exception as e:
        text, err = "", f"{e}"
    latency_ms = int((time.time() - t0) * 1000)
    return text, err, latency_ms


# def query_llama(prompt: str, max_tokens=150, temperature=0.7, top_p=0.9):
#     t0 = time.time()
#     try:
#         text = client.text_generation(  # we use chat_completion for LLaMA 3.x but text_generation worked for smaller models
#             prompt,
#             model=HF_MODEL_INSTRUCT,
#             max_new_tokens=max_tokens,
#             temperature=temperature,
#             top_p=top_p,
#             stream=False,
#         ).strip()
#         err = ""
#     except Exception as e:
#         text, err = "", f"{e}"
#     latency_ms = int((time.time() - t0) * 1000)
#     return text, err, latency_ms
