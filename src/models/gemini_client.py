import os, time, requests
from dotenv import load_dotenv

load_dotenv()
GEMINI_KEY = os.getenv("GEMINI_API_KEY")


URL = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent"
# URL = "https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-pro-latest:generateContent"
HEADERS = {"Content-Type": "application/json", "X-goog-api-key": GEMINI_KEY}


def query_gemini(prompt: str, max_tokens=150, temperature=0.7, top_p=0.9):
    t0 = time.time()
    payload = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {
            "temperature": temperature,
            "topP": top_p,
            "maxOutputTokens": max_tokens,
        },
    }
    try:
        r = requests.post(URL, headers=HEADERS, json=payload, timeout=60)
        r.raise_for_status()
        data = r.json()
        text = (
            data.get("candidates", [{}])[0]
            .get("content", {})
            .get("parts", [{}])[0]
            .get("text", "")
        ).strip()
        err = ""
    except Exception as e:
        text, err = "", f"{e}"
    latency_ms = int((time.time() - t0) * 1000)
    return text, err, latency_ms
