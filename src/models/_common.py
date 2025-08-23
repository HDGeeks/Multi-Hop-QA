# src/models/_common.py
import os, time, json, re
from datetime import datetime

def now_iso():
    return datetime.utcnow().isoformat()

def clean_text(s: str) -> str:
    if not isinstance(s, str):
        return ""
    s = s.strip()
    # remove surrounding code fences if present
    s = re.sub(r"^\s*```(?:json|txt)?\s*", "", s)
    s = re.sub(r"\s*```\s*$", "", s)
    return s.strip()

def get_env_or_raise(name: str) -> str:
    val = os.getenv(name, "").strip()
    if not val:
        raise RuntimeError(f"{name} not found in environment")
    return val

def backoff_sleep(attempt: int):
    time.sleep(min(2 ** attempt, 8))