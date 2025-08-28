# src/models/_common.py
"""
Common utility functions for the Multi-Hop-QA model.

This module provides helper functions for:
- Getting the current UTC time in ISO format.
- Cleaning text by removing code fences and whitespace.
- Fetching environment variables with error handling.
- Sleeping with exponential backoff.

Functions:
    now_iso(): Returns the current UTC time in ISO 8601 format.
    clean_text(s: str): Cleans input text by stripping whitespace and removing code fences.
    get_env_or_raise(name: str): Retrieves an environment variable or raises an error if not found.
    backoff_sleep(attempt: int): Sleeps for an exponentially increasing duration based on the attempt number.
"""

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