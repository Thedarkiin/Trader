"""Gemini REST client with retry, JSON extraction, and schema-light validation."""
import json
import os
import re
import time

import requests
from dotenv import load_dotenv

load_dotenv(os.path.join(os.path.dirname(__file__), ".env"))

API_KEY = os.environ["GEMINI_API_KEY"]
BASE = "https://generativelanguage.googleapis.com/v1beta/models"

MODEL_CHEAP = "gemini-flash-lite-latest"
MODEL_JUDGE = "gemini-flash-latest"  # stronger tier for the gate, temp 0


class AgentError(Exception):
    pass


def _extract_json(text: str) -> dict:
    # strip markdown fences if present
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if not m:
        raise AgentError(f"no JSON object in response: {text[:200]}")
    return json.loads(m.group(0))


def call_agent(system_prompt: str, user_payload: dict, model: str = MODEL_CHEAP,
               temperature: float = 0.2, required_keys: tuple = ()) -> dict:
    """One agent call. 1 retry on 429/5xx/bad JSON, then raises AgentError."""
    url = f"{BASE}/{model}:generateContent?key={API_KEY}"
    body = {
        "system_instruction": {"parts": [{"text": system_prompt}]},
        "contents": [{"role": "user",
                      "parts": [{"text": json.dumps(user_payload, default=str)}]}],
        "generationConfig": {"temperature": temperature,
                             "responseMimeType": "application/json"},
    }
    last_err = None
    for attempt in range(2):
        try:
            r = requests.post(url, json=body, timeout=60)
            if r.status_code == 429 or r.status_code >= 500:
                raise AgentError(f"HTTP {r.status_code}: {r.text[:200]}")
            r.raise_for_status()
            text = r.json()["candidates"][0]["content"]["parts"][0]["text"]
            out = _extract_json(text)
            missing = [k for k in required_keys if k not in out]
            if missing:
                raise AgentError(f"missing keys {missing} in agent output")
            return out
        except (AgentError, requests.RequestException, KeyError, json.JSONDecodeError) as e:
            last_err = e
            if attempt == 0:
                time.sleep(5)  # backoff before the single retry
    raise AgentError(f"agent failed after retry: {last_err}")
