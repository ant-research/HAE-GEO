"""Minimal OpenAI-compatible client used by the public Judge implementation."""

from __future__ import annotations

import os
import time

import requests


MAX_RETRIES = int(os.environ.get("JUDGE_MAX_RETRIES", "3"))
RETRY_STATUS = {429, 500, 502, 503, 504}


def chat(messages, model=None, **kwargs):
    api_url = kwargs.pop("api_url", None) or os.environ.get("JUDGE_API_URL")
    api_key = kwargs.pop("api_key", None) or os.environ.get("JUDGE_API_KEY")
    model = model or os.environ.get("JUDGE_MODEL")
    if not api_url or not api_key or not model:
        raise RuntimeError("Set JUDGE_API_URL, JUDGE_API_KEY, and JUDGE_MODEL")
    payload = {"model": model, "messages": messages, "stream": False, **kwargs}
    headers = {"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"}
    last_error = None
    for attempt in range(MAX_RETRIES):
        try:
            response = requests.post(api_url, headers=headers, json=payload, timeout=300)
            response.raise_for_status()
            return response.json()
        except requests.HTTPError as exc:
            last_error = exc
            status = exc.response.status_code if exc.response is not None else None
            if status not in RETRY_STATUS or attempt == MAX_RETRIES - 1:
                raise
        except (requests.ConnectionError, requests.Timeout) as exc:
            last_error = exc
            if attempt == MAX_RETRIES - 1:
                raise
        time.sleep(2 ** attempt)
    raise last_error or RuntimeError("judge request failed")
