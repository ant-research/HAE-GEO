"""Local reference retrieval for the ``modified`` generation path.

Set ``GEO_REFERENCE_CORPUS`` to public JSON/JSONL pages you are licensed to
process. If unset, generation safely falls back to deterministic templates.
"""

from __future__ import annotations

import json
import os
import re
from functools import lru_cache
from pathlib import Path


@lru_cache(maxsize=4)
def _load(path_text: str):
    path = Path(path_text)
    if path.suffix == ".jsonl":
        return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload.get("documents", []) if isinstance(payload, dict) else payload


def _terms(text: str):
    return set(re.findall(r"[\u4e00-\u9fff]|[A-Za-z0-9_-]+", text.lower()))


def ant_search(query, top_k=10, includeSites="", content_type="webPage"):
    del includeSites
    if content_type != "webPage":
        return []
    path = os.environ.get("GEO_REFERENCE_CORPUS")
    if not path:
        return []
    query_terms = _terms(query)
    docs = _load(str(Path(path).resolve()))
    ranked = sorted(
        docs,
        key=lambda row: len(query_terms & _terms(str(row.get("title", "")) + " " + str(row.get("content", "")))),
        reverse=True,
    )[:top_k]
    return [
        {
            "title": row.get("title", ""),
            "url": row.get("url", ""),
            "snippet": row.get("snippet", str(row.get("content", ""))[:240]),
            "content": row.get("content", ""),
            "timestamp": row.get("timestamp", ""),
        }
        for row in ranked
    ]
