#!/usr/bin/env python3
"""Public Search/Scrape tools for HEO-Bench.

The paper experiments used an internal retrieval service. Its credentials,
endpoints, storage layout, and proprietary corpus are intentionally not part of
this release. This module preserves the same tool contract using local JSON or
JSONL corpora, making the agent and evaluator reproducible without private
infrastructure.

Each document must contain ``url``, ``title``, and ``content``. Optional fields
include ``timestamp``, ``source_type``, ``category``, and ``level``. Hidden
provenance fields are retained in raw traces but projected out before the tool
result is returned to the evaluated model.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Iterable, List


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs" / "environments.json"
CACHE_ROOT = Path(os.environ.get("GEO_TOOL_CACHE_DIR", ROOT / ".cache" / "tools"))


def _tokens(text: str) -> List[str]:
    return re.findall(r"[\u4e00-\u9fff]|[A-Za-z0-9_-]+", (text or "").lower())


def _load_json_records(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(
            f"corpus not found: {path}. Copy configs/environments.example.json "
            "to configs/environments.json and configure local corpus paths."
        )
    if path.suffix == ".jsonl":
        rows = []
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                row = json.loads(line)
                if isinstance(row, dict):
                    rows.append(row)
        return rows
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, list):
        return [row for row in payload if isinstance(row, dict)]
    if isinstance(payload, dict) and isinstance(payload.get("documents"), list):
        return [row for row in payload["documents"] if isinstance(row, dict)]
    raise ValueError(f"unsupported corpus format: {path}")


@lru_cache(maxsize=1)
def _environment_config() -> Dict[str, Any]:
    path = Path(os.environ.get("GEO_ENVIRONMENTS_CONFIG", DEFAULT_CONFIG))
    if not path.exists():
        path = ROOT / "configs" / "environments.example.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    environments = payload.get("environments", payload)
    if not isinstance(environments, dict):
        raise ValueError("environment configuration must be a JSON object")
    return environments


def _resolve_corpus(knowledge_name: str) -> Path:
    entry = _environment_config().get(knowledge_name)
    if entry is None:
        raise KeyError(
            f"unknown environment {knowledge_name!r}; configure it in "
            "configs/environments.json"
        )
    value = entry.get("corpus") if isinstance(entry, dict) else entry
    path = Path(str(value))
    return path if path.is_absolute() else ROOT / path


@lru_cache(maxsize=16)
def _load_corpus(path_text: str) -> List[Dict[str, Any]]:
    return _load_json_records(Path(path_text))


def _score(query: str, doc: Dict[str, Any]) -> float:
    """Transparent lexical baseline; replace through the adapter for paper-scale retrieval."""
    query_terms = _tokens(query)
    if not query_terms:
        return 0.0
    title = str(doc.get("title", "")).lower()
    body = str(doc.get("content", doc.get("snippet", ""))).lower()
    hits = sum(3.0 if term in title else 1.0 if term in body else 0.0 for term in query_terms)
    return hits / len(query_terms)


def _cache_path(knowledge_name: str, url: str) -> Path:
    namespace = hashlib.sha256(knowledge_name.encode("utf-8")).hexdigest()[:16]
    key = hashlib.sha256(url.encode("utf-8")).hexdigest()
    return CACHE_ROOT / namespace / f"{key}.json"


def _cache_document(knowledge_name: str, doc: Dict[str, Any]) -> None:
    url = str(doc.get("url") or doc.get("ref") or "")
    if not url:
        return
    path = _cache_path(knowledge_name, url)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(doc, ensure_ascii=False), encoding="utf-8")


def _snippet(doc: Dict[str, Any], limit: int = 240) -> str:
    text = str(doc.get("snippet") or doc.get("content") or "")
    text = " ".join(text.split())
    return text if len(text) <= limit else text[:limit].rstrip() + "..."


def search_geo_knowledge_base(
    query: str,
    knowledge_name: str = "clean",
    env: str = "local",
    top_k: int = 10,
    threshold: str = "0.001",
    rankingThreshold: str = "0.1",
    dataset: str = "all",
    query_method: str = "hybrid",
) -> str:
    """Retrieve local documents while preserving the paper's tool response schema."""
    del env, threshold, rankingThreshold, query_method
    docs = _load_corpus(str(_resolve_corpus(knowledge_name).resolve()))
    if dataset not in {"", "all"}:
        docs = [doc for doc in docs if str(doc.get("dataset", "all")) in {dataset, "all"}]
    ranked = sorted(docs, key=lambda doc: _score(query, doc), reverse=True)[: max(1, top_k)]
    for doc in ranked:
        _cache_document(knowledge_name, doc)
    result_docs = []
    for rank, doc in enumerate(ranked, 1):
        result_docs.append(
            {
                "rank": rank,
                "title": doc.get("title", ""),
                "url": doc.get("url") or doc.get("ref") or "",
                "snippet": _snippet(doc),
                "timestamp": doc.get("timestamp", ""),
                "source_type": doc.get("source_type", "unknown"),
                "score": _score(query, doc),
            }
        )
    return json.dumps(
        {
            "success": True,
            "tool": "search_geo_knowledge_base",
            "query": query,
            "knowledge_name": knowledge_name,
            "documents": result_docs,
        },
        ensure_ascii=False,
    )


def scrape_geo_webpage(url: str, max_tokens: int = 4096, knowledge_name: str = "clean") -> str:
    """Return cached full text for a URL previously surfaced by Search."""
    path = _cache_path(knowledge_name, url)
    if not path.exists():
        return json.dumps(
            {
                "success": False,
                "tool": "scrape_geo_webpage",
                "url": url,
                "error": "URL is not in this environment's Search cache; search first.",
            },
            ensure_ascii=False,
        )
    doc = json.loads(path.read_text(encoding="utf-8"))
    content = str(doc.get("content") or "")
    # Portable fallback used only by the public local backend.
    max_chars = max_tokens * 2
    truncated = len(content) > max_chars
    content = content[:max_chars]
    return json.dumps(
        {
            "success": True,
            "tool": "scrape_geo_webpage",
            "title": doc.get("title", ""),
            "url": url,
            "timestamp": doc.get("timestamp", ""),
            "source_type": doc.get("source_type", "unknown"),
            "content": content,
            "truncated": truncated,
        },
        ensure_ascii=False,
    )


GEO_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "search_geo_knowledge_base",
            "description": "Search the active benchmark environment and return ranked page metadata.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                    "knowledge_name": {"type": "string", "default": "clean"},
                    "top_k": {"type": "integer", "default": 10},
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "scrape_geo_webpage",
            "description": "Read full content for a URL returned by Search.",
            "parameters": {
                "type": "object",
                "properties": {
                    "url": {"type": "string"},
                    "max_tokens": {"type": "integer", "default": 4096},
                },
                "required": ["url"],
            },
        },
    },
]


def dispatch_tool(name: str, arguments: Dict[str, Any]) -> str:
    if name == "search_geo_knowledge_base":
        return search_geo_knowledge_base(**arguments)
    if name == "scrape_geo_webpage":
        return scrape_geo_webpage(**arguments)
    return json.dumps({"success": False, "tool": name, "error": "unknown tool"})
