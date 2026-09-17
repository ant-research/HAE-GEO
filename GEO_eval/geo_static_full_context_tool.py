#!/usr/bin/env python3
"""One-shot local retrieval returning top-k page bodies in a single tool call."""

from __future__ import annotations

import json
from typing import Any, Dict, List, Tuple

from geo_qwen_tools import _load_corpus, _resolve_corpus, _score


TOOL_NAME = "search_geo_full_content"
STATIC_FULL_CONTEXT_TOOL = {
    "type": "function",
    "function": {
        "name": TOOL_NAME,
        "description": "Search once and return full content for the top-ranked pages.",
        "parameters": {
            "type": "object",
            "properties": {"query": {"type": "string"}},
            "required": ["query"],
            "additionalProperties": False,
        },
    },
}


def _allocate(docs: List[Dict[str, Any]], budget: int) -> Tuple[List[Dict[str, Any]], int, int]:
    remaining = max(0, budget * 2)
    projected = []
    original_total = returned_total = 0
    for index, doc in enumerate(docs):
        original = str(doc.get("content", ""))
        original_total += len(original)
        docs_left = len(docs) - index
        share = remaining // docs_left if docs_left else 0
        content = original[:share]
        remaining -= len(content)
        returned_total += len(content)
        projected.append(
            {
                "rank": index + 1,
                "title": doc.get("title", ""),
                "url": doc.get("url", ""),
                "timestamp": doc.get("timestamp", ""),
                "content": content,
                "source_type": doc.get("source_type", "unknown"),
                "content_truncated": len(content) < len(original),
            }
        )
    return projected, original_total, returned_total


def search_geo_full_content(query: str, *, knowledge_name: str, env: str = "local",
                            top_k: int = 10, dataset: str = "all",
                            query_method: str = "hybrid",
                            max_total_content_tokens: int = 24000,
                            max_retries: int = 1) -> Tuple[str, str]:
    del env, query_method, max_retries
    docs = _load_corpus(str(_resolve_corpus(knowledge_name).resolve()))
    if dataset not in {"", "all"}:
        docs = [doc for doc in docs if str(doc.get("dataset", "all")) in {dataset, "all"}]
    ranked = sorted(docs, key=lambda doc: _score(query, doc), reverse=True)[:top_k]
    budgeted, original_chars, returned_chars = _allocate(ranked, max_total_content_tokens)
    visible_docs = [
        {key: value for key, value in doc.items() if key != "source_type"}
        for doc in budgeted
    ]
    visible = {
        "success": True,
        "query": query,
        "documents": visible_docs,
        "instruction": "Answer the original question using the page-content evidence returned above.",
    }
    raw = {
        "success": True,
        "tool": TOOL_NAME,
        "query": query,
        "knowledge_name": knowledge_name,
        "documents": budgeted,
        "original_content_chars": original_chars,
        "returned_content_chars": returned_chars,
    }
    return json.dumps(visible, ensure_ascii=False), json.dumps(raw, ensure_ascii=False)
