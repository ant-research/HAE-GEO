#!/usr/bin/env python3
"""Batch runner for one-shot full-content GEO retrieval."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict

import agent_infer_multi_tool as chat_runtime
import agent_infer_static_full_context as runtime
from run_geo_eval import (
    DEFAULT_QUERIES,
    infer_attack_type,
    is_complete,
    load_queries,
    parse_csv,
    safe_file_name,
    select_queries,
    write_json,
)


def run_one(
    item: Dict[str, Any],
    *,
    run_dir: Path,
    query_file: Path,
    query_sha256: str,
    args: argparse.Namespace,
) -> Dict[str, Any]:
    query_id = str(item["query_id"])
    output_path = run_dir / f"{safe_file_name(query_id)}.json"
    partial_path = run_dir / f".{safe_file_name(query_id)}.partial.json"
    started_at = datetime.now(timezone.utc).isoformat()
    try:
        result = runtime.run_static_full_context(
            str(item["user_query"]),
            model=args.model,
            prompt_mode=args.prompt_mode,
            knowledge_name=args.search_knowledge_name,
            search_top_k=args.search_top_k,
            search_dataset=args.search_dataset,
            search_query_method=args.search_query_method,
            max_total_content_tokens=args.content_budget_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            top_k=args.model_top_k,
            max_tokens=args.max_tokens,
            timeout=args.timeout,
            max_retries=args.max_retries,
            retry_sleep=args.retry_sleep,
            enable_thinking=True,
            preserve_reasoning=True,
            trace_path=str(partial_path),
        )
        final_answer = result.get("final", "")
        status = "complete" if final_answer and not result.get("error") else "failed"
        artifact: Dict[str, Any] = {
            **item,
            "attack_type": infer_attack_type(item),
            "status": status,
            "interaction_mode": "static_full_context",
            "prompt_mode": args.prompt_mode,
            "model": args.model,
            "api_url": args.chat_url,
            "benchmark": {
                "query_file": str(query_file),
                "query_file_sha256": query_sha256,
                "attack_scope": args.attack_scope,
            },
            "generation_config": {
                "enable_thinking": True,
                "temperature": args.temperature,
                "top_p": args.top_p,
                "model_top_k": args.model_top_k,
                "max_tokens": args.max_tokens,
                "model_calls": 2,
                "max_tool_calls": 1,
                "forced_final_answer_after_search": True,
            },
            "retrieval_config": {
                "knowledge_name": args.search_knowledge_name,
                "dataset": args.search_dataset,
                "query_method": args.search_query_method,
                "top_k": args.search_top_k,
                "content_budget_tokens": args.content_budget_tokens,
                "returns_full_content_directly": True,
                "writes_oss_cache": False,
            },
            "started_at": started_at,
            "finished_at": datetime.now(timezone.utc).isoformat(),
            "trajectory": result.get("trace", []),
            "final_answer": final_answer,
            "forced_finalization": True,
        }
        if result.get("error"):
            artifact["error"] = result["error"]
        write_json(output_path, artifact)
        partial_path.unlink(missing_ok=True)
        return {"query_id": query_id, "status": status, "error": artifact.get("error", "")}
    except Exception as exc:  # noqa: BLE001
        trajectory = []
        if partial_path.exists():
            try:
                candidate = json.loads(partial_path.read_text(encoding="utf-8"))
                if isinstance(candidate, list):
                    trajectory = candidate
            except Exception:
                pass
        error = f"{type(exc).__name__}: {exc}"
        write_json(
            output_path,
            {
                **item,
                "attack_type": infer_attack_type(item),
                "status": "failed",
                "interaction_mode": "static_full_context",
                "prompt_mode": args.prompt_mode,
                "model": args.model,
                "api_url": args.chat_url,
                "started_at": started_at,
                "finished_at": datetime.now(timezone.utc).isoformat(),
                "trajectory": trajectory,
                "final_answer": "",
                "error": error,
            },
        )
        return {"query_id": query_id, "status": "failed", "error": error}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--queries", type=Path, default=DEFAULT_QUERIES)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--summary-output", type=Path)
    parser.add_argument("--chat-url", default=os.environ.get("MODEL_API_URL", ""))
    parser.add_argument("--api-key", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--limit", type=int, default=20)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--categories")
    parser.add_argument("--task-types")
    parser.add_argument("--query-ids")
    parser.add_argument("--attack-scope", choices=("a", "all"), default="all")
    parser.add_argument("--prompt-mode", choices=sorted(runtime.SYSTEM_PROMPTS), default="baseline")
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--model-top-k", type=int, default=20)
    parser.add_argument("--max-tokens", type=int, default=4096)
    parser.add_argument("--search-knowledge-name", required=True)
    parser.add_argument("--search-top-k", type=int, default=10)
    parser.add_argument("--search-dataset", choices=("all", "clean", "toudu"), default="all")
    parser.add_argument("--search-query-method", choices=("text", "vector", "hybrid"), default="hybrid")
    parser.add_argument("--content-budget-tokens", type=int, default=24000)
    parser.add_argument("--timeout", type=int, default=1200)
    parser.add_argument("--max-retries", type=int, default=10)
    parser.add_argument("--retry-sleep", type=float, default=2.0)
    parser.add_argument("--rerun-complete", action="store_true")
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("--workers must be >= 1")

    chat_runtime.API_URL = args.chat_url
    chat_runtime.API_KEY = args.api_key
    selected = select_queries(
        load_queries(args.queries),
        categories=parse_csv(args.categories),
        task_types=parse_csv(args.task_types),
        query_ids=parse_csv(args.query_ids),
        attack_scope=args.attack_scope,
        offset=max(0, args.offset),
        limit=None if args.all else args.limit,
    )
    run_dir = args.output_dir / safe_file_name(args.run_name)
    run_dir.mkdir(parents=True, exist_ok=True)
    query_sha256 = hashlib.sha256(args.queries.read_bytes()).hexdigest()
    pending = []
    skipped = 0
    for item in selected:
        path = run_dir / f"{safe_file_name(str(item['query_id']))}.json"
        if not args.rerun_complete and is_complete(path):
            skipped += 1
        else:
            pending.append(item)
    print(
        f"[static batch] selected={len(selected)} pending={len(pending)} skipped={skipped} "
        f"workers={args.workers} model={args.model} prompt={args.prompt_mode} "
        f"kb={args.search_knowledge_name} top_k={args.search_top_k} "
        f"content_budget={args.content_budget_tokens}",
        flush=True,
    )
    results = []
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(
                run_one,
                item,
                run_dir=run_dir,
                query_file=args.queries,
                query_sha256=query_sha256,
                args=args,
            ): item
            for item in pending
        }
        for index, future in enumerate(as_completed(futures), 1):
            item = futures[future]
            try:
                result = future.result()
            except Exception as exc:  # defensive
                result = {
                    "query_id": item.get("query_id"),
                    "status": "failed",
                    "error": f"{type(exc).__name__}: {exc}",
                }
            results.append(result)
            print(
                f"[{index}/{len(futures)}] {result['status'].upper()} "
                f"{result['query_id']} {result.get('error', '')}",
                flush=True,
            )
    summary = {
        "selected": len(selected),
        "submitted": len(pending),
        "skipped_complete": skipped,
        "completed": sum(row["status"] == "complete" for row in results),
        "failed": sum(row["status"] != "complete" for row in results),
        "interaction_mode": "static_full_context",
        "model": args.model,
        "prompt_mode": args.prompt_mode,
        "run_dir": str(run_dir),
    }
    summary_path = args.summary_output or args.output_dir / f"{safe_file_name(args.run_name)}_summary.json"
    write_json(summary_path, summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    return 1 if summary["failed"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
