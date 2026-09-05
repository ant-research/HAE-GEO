"""Concurrent GEO evaluation against a local OpenAI-compatible vLLM server."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict

import requests

import agent_infer_multi_tool as runtime
from run_geo_eval import (
    DEFAULT_OUTPUT_DIR,
    DEFAULT_QUERIES,
    infer_attack_type,
    is_complete,
    load_queries,
    parse_csv,
    safe_file_name,
    select_queries,
    write_json,
)


DEFAULT_CHAT_URL = os.environ.get(
    "MODEL_API_URL",
    "http://127.0.0.1:36901/v1/chat/completions",
)
DEFAULT_MODEL = os.environ.get("MODEL_NAME", "your-tool-capable-model")


def models_url(chat_url: str) -> str:
    suffix = "/chat/completions"
    return chat_url[: -len(suffix)] + "/models" if chat_url.endswith(suffix) else chat_url.rstrip("/") + "/models"


def check_server(chat_url: str, api_key: str) -> None:
    response = requests.get(
        models_url(chat_url),
        headers={"Authorization": f"Bearer {api_key}"},
        timeout=30,
    )
    response.raise_for_status()
    models = [item.get("id") for item in response.json().get("data", [])]
    print(f"[server] models={models}", flush=True)


def run_one(
    item: Dict[str, Any],
    *,
    run_dir: Path,
    query_file: Path,
    query_sha256: str,
    args: argparse.Namespace,
) -> Dict[str, Any]:
    query_id = str(item["query_id"])
    stem = safe_file_name(query_id)
    output_path = run_dir / f"{stem}.json"
    partial_path = run_dir / f".{stem}.partial.json"
    started_at = datetime.now(timezone.utc).isoformat()

    try:
        result = runtime.run_react(
            str(item["user_query"]),
            model=args.model,
            max_turns=args.max_turns,
            temperature=args.temperature,
            top_p=args.top_p,
            top_k=args.model_top_k,
            max_tokens=args.max_tokens,
            timeout=args.timeout,
            trace_path=str(partial_path),
            preserve_reasoning=True,
            enable_thinking=args.enable_thinking,
            max_retries=args.max_retries,
            retry_sleep=args.retry_sleep,
            prompt_mode=args.prompt_mode,
            search_knowledge_name=args.search_knowledge_name,
            search_top_k=args.search_top_k,
            search_dataset=args.search_dataset,
            search_query_method=args.search_query_method,
        )
        final_answer = result.get("final", "")
        status = "complete" if final_answer and not result.get("error") else "failed"
        artifact: Dict[str, Any] = {
            **item,
            "attack_type": infer_attack_type(item),
            "status": status,
            "prompt_mode": args.prompt_mode,
            "model": args.model,
            "api_url": args.chat_url,
            "benchmark": {
                "query_file": str(query_file),
                "query_file_sha256": query_sha256,
                "attack_scope": args.attack_scope,
            },
            "generation_config": {
                "enable_thinking": args.enable_thinking,
                "temperature": args.temperature,
                "top_p": args.top_p,
                "model_top_k": args.model_top_k,
                "max_tokens": args.max_tokens,
                "max_turns": args.max_turns,
                "max_tool_turns": args.max_turns,
                "force_final_answer_after_max_turns": True,
            },
            "retrieval_config": {
                "knowledge_name": args.search_knowledge_name,
                "dataset": args.search_dataset,
                "query_method": args.search_query_method,
                "top_k": args.search_top_k,
            },
            "started_at": started_at,
            "finished_at": datetime.now(timezone.utc).isoformat(),
            "trajectory": result.get("trace", []),
            "final_answer": final_answer,
            "forced_finalization": bool(result.get("forced_finalization")),
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
                partial = json.loads(partial_path.read_text(encoding="utf-8"))
                if isinstance(partial, list):
                    trajectory = partial
            except Exception:
                pass
        error = f"{type(exc).__name__}: {exc}"
        write_json(
            output_path,
            {
                **item,
                "attack_type": infer_attack_type(item),
                "status": "failed",
                "prompt_mode": args.prompt_mode,
                "model": args.model,
                "api_url": args.chat_url,
                "benchmark": {
                    "query_file": str(query_file),
                    "query_file_sha256": query_sha256,
                    "attack_scope": args.attack_scope,
                },
                "retrieval_config": {
                    "knowledge_name": args.search_knowledge_name,
                    "dataset": args.search_dataset,
                    "query_method": args.search_query_method,
                    "top_k": args.search_top_k,
                },
                "started_at": started_at,
                "finished_at": datetime.now(timezone.utc).isoformat(),
                "trajectory": trajectory,
                "final_answer": "",
                "error": error,
            },
        )
        return {"query_id": query_id, "status": "failed", "error": error}


def main() -> int:
    parser = argparse.ArgumentParser(description="Concurrent local-vLLM GEO evaluation.")
    parser.add_argument("--queries", type=Path, default=DEFAULT_QUERIES)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--run-name", default="local_qwen3_6_35b_a3b_test")
    parser.add_argument(
        "--summary-output",
        type=Path,
        help="Optional summary path; defaults to <output-dir>/<run-name>_summary.json.",
    )
    parser.add_argument("--chat-url", default=DEFAULT_CHAT_URL)
    parser.add_argument("--api-key", default=os.environ.get("MODEL_API_KEY", "EMPTY"))
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--limit", type=int, default=20, help="Default 20 for a safe test; use --all for every selected query.")
    parser.add_argument("--all", action="store_true", help="Run all selected queries; overrides --limit.")
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--categories")
    parser.add_argument("--task-types")
    parser.add_argument("--query-ids")
    parser.add_argument("--attack-scope", choices=("a", "all"), default="all")
    parser.add_argument("--prompt-mode", choices=sorted(runtime.SYSTEM_PROMPTS), default="baseline")
    parser.add_argument("--max-turns", type=int, default=10)
    parser.add_argument("--max-tokens", type=int, default=4096)
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--model-top-k", type=int, default=20)
    parser.add_argument("--search-knowledge-name", default="clean")
    parser.add_argument("--search-top-k", type=int, default=10)
    parser.add_argument("--search-dataset", choices=("all", "clean", "toudu"), default="all")
    parser.add_argument("--search-query-method", choices=("text", "vector", "hybrid"), default="hybrid")
    parser.add_argument("--timeout", type=int, default=1200)
    parser.add_argument("--max-retries", type=int, default=4)
    parser.add_argument("--retry-sleep", type=float, default=1.0)
    parser.add_argument("--enable-thinking", dest="enable_thinking", action="store_true", default=True)
    parser.add_argument("--no-enable-thinking", dest="enable_thinking", action="store_false")
    parser.add_argument("--rerun-complete", action="store_true")
    parser.add_argument(
        "--skip-server-check",
        action="store_true",
        help="Skip GET /v1/models, for remote OpenAI-compatible endpoints.",
    )
    args = parser.parse_args()

    if args.workers < 1:
        parser.error("--workers must be >= 1")
    if not args.skip_server_check:
        check_server(args.chat_url, args.api_key)
    else:
        print(f"[server] skip models check for {args.chat_url}", flush=True)
    runtime.API_URL = args.chat_url
    runtime.API_KEY = args.api_key

    limit = None if args.all else args.limit
    items = select_queries(
        load_queries(args.queries),
        categories=parse_csv(args.categories),
        task_types=parse_csv(args.task_types),
        query_ids=parse_csv(args.query_ids),
        attack_scope=args.attack_scope,
        offset=max(0, args.offset),
        limit=limit,
    )
    run_dir = args.output_dir / safe_file_name(args.run_name)
    run_dir.mkdir(parents=True, exist_ok=True)
    query_sha256 = hashlib.sha256(args.queries.read_bytes()).hexdigest()

    pending = []
    skipped = 0
    for item in items:
        output_path = run_dir / f"{safe_file_name(str(item['query_id']))}.json"
        if not args.rerun_complete and is_complete(output_path):
            skipped += 1
        else:
            pending.append(item)

    print(
        f"[batch] selected={len(items)} pending={len(pending)} skipped={skipped} "
        f"workers={args.workers} model={args.model} "
        f"knowledge_name={args.search_knowledge_name} url={args.chat_url}",
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
        for index, future in enumerate(as_completed(futures), start=1):
            item = futures[future]
            try:
                result = future.result()
            except Exception as exc:  # defensive; run_one already catches
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
        "selected": len(items),
        "submitted": len(pending),
        "skipped_complete": skipped,
        "completed": sum(r["status"] == "complete" for r in results),
        "failed": sum(r["status"] != "complete" for r in results),
        "workers": args.workers,
        "model": args.model,
        "chat_url": args.chat_url,
        "run_dir": str(run_dir),
    }
    summary_output = args.summary_output or (
        args.output_dir / f"{safe_file_name(args.run_name)}_summary.json"
    )
    write_json(summary_output, summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    return 1 if summary["failed"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
