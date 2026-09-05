#!/usr/bin/env python3
"""Run the multi-tool GEO agent over a labelled query set.

Gold labels are written to evaluation artifacts but are never included in the
messages sent to the agent. Each query has an independent output file so an
interrupted run can resume safely.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set

from agent_infer_multi_tool import (
    DEFAULT_ENABLE_THINKING,
    DEFAULT_MAX_RETRIES,
    DEFAULT_MODEL,
    SYSTEM_PROMPTS,
    run_react,
)


ROOT = Path(__file__).resolve().parent
DEFAULT_QUERIES = ROOT / "data" / "queries" / "queries_v3.json"
DEFAULT_OUTPUT_DIR = ROOT / "data" / "eval_runs"


def parse_csv(value: Optional[str]) -> Set[str]:
    return {item.strip() for item in (value or "").split(",") if item.strip()}


def safe_file_name(value: str) -> str:
    value = re.sub(r"[^0-9A-Za-z._\-\u4e00-\u9fff]+", "_", value.strip())
    return value or "query"


def load_queries(path: Path) -> List[Dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError(f"query file must contain a JSON list: {path}")
    required = ("query_id", "user_query")
    for index, item in enumerate(data):
        if not isinstance(item, dict) or any(not item.get(key) for key in required):
            raise ValueError(f"invalid query item at index {index}: require {required}")
    return data


def select_queries(
    items: Iterable[Dict[str, Any]],
    *,
    categories: Set[str],
    task_types: Set[str],
    query_ids: Set[str],
    attack_scope: str,
    offset: int,
    limit: Optional[int],
) -> List[Dict[str, Any]]:
    selected = [
        item for item in items
        if (not categories or item.get("category") in categories)
        and (not task_types or item.get("task_type") in task_types)
        and (not query_ids or item.get("query_id") in query_ids)
        and (attack_scope == "all" or bool(item.get("fake_brands")))
    ]
    return selected[offset:] if limit is None else selected[offset:offset + limit]


def is_complete(path: Path) -> bool:
    if not path.exists():
        return False
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return False
    return isinstance(data, dict) and data.get("status") == "complete" and bool(data.get("final_answer"))


def write_json(path: Path, data: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_suffix(path.suffix + ".tmp")
    temp_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    temp_path.replace(path)


def infer_attack_type(item: Dict[str, Any]) -> str:
    if item.get("attack_type"):
        return str(item["attack_type"])
    if item.get("fake_brands") and (item.get("real_brands") or item.get("real_brands_in_query")):
        return "fake_brand_boost"
    if item.get("fake_brands"):
        return "fake_brand_boost"
    return "clean_control"


def main() -> int:
    parser = argparse.ArgumentParser(description="Batch-run evaluation queries through the GEO multi-tool agent.")
    parser.add_argument("--queries", type=Path, default=DEFAULT_QUERIES)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--run-name", default="baseline")
    parser.add_argument("--prompt-mode", choices=sorted(SYSTEM_PROMPTS), default="baseline")
    parser.add_argument("--categories", help="Comma-separated category filter")
    parser.add_argument("--task-types", help="Comma-separated task_type filter")
    parser.add_argument("--query-ids", help="Comma-separated query_id filter")
    parser.add_argument(
        "--attack-scope",
        choices=("a", "all"),
        default="a",
        help="a: only queries labelled with fake_brands; all: include non-attack/control queries",
    )
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--max-turns", type=int, default=10)
    parser.add_argument("--max-tokens", type=int, default=4096)
    parser.add_argument("--temperature", type=float)
    parser.add_argument("--top-p", type=float)
    parser.add_argument("--model-top-k", "--top-k", dest="model_top_k", type=int, default=20)
    parser.add_argument("--search-top-k", type=int, default=10)
    parser.add_argument("--search-dataset", choices=("all", "clean", "toudu"), default="all")
    parser.add_argument("--search-query-method", choices=("text", "vector", "hybrid"), default="hybrid")
    parser.add_argument("--timeout", type=int, default=300)
    parser.add_argument("--max-retries", type=int, default=DEFAULT_MAX_RETRIES)
    parser.add_argument("--retry-sleep", type=float, default=1.0)
    parser.add_argument("--enable-thinking", dest="enable_thinking", action="store_true", default=DEFAULT_ENABLE_THINKING)
    parser.add_argument("--no-enable-thinking", dest="enable_thinking", action="store_false")
    parser.add_argument("--rerun-complete", action="store_true")
    args = parser.parse_args()

    temperature = args.temperature if args.temperature is not None else (0.6 if args.enable_thinking else 0.7)
    top_p = args.top_p if args.top_p is not None else (0.95 if args.enable_thinking else 0.8)
    items = select_queries(
        load_queries(args.queries),
        categories=parse_csv(args.categories),
        task_types=parse_csv(args.task_types),
        query_ids=parse_csv(args.query_ids),
        attack_scope=args.attack_scope,
        offset=max(0, args.offset),
        limit=args.limit,
    )
    run_dir = args.output_dir / safe_file_name(args.run_name)
    run_dir.mkdir(parents=True, exist_ok=True)
    query_sha256 = hashlib.sha256(args.queries.read_bytes()).hexdigest()
    print(
        f"[batch] selected={len(items)} attack_scope={args.attack_scope} output={run_dir} "
        f"prompt_mode={args.prompt_mode} retrieval={args.search_dataset}/"
        f"{args.search_query_method}/top{args.search_top_k}",
        flush=True,
    )

    completed = failed = skipped = 0
    for index, item in enumerate(items, start=1):
        query_id = str(item["query_id"])
        output_path = run_dir / f"{safe_file_name(query_id)}.json"
        partial_path = run_dir / f".{safe_file_name(query_id)}.partial.json"
        if not args.rerun_complete and is_complete(output_path):
            skipped += 1
            print(f"[{index}/{len(items)}] SKIP complete {query_id}", flush=True)
            continue

        print(f"[{index}/{len(items)}] RUN {query_id}: {item['user_query']}", flush=True)
        started_at = datetime.now(timezone.utc).isoformat()
        try:
            result = run_react(
                str(item["user_query"]),
                model=args.model,
                max_turns=args.max_turns,
                temperature=temperature,
                top_p=top_p,
                top_k=args.model_top_k,
                max_tokens=args.max_tokens,
                timeout=args.timeout,
                trace_path=str(partial_path),
                preserve_reasoning=True,
                enable_thinking=args.enable_thinking,
                max_retries=args.max_retries,
                retry_sleep=args.retry_sleep,
                prompt_mode=args.prompt_mode,
                search_top_k=args.search_top_k,
                search_dataset=args.search_dataset,
                search_query_method=args.search_query_method,
            )
            final_answer = result.get("final", "")
            status = "complete" if final_answer and not result.get("error") else "failed"
            artifact = {
                **item,
                "attack_type": infer_attack_type(item),
                "status": status,
                "prompt_mode": args.prompt_mode,
                "model": args.model,
                "benchmark": {
                    "query_file": str(args.queries),
                    "query_file_sha256": query_sha256,
                    "attack_scope": args.attack_scope,
                },
                "generation_config": {
                    "enable_thinking": args.enable_thinking,
                    "temperature": temperature,
                    "top_p": top_p,
                    "model_top_k": args.model_top_k,
                    "max_tokens": args.max_tokens,
                    "max_turns": args.max_turns,
                    "max_tool_turns": args.max_turns,
                    "force_final_answer_after_max_turns": True,
                },
                "retrieval_config": {
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
            if partial_path.exists():
                partial_path.unlink()
            if status == "complete":
                completed += 1
            else:
                failed += 1
        except Exception as exc:  # noqa: BLE001
            failed += 1
            artifact = {
                **item,
                "attack_type": infer_attack_type(item),
                "status": "failed",
                "prompt_mode": args.prompt_mode,
                "model": args.model,
                "benchmark": {
                    "query_file": str(args.queries),
                    "query_file_sha256": query_sha256,
                    "attack_scope": args.attack_scope,
                },
                "retrieval_config": {
                    "dataset": args.search_dataset,
                    "query_method": args.search_query_method,
                    "top_k": args.search_top_k,
                },
                "started_at": started_at,
                "finished_at": datetime.now(timezone.utc).isoformat(),
                "trajectory": [],
                "final_answer": "",
                "error": f"{type(exc).__name__}: {exc}",
            }
            if partial_path.exists():
                try:
                    partial = json.loads(partial_path.read_text(encoding="utf-8"))
                    if isinstance(partial, list):
                        artifact["trajectory"] = partial
                except Exception:
                    pass
            write_json(output_path, artifact)
            print(f"  FAILED {artifact['error']}", flush=True)

    print(f"[batch] complete={completed} failed={failed} skipped={skipped}", flush=True)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
