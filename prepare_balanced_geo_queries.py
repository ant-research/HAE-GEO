#!/usr/bin/env python3
"""Create deterministic full/interleaved or strictly balanced GEO query sets."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from collections import Counter, defaultdict
from pathlib import Path


def read_queries(path: Path):
    rows = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(rows, list):
        raise ValueError(f"query file must be a JSON list: {path}")
    seen = set()
    for index, row in enumerate(rows):
        if not isinstance(row, dict) or not row.get("query_id") or not row.get("category"):
            raise ValueError(f"row {index} requires query_id and category")
        query_id = str(row["query_id"])
        if query_id in seen:
            raise ValueError(f"duplicate query_id: {query_id}")
        seen.add(query_id)
    return rows


def interleave(groups, category_order):
    output = []
    index = 0
    while True:
        added = False
        for category in category_order:
            rows = groups[category]
            if index < len(rows):
                output.append(rows[index])
                added = True
        if not added:
            return output
        index += 1


def select_queries(rows, *, mode, seed, per_category=None):
    groups = defaultdict(list)
    for row in rows:
        groups[str(row["category"])].append(row)
    categories = sorted(groups)
    rng = random.Random(seed)
    for category in categories:
        rng.shuffle(groups[category])

    if mode == "balanced":
        target = per_category if per_category is not None else min(
            len(groups[category]) for category in categories
        )
        if target < 1:
            raise ValueError("per-category target must be >= 1")
        too_small = {
            category: len(groups[category])
            for category in categories
            if len(groups[category]) < target
        }
        if too_small:
            raise ValueError(f"categories smaller than --per-category={target}: {too_small}")
        groups = {
            category: groups[category][:target]
            for category in categories
        }
    elif per_category is not None:
        raise ValueError("--per-category is only valid with --mode balanced")

    return interleave(groups, categories)


def sha256(path: Path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--queries", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--mode", choices=("full", "balanced"), default="full")
    parser.add_argument("--seed", type=int, default=20260810)
    parser.add_argument("--per-category", type=int)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    source_hash = sha256(args.queries)
    requested = {
        "source_sha256": source_hash,
        "mode": args.mode,
        "seed": args.seed,
        "per_category": args.per_category,
    }
    if args.output.exists() and args.manifest.exists() and not args.force:
        manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
        if all(manifest.get(key) == value for key, value in requested.items()):
            print(
                f"[query sample] reuse rows={manifest.get('output_count')} "
                f"path={args.output}"
            )
            return 0
        raise RuntimeError(
            "existing output was built with different settings; use --force or a new path"
        )

    rows = read_queries(args.queries)
    selected = select_queries(
        rows,
        mode=args.mode,
        seed=args.seed,
        per_category=args.per_category,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(selected, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    source_counts = Counter(str(row["category"]) for row in rows)
    output_counts = Counter(str(row["category"]) for row in selected)
    manifest = {
        **requested,
        "source": str(args.queries.resolve()),
        "output": str(args.output.resolve()),
        "source_count": len(rows),
        "output_count": len(selected),
        "source_category_counts": dict(sorted(source_counts.items())),
        "output_category_counts": dict(sorted(output_counts.items())),
        "output_sha256": sha256(args.output),
        "ordering": "seeded-within-category-round-robin",
    }
    args.manifest.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
