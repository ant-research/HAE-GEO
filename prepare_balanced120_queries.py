#!/usr/bin/env python3
"""Build a deterministic 120-query subset from the existing balanced240 set."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from collections import Counter, defaultdict
from pathlib import Path


TASKS = ("compare_and_recommend", "verification")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def counts(rows, key):
    return dict(sorted(Counter(str(row.get(key, "")) for row in rows).items()))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260821)
    args = parser.parse_args()

    source_rows = json.loads(args.source.read_text(encoding="utf-8"))
    eligible = [row for row in source_rows if row.get("task_type") in TASKS]
    categories = sorted({str(row["category"]) for row in eligible})
    if len(categories) != 8:
        raise ValueError(f"expected 8 categories, found {len(categories)}: {categories}")

    rng = random.Random(args.seed)
    quota_order = categories[:]
    rng.shuffle(quota_order)
    compare_heavy = set(quota_order[:4])

    strata = defaultdict(list)
    for row in eligible:
        strata[(str(row["category"]), str(row["task_type"]))].append(row)

    selected = []
    quotas = {}
    for category in categories:
        compare_n = 8 if category in compare_heavy else 7
        verify_n = 15 - compare_n
        quotas[category] = {
            "compare_and_recommend": compare_n,
            "verification": verify_n,
        }
        for task, target in quotas[category].items():
            pool = sorted(strata[(category, task)], key=lambda row: str(row["query_id"]))
            if len(pool) < target:
                raise ValueError(
                    f"insufficient rows for category={category} task={task}: "
                    f"need={target} found={len(pool)}"
                )
            selected.extend(rng.sample(pool, target))

    rng.shuffle(selected)
    query_ids = [str(row["query_id"]) for row in selected]
    if len(selected) != 120 or len(set(query_ids)) != 120:
        raise AssertionError("selection must contain 120 unique queries")
    if Counter(row["task_type"] for row in selected) != Counter(
        {"compare_and_recommend": 60, "verification": 60}
    ):
        raise AssertionError("task split must be exactly 60/60")
    if set(Counter(row["category"] for row in selected).values()) != {15}:
        raise AssertionError("each category must contain exactly 15 queries")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(selected, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    manifest = {
        "selection_version": "balanced120-eight-category-task50-50-v1",
        "seed": args.seed,
        "source": str(args.source.resolve()),
        "source_sha256": sha256(args.source),
        "source_count": len(source_rows),
        "eligible_tasks": list(TASKS),
        "eligible_count": len(eligible),
        "output": str(args.output.resolve()),
        "output_sha256": sha256(args.output),
        "output_count": len(selected),
        "category_count": len(categories),
        "category_counts": counts(selected, "category"),
        "task_counts": counts(selected, "task_type"),
        "fake_brand_presence_counts": {
            "with_fake_brands": sum(bool(row.get("fake_brands")) for row in selected),
            "without_fake_brands": sum(not row.get("fake_brands") for row in selected),
        },
        "category_fake_brand_presence_counts": {
            category: {
                "with_fake_brands": sum(
                    row["category"] == category and bool(row.get("fake_brands"))
                    for row in selected
                ),
                "without_fake_brands": sum(
                    row["category"] == category and not row.get("fake_brands")
                    for row in selected
                ),
            }
            for category in categories
        },
        "category_task_quotas": quotas,
        "query_ids": query_ids,
    }
    args.manifest.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({key: value for key, value in manifest.items() if key != "query_ids"}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
