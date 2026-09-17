"""Minimal CLI: ``python -m brandgen generate ...``.

Specify run size and model directly: categories, brand types, brands (all=all),
pages per brand, LLM provider/model/temperature, and more.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .models import BRAND_SOURCE_POLICY, GenerationConfig
from .orchestrator import generate
from .utils import read_categories


GENERATOR_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = GENERATOR_ROOT.parent


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="brandgen", description="Generate controlled GEO benchmark assets.")
    subparsers = parser.add_subparsers(dest="command")

    gen = subparsers.add_parser("generate", help="Run the generation pipeline.")
    gen.add_argument("--categories", default=None,
                     help="Comma-separated categories; quote names containing spaces. Defaults to all categories in assets/store.md. Example: --categories \"children's shoes,liver supplements\"")
    gen.add_argument("--brand-source", nargs="+", default=["fake"],
                     choices=sorted(BRAND_SOURCE_POLICY.keys()),
                     help="Brand types; multiple allowed: real/niche/fake. Default: fake.")
    gen.add_argument("--brands", default="all",
                     help="Comma-separated brand names; all=all brands in the category's source file (default: all).")
    gen.add_argument("--pages-per-brand", type=int, default=10, help="Pages to generate per brand per level. Default: 10.")
    gen.add_argument("--level", default="all",
                     help="Comma-separated attack difficulty levels: L1/L2/L3; all=all three, with page_num pages per level (default: all).")
    gen.add_argument("--output-dir", type=Path, default=Path("output"), help="Output directory.")
    gen.add_argument("--anchor-date", default="2026-07-28", help="Anchor date.")
    gen.add_argument("--seed", default="gap", help="Reproducibility seed.")
    # Brand file directory
    gen.add_argument("--brand-source-dir", type=Path,
                     default=REPO_ROOT / "data" / "examples" / "brands",
                     help="Directory containing brand source files.")
    gen.add_argument("--domain-pool", type=Path, default=GENERATOR_ROOT / "assets" / "domain_pool.json",
                     help="Offline pool of URL appearances (domain_pool.json).")
    # LLM
    gen.add_argument("--llm-provider", default="none", choices=("none", "openai-compatible"),
                     help="LLM provider. Default: none (templates only).")
    gen.add_argument("--model", default=None, help="LLM model name, overriding the provider default (LLM_MODEL).")
    gen.add_argument("--llm-temperature", type=float, default=0.75)
    gen.add_argument("--llm-timeout", type=int, default=180)
    gen.add_argument("--llm-max-attempts", type=int, default=2)
    gen.add_argument("--llm-max-tokens", type=int, default=3000)
    gen.add_argument("--enable-thinking", default=None, choices=("true", "false"), help="Toggle model thinking mode.")
    gen.add_argument("--debug-llm", action="store_true")
    # Sampling
    gen.add_argument("--min-attacks-per-page", type=int, default=1)
    gen.add_argument("--max-attacks-per-page", type=int, default=2)
    gen.add_argument("--path-generated", type=float, default=-1.0,
                     help="Generated-path proportion, overriding each level's default preference. -1 (default)=automatically use the level's LEVEL_SPECS.path_ratio.")
    gen.add_argument("--force", action="store_true", help="Overwrite existing artifacts.")
    gen.add_argument("--profile-reuse", choices=("once", "per_page"), default="once",
                     help="Brand profile reuse: once (default)=generate once per brand (base profile + L3 professional signals) and reuse throughout; "
                          "per_page=regenerate before each page.")
    return parser


def config_from_args(args) -> GenerationConfig:
    store = GENERATOR_ROOT / "assets" / "store.md"
    if args.categories:
        categories = read_categories(store, args.categories)
    else:
        categories = read_categories(store, None)
    brands = (
        None if args.brands.strip().lower() in ("all", "")
        else [brand.strip() for brand in args.brands.split(",") if brand.strip()]
    )
    levels = _resolve_levels(args.level)
    # By default, use each level's LEVEL_SPECS.path_ratio for generated/modified (pass None).
    # An explicit nonnegative --path-generated is a global override. Note: the current
    # per-level flow uses the strategy's path_preference; the global override is not yet
    # wired up. See orchestrator.build_single_page.
    if args.path_generated >= 0.0:
        path_distribution = {
            "generated": args.path_generated,
            "modified": max(0.0, 1.0 - args.path_generated),
        }
    else:
        path_distribution = {"generated": 0.5, "modified": 0.5}  # Fallback only; actual values depend on level.
    enable_thinking = None
    if args.enable_thinking is not None:
        enable_thinking = args.enable_thinking == "true"
    return GenerationConfig(
        categories=categories,
        output_dir=args.output_dir,
        anchor_date=args.anchor_date,
        brand_source=tuple(args.brand_source),
        brand_source_dir=args.brand_source_dir,
        brands=brands,
        levels=levels,
        pages_per_brand=args.pages_per_brand,
        llm_provider=args.llm_provider,
        model=args.model,
        temperature=args.llm_temperature,
        timeout=args.llm_timeout,
        max_tokens=args.llm_max_tokens,
        max_attempts=args.llm_max_attempts,
        enable_thinking=enable_thinking,
        debug_llm=args.debug_llm,
        seed=args.seed,
        min_attacks_per_page=args.min_attacks_per_page,
        max_attacks_per_page=args.max_attacks_per_page,
        path_distribution=path_distribution,
        domain_pool=args.domain_pool,
        force=args.force,
        profile_reuse=args.profile_reuse,
    )


def _split(value: str) -> list[str]:
    import re
    return [p for p in re.split(r"[,，、\s]+", value) if p.strip()]


def _resolve_levels(value: str) -> tuple[str, ...]:
    """Parse --level: a subset of L1-L3; empty/all/none means all three."""
    all_levels = ("L1", "L2", "L3")
    raw = _split(value)
    if not raw or any(r.lower() in ("all", "none", "") for r in raw):
        return all_levels
    picked = tuple(r.upper() for r in raw if r.upper() in all_levels)
    if not picked:
        raise SystemExit(f"[error] invalid --level: {value}; expected L1/L2/L3 or all")
    return picked


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if not args.command:
        parser.print_help()
        return 1
    if args.command == "generate":
        config = config_from_args(args)
        result = generate(config)
        print(f"\n[done] brands={len(result['manifest'])} llm_stats={result['llm_stats']}")
        return 0
    parser.error(f"unknown command: {args.command}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
