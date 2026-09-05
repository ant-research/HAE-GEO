"""极简 CLI：``python -m brandgen generate ...``。

支持从命令行直接指定运行规模与模型：品类、品牌类型、品牌（all=全部）、
每品牌 page 数、LLM provider/model/temperature 等。
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
                     help="品类，逗号/空格分隔；默认读 assets/store.md 全部品类。例：儿童鞋,护肝片")
    gen.add_argument("--brand-source", nargs="+", default=["fake"],
                     choices=sorted(BRAND_SOURCE_POLICY.keys()),
                     help="品牌类型，可多选：real/niche/fake。默认 fake。")
    gen.add_argument("--brands", default="all",
                     help="指定品牌名，逗号分隔；all=跑来源文件里该品类全部品牌（默认 all）。")
    gen.add_argument("--pages-per-brand", type=int, default=10, help="每品牌、每 level 生成 page 数。默认 10。")
    gen.add_argument("--level", default="all",
                     help="攻击难度 level，逗号分隔：L1/L2/L3；all=全部 3 个，每 level 各 page_num 条（默认 all）。")
    gen.add_argument("--output-dir", type=Path, default=Path("output"), help="输出目录。")
    gen.add_argument("--anchor-date", default="2026-07-28", help="锚定日期。")
    gen.add_argument("--seed", default="gap", help="复现种子。")
    # 品牌文件目录
    gen.add_argument("--brand-source-dir", type=Path,
                     default=REPO_ROOT / "data" / "examples" / "brands",
                     help="品牌来源文件目录。")
    gen.add_argument("--domain-pool", type=Path, default=GENERATOR_ROOT / "assets" / "domain_pool.json",
                     help="离线 URL 外观池（domain_pool.json）。")
    # LLM
    gen.add_argument("--llm-provider", default="none", choices=("none", "openai-compatible"),
                     help="LLM provider。默认 none（纯模板）。")
    gen.add_argument("--model", default=None, help="LLM 模型名，覆盖 provider 默认（LLM_MODEL）。")
    gen.add_argument("--llm-temperature", type=float, default=0.75)
    gen.add_argument("--llm-timeout", type=int, default=180)
    gen.add_argument("--llm-max-attempts", type=int, default=2)
    gen.add_argument("--llm-max-tokens", type=int, default=3000)
    gen.add_argument("--enable-thinking", default=None, choices=("true", "false"), help="模型思考模式开关。")
    gen.add_argument("--debug-llm", action="store_true")
    # 采样
    gen.add_argument("--min-attacks-per-page", type=int, default=1)
    gen.add_argument("--max-attacks-per-page", type=int, default=2)
    gen.add_argument("--path-generated", type=float, default=-1.0,
                     help="generated 路径占比，覆盖各 level 默认偏好。-1（默认）= 按 level 的 LEVEL_SPECS.path_ratio 自动确定。")
    gen.add_argument("--force", action="store_true", help="覆盖已有产物。")
    gen.add_argument("--profile-reuse", choices=("once", "per_page"), default="once",
                     help="品牌画像复用粒度：once（默认）=每品牌生成一次（基础画像+L3 专业信号）全程复用；"
                          "per_page=每次生成页面前重新生成。")
    return parser


def config_from_args(args) -> GenerationConfig:
    store = GENERATOR_ROOT / "assets" / "store.md"
    if args.categories:
        categories = read_categories(store, args.categories)
    else:
        categories = read_categories(store, None)
    brands = None if args.brands.strip().lower() in ("all", "") else [b for b in _split(args.brands) if b]
    levels = _resolve_levels(args.level)
    # 默认按 level 的 LEVEL_SPECS.path_ratio 决定 generated/modified 占比（传 None）；
    # 显式指定 --path-generated（非负）时作为全局覆盖（注意：当前 per-level 流程由策略自带 path_preference，
    # 全局覆盖暂未接线，见 orchestrator.build_single_page）。
    if args.path_generated >= 0.0:
        path_distribution = {
            "generated": args.path_generated,
            "modified": max(0.0, 1.0 - args.path_generated),
        }
    else:
        path_distribution = {"generated": 0.5, "modified": 0.5}  # 仅作兜底；实际按 level 取
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
    """解析 --level：L1–L3 子集，空/all/none 表示全部 3 个。"""
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
