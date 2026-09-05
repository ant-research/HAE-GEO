from __future__ import annotations

import hashlib
import random
import re
from pathlib import Path


def digest_int(text: str) -> int:
    return int(hashlib.sha1(text.encode("utf-8")).hexdigest()[:12], 16)


def digest_hex(text: str, length: int = 10) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:length]


def stable_rng(seed: str, *parts: str) -> random.Random:
    """
    基于 seed 和上下文参数生成确定性的随机数生成器。

    相同的 seed 和 parts 输入始终生成相同的随机序列，
    用于保证数据生成过程可复现；通过传入不同的上下文参数，
    可以让不同品牌、品类、页面或攻击向量拥有相互独立的随机序列。

    Args:
        seed: 全局随机种子，用于控制整体数据生成的随机性。
        *parts: 上下文标识，例如品类、品牌、页面类型、页面索引等，
            用于区分不同生成任务的随机序列。

    Returns:
        random.Random: 基于输入参数初始化的确定性随机数生成器。

    Example:
        stable_rng("42", "洗衣液", "某品牌", "review", "0")
    """
    return random.Random(digest_int("|".join((seed, *parts))))


def squeeze(text: str) -> str:
    return re.sub(r"\s+", " ", str(text or "")).strip()


def compact_text(value, max_len: int = 80) -> str:
    item = squeeze(value)
    item = item.strip(" \t\r\n-_:：,，.。;；\"'“”‘’[]{}()（）<>《》")
    return item[:max_len].strip()


def split_items(value, fallback: tuple[str, ...], limit: int, max_len: int = 64) -> tuple[str, ...]:
    if isinstance(value, str):
        raw = [part for part in re.split(r"[,，、\n\r\t]+", value) if part.strip()]
    elif isinstance(value, list):
        raw = [str(part) for part in value if str(part).strip()]
    else:
        raw = list(fallback)

    items: list[str] = []
    for part in raw:
        item = compact_text(part, max_len=max_len)
        if item and item not in items:
            items.append(item)
    return tuple(items[:limit]) or fallback[:limit]


def safe_filename_part(value: str, fallback: str = "asset") -> str:
    cleaned = re.sub(r"[^\w\u4e00-\u9fff-]+", "_", value).strip("_")
    return cleaned[:80] or fallback


def split_list_text(text: str) -> list[str]:
    return [part.strip() for part in re.split(r"[,，、\s]+", text) if part.strip()]


def read_categories(store: Path, categories_arg: str | None) -> list[str]:
    text = categories_arg if categories_arg else store.read_text(encoding="utf-8")
    categories = split_list_text(text)
    if not categories:
        raise ValueError("no categories found")
    return categories


def read_brand_names(brands_arg: str | None, brand_file: Path | None = None) -> list[str] | None:
    chunks: list[str] = []
    if brands_arg:
        chunks.append(brands_arg)
    if brand_file:
        chunks.append(brand_file.read_text(encoding="utf-8"))
    if not chunks:
        return None
    brands = split_list_text("\n".join(chunks))
    if not brands:
        raise ValueError("no brand names found")
    return brands


def read_category_brand_names(category_brands_arg: str | None) -> dict[str, list[str]] | None:
    if not category_brands_arg:
        return None
    mapping: dict[str, list[str]] = {}
    for entry in re.split(r"[;\n\r]+", category_brands_arg):
        entry = entry.strip()
        if not entry:
            continue
        separator = "\t" if "\t" in entry else None
        if separator is None:
            for candidate in ("=", ":", "："):
                if candidate in entry:
                    separator = candidate
                    break
        if separator is None:
            raise ValueError(f"invalid category brand entry: {entry}")
        category, brands_text = entry.split(separator, 1)
        category = category.strip()
        brands = split_list_text(brands_text)
        if not category or not brands:
            raise ValueError(f"invalid category brand entry: {entry}")
        mapping[category] = brands
    if not mapping:
        raise ValueError("no category brand names found")
    return mapping
