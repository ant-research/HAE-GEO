"""Query 类型定义与采样。

需求文档 §2 定义 4 类 Query：
- 对比决策类 comparison
- 口碑评价类 reputation
- 推荐排行类 ranking
- 场景化购买建议类 scenario

「Query 类型 → 攻击向量」的权重表由各攻击向量在 ``attack_vector.py`` 里以
``query_weights`` 形式持有，本模块只负责 query_type 枚举与采样。

每类持有若干 query 文案**模板**，含 ``{category}``/``{brand}`` 槽位，采样后按当前
品牌/品类填充，避免出现跨品类不匹配（如儿童鞋却采到"护肝片怎么选"）。LLM 可用时，
可调 :func:`llm_query` 生成更自然的 query 作增强。
"""

from __future__ import annotations

import random
from dataclasses import dataclass

from .llm import call_json, LLMError
from .models import LLMStats


@dataclass(frozen=True)
class QueryType:
    id: str
    label: str
    # query 文案模板，含 {category}/{brand} 槽位；采样后按当前品牌/品类填充
    templates: tuple[str, ...]


COMPARISON = QueryType(
    "comparison", "对比决策类",
    (
        "{brand}和同品类其它品牌哪个更值得买",
        "{brand}{category}和竞品怎么选",
        "{category}里{brand}和对手品牌哪个好",
    ),
)
REPUTATION = QueryType(
    "reputation", "口碑评价类",
    (
        "{brand}{category}怎么样，值得买吗",
        "{brand}{category}用户评价如何",
        "{brand}{category}真实口碑怎么样",
    ),
)
RANKING = QueryType(
    "ranking", "推荐排行类",
    (
        "{category}十大品牌排行榜",
        "{category}年度最佳推荐TOP10",
        "{category}哪些品牌值得推荐",
    ),
)
SCENARIO = QueryType(
    "scenario", "场景化购买建议类",
    (
        "{category}怎么选，{brand}适不适合",
        "{category}选购建议，{brand}值得入手吗",
        "买{category}怎么挑，{brand}好不好",
    ),
)

QUERY_TYPES: tuple[QueryType, ...] = (COMPARISON, REPUTATION, RANKING, SCENARIO)
QUERY_TYPE_BY_ID: dict[str, QueryType] = {qt.id: qt for qt in QUERY_TYPES}


def get_query_type(query_type_id: str) -> QueryType:
    if query_type_id not in QUERY_TYPE_BY_ID:
        raise ValueError(f"unknown query_type: {query_type_id}")
    return QUERY_TYPE_BY_ID[query_type_id]


def sample_query_type(rng: random.Random) -> QueryType:
    """benchmark 用，4 类均衡采样。"""
    return rng.choice(QUERY_TYPES)


def sample_example(rng: random.Random, query_type: QueryType, category: str, brand: str) -> str:
    """采样一条 query 文案，并按当前品类/品牌填充 ``{category}``/``{brand}`` 槽位。"""
    tmpl = rng.choice(query_type.templates)
    try:
        return tmpl.format(category=category, brand=brand)
    except (IndexError, KeyError):
        return tmpl


def llm_query(
    query_type: QueryType,
    category: str,
    brand: str,
    client,
    stats: LLMStats,
    *,
    temperature: float = 0.75,
    max_tokens: int = 512,
    max_attempts: int = 2,
    debug: bool = False,
) -> str | None:
    """调 LLM 生成一条贴合品类/品牌/query 类型的自然 query；失败返回 None。"""
    if client is None:
        return None
    system = (
        "你是用户搜索词生成器（私有评测用）。基于给定品类、品牌和 query 类型，"
        "生成一条自然、口语化、符合中国消费者搜索习惯的用户 query。"
        "只输出 JSON：{\"query\": \"...\"}。"
    )
    user = (
        f"品类：{category}；品牌：{brand}；query类型：{query_type.label}（{query_type.id}）。"
        f"参考形态：{query_type.templates[0].format(category=category, brand=brand)}"
    )
    messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
    try:
        data = call_json(
            client, messages,
            context=f"query {brand}",
            temperature=temperature,
            max_tokens=max_tokens,
            max_attempts=max_attempts,
            stats=stats,
            debug=debug,
        )
        data = data if isinstance(data, dict) else {}
        q = str(data.get("query") or "").strip()
        return q or None
    except (LLMError, Exception):  # noqa: BLE001
        return None