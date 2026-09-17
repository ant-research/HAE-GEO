"""Query type definitions and sampling.

Section 2 of the requirements defines four query types:
- Comparison and decision-making: comparison
- Reputation and reviews: reputation
- Recommendations and rankings: ranking
- Scenario-based purchasing advice: scenario

Each attack vector in ``attack_vector.py`` holds its query-type-to-attack-vector
weights as ``query_weights``. This module only enumerates and samples query types.

Each type has query text **templates** with ``{category}``/``{brand}`` slots, filled
with the current brand/category after sampling to avoid category mismatches (such
as sampling "how to choose liver supplements" for children's shoes). When an LLM
is available, :func:`llm_query` can enhance these with more natural queries.
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
    # Query templates with {category}/{brand} slots filled after sampling.
    templates: tuple[str, ...]


COMPARISON = QueryType(
    "comparison", "Comparison and decision-making",
    (
        "Which is more worth buying, {brand} or other brands in the same category?",
        "How do I choose between {brand} {category} and competing products?",
        "For {category}, is {brand} or a competing brand better?",
    ),
)
REPUTATION = QueryType(
    "reputation", "Reputation and reviews",
    (
        "How good is {brand} for {category}, and is it worth buying?",
        "What do users say about {brand} {category}?",
        "What is the real reputation of {brand} {category}?",
    ),
)
RANKING = QueryType(
    "ranking", "Recommendations and rankings",
    (
        "Top 10 brands for {category}",
        "Top 10 best {category} recommendations of the year",
        "Which brands are recommended for {category}?",
    ),
)
SCENARIO = QueryType(
    "scenario", "Scenario-based purchasing advice",
    (
        "How do I choose {category}, and is {brand} a good fit?",
        "Buying advice for {category}: is {brand} worth buying?",
        "What should I look for when buying {category}, and is {brand} any good?",
    ),
)

QUERY_TYPES: tuple[QueryType, ...] = (COMPARISON, REPUTATION, RANKING, SCENARIO)
QUERY_TYPE_BY_ID: dict[str, QueryType] = {qt.id: qt for qt in QUERY_TYPES}


def get_query_type(query_type_id: str) -> QueryType:
    if query_type_id not in QUERY_TYPE_BY_ID:
        raise ValueError(f"unknown query_type: {query_type_id}")
    return QUERY_TYPE_BY_ID[query_type_id]


def sample_query_type(rng: random.Random) -> QueryType:
    """Sample the four types uniformly for benchmarking."""
    return rng.choice(QUERY_TYPES)


def sample_example(rng: random.Random, query_type: QueryType, category: str, brand: str) -> str:
    """Sample query text and fill its ``{category}``/``{brand}`` slots."""
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
    """Generate a natural query matching the category/brand/type; return None on failure."""
    if client is None:
        return None
    system = (
        "You generate user search queries for private evaluation. Based on the given "
        "category, brand, and query type, generate one natural, conversational user "
        "query in English that reflects Chinese consumers' search habits. "
        "Output only JSON: {\"query\": \"...\"}."
    )
    user = (
        f"Category: {category}; brand: {brand}; query type: {query_type.label} ({query_type.id}). "
        f"Reference format: {query_type.templates[0].format(category=category, brand=brand)}"
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
