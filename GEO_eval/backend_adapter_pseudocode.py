"""Pseudocode for replacing the local backend with an external search index.

This file is deliberately non-executable. Do not commit private endpoints,
credentials, tenant identifiers, or storage paths when implementing it.
"""


def search_backend(query, environment, top_k):
    config = load_environment_config(environment)  # public ID -> deployment secret
    candidates = bm25_retrieve(config.index, query, top_k=5 * top_k)
    candidates += dense_retrieve(config.index, query, top_k=5 * top_k)
    ranked = rerank_and_deduplicate(query, candidates)[:top_k]
    cache_by_environment_and_url(environment, ranked)
    return [public_search_projection(doc) for doc in ranked]


def scrape_backend(url, environment, content_budget):
    document = read_environment_scoped_cache(environment, url)
    return truncate_to_budget(document.content, content_budget)
