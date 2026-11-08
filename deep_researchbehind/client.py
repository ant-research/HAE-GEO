"""HTTP client for an explicitly configured Xunsearch environment."""

import math
from urllib.parse import urlsplit

import requests


def search_documents(entry, environment, query, top_k, dataset='all'):
    endpoint = str(entry.get('endpoint', '')).rstrip('/')
    url = urlsplit(endpoint)
    if url.scheme not in ('http', 'https') or not url.netloc:
        raise ValueError('An xunsearch environment needs an HTTP(S) endpoint')
    revision = entry.get('index_revision')
    if not revision:
        raise ValueError('An xunsearch environment needs index_revision; rebuild its configuration')
    local = url.hostname in ('localhost', '127.0.0.1', '::1')
    session = requests.Session()
    # A machine-wide proxy must not redirect a local backend request.
    if local:
        session.trust_env = False
    with session:
        response = session.post(endpoint + '/search', json=dict(
            environment=environment, query=query, top_k=top_k, dataset=dataset,
            index_revision=revision,
        ), timeout=float(entry.get('timeout_seconds', 30)))
        response.raise_for_status()
        payload = response.json()
    if (payload.get('success') is not True or payload.get('environment') != environment
            or payload.get('index_revision') != revision):
        raise ValueError('Search backend returned a failed or mismatched environment/revision')
    docs = payload.get('documents')
    if not isinstance(docs, list) or len(docs) > top_k:
        raise ValueError('Search backend returned an invalid document list')
    seen = set()
    for doc in docs:
        if not isinstance(doc, dict) or not doc.get('url') or not isinstance(doc.get('content'), str):
            raise ValueError('Search backend documents need a URL and stored body')
        if doc['url'] in seen:
            raise ValueError('Search backend returned duplicate URLs')
        score = doc.get('score')
        if isinstance(score, bool) or not isinstance(score, (int, float)) or not math.isfinite(score):
            raise ValueError('Search backend returned an invalid ranking score')
        seen.add(doc['url'])
    return docs
