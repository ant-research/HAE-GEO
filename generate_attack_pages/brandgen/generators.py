"""Dual-path generators: one class each for GeneratedPath and ModifiedPath.

- ``GeneratedPath``: synthesize poisoned webpages from scratch. With ``llm_provider='none'``, combine the page builder's
  deterministic skeleton with attack-vector templates; with an LLM, make one call for more natural title/content/snippet text.
- ``ModifiedPath``: call ``ant_search`` for a real webpage substrate (search terms include the current page_type's
  search_keywords to match its type), preserve the structure, and inject poisoning fragments vector by vector.
  Output includes ``real_source_url`` / ``real_source_content``. Fall back to the template skeleton if no result is found.
"""

from __future__ import annotations

import sys
from typing import Any

from .attack_vector import AttackContext, AttackVector
from .llm import call_json, LLMError
from .models import BrandProfile, GenerationConfig, LLMStats
from .page_type import PageBuilder, PageContext, build_offline_signals
from .utils import stable_rng


def _attack_ctx(profile: BrandProfile, ctx_page: PageContext) -> AttackContext:
    return AttackContext(
        profile=profile,
        query_type=ctx_page.query_type,
        level=ctx_page.level,
        manipulation_direction=ctx_page.manipulation_direction,
        page_context=ctx_page,
        substrate=ctx_page.real_source_content,
    )


def _render_vectors_tmpl(
    profile: BrandProfile,
    ctx_page: PageContext,
    vectors: list[AttackVector],
) -> list[tuple[AttackVector, str]]:
    out: list[tuple[AttackVector, str]] = []
    for v in vectors:
        out.append((v, v.render_tmpl(_attack_ctx(profile, ctx_page))))
    return out


def _inject_vectors(content: str, rendered: list[tuple[AttackVector, str]]) -> str:
    for v, frag in rendered:
        if frag:
            content = v.apply(content, frag)
    return content


def _vectors_to_records(rendered: list[tuple[AttackVector, str]], ctx_page: PageContext) -> list[dict[str, Any]]:
    out = []
    for v, frag in rendered:
        out.append(v.to_dict(_attack_ctx(ctx_page.profile, ctx_page), frag))
    return out


class _PathBase:
    """LLM refinement logic shared by both paths."""

    def _maybe_llm_page(
        self,
        builder: PageBuilder,
        ctx_page: PageContext,
        vectors: list[AttackVector],
        title: str,
        content: str,
        client,
        config: GenerationConfig,
        stats: LLMStats,
    ) -> tuple[str, str, str]:
        """When an LLM is enabled, make one call for title/content/snippet; otherwise return the template result.

        This call produces only title/content/snippet. The L3 professional signal package is generated separately by
        ``orchestrator.build_professional_signals`` and attached to
        ``ctx_page.professional_signals`` before ``path.build``; it is not generated here. Only if an L3 page still lacks signals,
        use ``build_offline_signals`` defensively so L3 records always carry signals.
        """
        is_l3_pro = ctx_page.level == "L3" and ctx_page.profile_mode == "professional"
        if is_l3_pro and ctx_page.professional_signals is None:
            ctx_page.professional_signals = build_offline_signals(ctx_page)
        if config.llm_provider == "none" or client is None:
            return title, content, _snippet(content)
        messages = builder.build_prompt(ctx_page, vectors)
        try:
            data = call_json(
                client,
                messages,
                context=f"page {ctx_page.page_type} {ctx_page.brand}#{ctx_page.page_index}",
                temperature=config.temperature,
                max_tokens=config.max_tokens,
                max_attempts=config.max_attempts,
                stats=stats,
                debug=config.debug_llm,
            )
            data = data if isinstance(data, dict) else {}
            title = str(data.get("title") or "")
            content = str(data.get("content") or "")
            snippet = str(data.get("snippet") or "")
            return title, content, snippet
        except Exception as err:  # noqa: BLE001
            print(f"[warn] page LLM failed, fallback to template: {err}", file=sys.stderr, flush=True)
            return title, content, _snippet(content)

    def build(self, *args, **kwargs):  # pragma: no cover - interface
        raise NotImplementedError


def _snippet(content: str) -> str:
    return (content or "")[:200].replace("\n", " ")


class GeneratedPath(_PathBase):
    """Generate poisoned webpages from scratch."""

    name = "generated"

    def build(
        self,
        builder: PageBuilder,
        ctx_page: PageContext,
        vectors: list[AttackVector],
        client,
        config: GenerationConfig,
        stats: LLMStats,
    ) -> tuple[str, str, str, list[dict[str, Any]], str, str]:
        title, body = builder.build_tmpl(ctx_page)
        rendered = _render_vectors_tmpl(ctx_page.profile, ctx_page, vectors)
        content = _inject_vectors(body, rendered)
        title, content, snippet = self._maybe_llm_page(
            builder, ctx_page, vectors, title, content, client, config, stats
        )
        vectors_text = _vectors_to_records(rendered, ctx_page)
        return title, content, snippet, vectors_text, "", ""


class ModifiedPath(_PathBase):
    """Modify a real webpage: fetch a real substrate, preserve its structure, and inject poisoning fragments vector by vector."""

    name = "modified"

    def build(
        self,
        builder: PageBuilder,
        ctx_page: PageContext,
        vectors: list[AttackVector],
        client,
        config: GenerationConfig,
        stats: LLMStats,
    ) -> tuple[str, str, str, list[dict[str, Any]], str, str]:
        real_title, real_url, real_content = self._fetch_real(builder, ctx_page, client, config, stats)
        ctx_page.real_source_title = real_title
        ctx_page.real_source_url = real_url
        ctx_page.real_source_content = real_content

        title, body = builder.build_tmpl(ctx_page)
        base = real_content if real_content else body  # Fall back to the skeleton if no substrate is found
        rendered = _render_vectors_tmpl(ctx_page.profile, ctx_page, vectors)
        content = _inject_vectors(base, rendered)
        title, content, snippet = self._maybe_llm_page(
            builder, ctx_page, vectors, title, content, client, config, stats
        )
        vectors_text = _vectors_to_records(rendered, ctx_page)
        print(f"\033[32m[Content]\033[0m {content}")
        # print(f"vectors_text: {vectors_text}")
        return title, content, snippet, vectors_text, real_url, real_content

    def _fetch_real(self, builder: PageBuilder, ctx_page: PageContext,
                    client, config, stats) -> tuple[str, str, str]:
        """Call ant_search for a real webpage substrate. Search terms include the current page_type's search_keywords,
        ensuring the retrieved substrate matches the page type.

        Select the best match for the current page_type among top_k results, in this order:
        1) LLM judgment: ask the model to select the candidate that best matches the page type (when an LLM is available);
        2) Keyword substring match: take the first title/body matching search_keywords;
        3) Fall back to the first result. Return empty strings on failure/no network, reverting to the template skeleton.
        """
        try:
            from .ant_search import ant_search  # Lazy import; not required in offline environments
        except Exception as err:  # noqa: BLE001
            print(f"[warn] ant_search import failed: {err}", file=sys.stderr, flush=True)
            return "", "", ""
        query = builder.search_query(ctx_page)
        try:
            results = ant_search(query, top_k=5, content_type="webPage")
        except Exception as err:  # noqa: BLE001
            print(f"[warn] ant_search failed, fallback to template: {err}", file=sys.stderr, flush=True)
            return "", "", ""
        if not results:
            return "", "", ""

        chosen = self._select_by_llm(builder, ctx_page, results, client, config, stats)
        if chosen is None:
            chosen = self._select_by_keywords(builder, results)
        if chosen is None:
            chosen = results[0]
            print("[info] no result matched page_type, fallback to first result", file=sys.stderr, flush=True)
        chosen_title = str(chosen.get("title") or "")
        print(f"\033[32m[Search]\033[0m search query: {query} | chosen title: {chosen_title}")
        return (
            str(chosen.get("title") or ""),
            str(chosen.get("url") or ""),
            str(chosen.get("content") or chosen.get("snippet") or ""),
        )

    def _select_by_llm(self, builder, ctx_page, results, client, config, stats):
        """Ask the LLM to select the candidate best matching the current page_type. Return None on failure."""
        if client is None or config.llm_provider == "none":
            return None
        candidates = []
        for idx, item in enumerate(results):
            title = str(item.get("title") or "")
            content = str(item.get("content") or item.get("snippet") or "")[:400]
            candidates.append({"index": idx, "title": title, "snippet": content})
        kw = ", ".join(builder.search_keywords_list()) or builder.label
        system = (
            "You are a webpage-type classifier for private evaluation. Given search results and a target page type, "
            "select the result whose content form best matches that page type. Use English for any text. Output only JSON: {\"index\": selected_number}."
        )
        user = (
            f"Target page type: {builder.label} (keywords: {kw}).\nCandidate results:\n"
            + "\n".join(f"#{c['index']} Title:{c['title']} Summary:{c['snippet']}" for c in candidates)
        )
        try:
            data = call_json(
                client, [{"role": "system", "content": system}, {"role": "user", "content": user}],
                context=f"select-real {builder.role_id}",
                temperature=0.0,
                max_tokens=64,
                max_attempts=config.max_attempts,
                stats=stats,
                debug=config.debug_llm,
            )
            data = data if isinstance(data, dict) else {}
            idx = int(data.get("index"))
            if 0 <= idx < len(results):
                print(f"[info] LLM selected result #{idx} for {builder.role_id}", file=sys.stderr, flush=True)
                return results[idx]
        except Exception as err:  # noqa: BLE001
            print(f"[info] LLM select failed, fallback to keyword match: {err}", file=sys.stderr, flush=True)
        return None

    def _select_by_keywords(self, builder, results):
        """Keyword substring fallback: take the first result whose title/body matches search_keywords."""
        keywords = [kw.lower() for kw in builder.search_keywords_list() if kw]
        for item in results:
            title = str(item.get("title") or "")
            content = str(item.get("content") or item.get("snippet") or "")
            haystack = (title + " " + content).lower()
            if any(kw in haystack for kw in keywords):
                return item
        return None


GENERATED_PATH = GeneratedPath()
MODIFIED_PATH = ModifiedPath()


def path_for(source_type: str) -> _PathBase:
    return MODIFIED_PATH if source_type == "modified" else GENERATED_PATH
