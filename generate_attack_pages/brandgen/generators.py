"""双路径生成器：GeneratedPath 与 ModifiedPath，各一个 class。

- ``GeneratedPath``：凭空合成投毒网页。``llm_provider='none'`` 时直接用 page builder
  的确定性骨架 + 各攻击向量模板拼接；启用 LLM 时调一次合成更自然的 title/content/snippet。
- ``ModifiedPath``：调 ``ant_search`` 取真实网页作为基底（检索关键词含当前 page_type 的
  search_keywords，保证基底与本页面类型同类），保持结构，逐个向量注入投毒段。
  产出携带 ``real_source_url`` / ``real_source_content``。未检索到时退回模板骨架。
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
    """两条路径共享的 LLM 润色逻辑。"""

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
        """启用 LLM 时调一次合成 title/content/snippet；否则返回模板结果。

        本调用只产出 title/content/snippet。L3 专业信号包由独立的
        ``orchestrator.build_professional_signals`` 生成并已在 ``path.build`` 之前挂到
        ``ctx_page.professional_signals``；此处不再产出。仅当 L3 页面因故仍未挂载时，
        防御性地用 ``build_offline_signals`` 兜底，保证 L3 record 始终携带信号。
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
    """凭空生成投毒网页。"""

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
    """基于真实网页改造：取真实基底，保持结构，逐个向量注入投毒段。"""

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
        base = real_content if real_content else body  # 未检索到基底则退回骨架
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
        """调 ant_search 取一条真实网页作为基底。检索词含当前 page_type 的 search_keywords，
        保证检索基底与本页面类型同类。

        在 top_k 结果里选最贴合当前 page_type 的一条，优先级：
        1) LLM 判断：让模型从候选里选最符合该页面类型的一条（LLM 可用时）；
        2) 关键词子串匹配：标题/正文命中 search_keywords 的取首个命中的；
        3) 取第一条兜底。失败/无网络时返回空串，退回模板骨架。
        """
        try:
            from .ant_search import ant_search  # 延迟导入，离线环境不依赖
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
        """让 LLM 从候选结果里选出最符合当前 page_type 的一条。失败返回 None。"""
        if client is None or config.llm_provider == "none":
            return None
        candidates = []
        for idx, item in enumerate(results):
            title = str(item.get("title") or "")
            content = str(item.get("content") or item.get("snippet") or "")[:400]
            candidates.append({"index": idx, "title": title, "snippet": content})
        kw = "、".join(builder.search_keywords_list()) or builder.label
        system = (
            "你是网页类型判别器（私有评测用）。给定若干检索结果与目标页面类型，"
            "选出在内容形态上最符合该页面类型的那一条。只输出 JSON：{\"index\": 选中的编号}。"
        )
        user = (
            f"目标页面类型：{builder.label}（关键词：{kw}）。\n候选结果：\n"
            + "\n".join(f"#{c['index']} 标题:{c['title']} 摘要:{c['snippet']}" for c in candidates)
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
        """关键词子串匹配兜底：标题/正文命中 search_keywords 的取首个命中的。"""
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