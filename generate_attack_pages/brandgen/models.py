"""数据模型与运行配置（精简重建版）。

保留 ``LLMStats``（``llm.py`` 依赖），并提供生成链路所需的
``BrandProfile`` / ``PageRecord`` / ``GenerationConfig`` / ``BRAND_SOURCE_POLICY``。

品牌画像生产策略：``llm_provider != none`` 时由 LLM 生成（沿用 HEAD 的 llm_profile
思路，落在 ``orchestrator.build_profile``），``none`` 时回退到确定性模板
``fallback_profile``。品牌名一律从 ``data/brands/*.json`` 读取，不调 LLM。
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


GENERATOR_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = GENERATOR_ROOT.parent


# 品牌来源 → 方向/品牌文件 策略（单一事实源）。
# geo_direction 决定操纵方向（positive=扶植假/小众品牌，negative=抑制真实品牌）。
# brand_source 键名直接用对外品牌类型：real / niche / fake。
BRAND_SOURCE_POLICY: dict[str, dict[str, str]] = {
    "real": {"file": "real_brands.json", "geo_direction": "negative", "brand_type": "real"},
    "niche": {"file": "niche_brands.json", "geo_direction": "positive", "brand_type": "niche"},
    "fake": {"file": "fake_brands.json", "geo_direction": "positive", "brand_type": "fake"},
}


@dataclass(frozen=True)
class BrandProfile:
    """品牌画像。身份层 + 邻近品牌池 + 业务画像字段。

    软画像字段（operator_name / positioning / products / competitors …）由
    ``orchestrator.build_profile`` 填充：LLM 可用时让模型生成，否则用确定性模板。
    随页面漂移的虚假指标（销量/榜单/认证）不出现在画像里，由攻击向量在渲染页面时虚构。

    发布形态（record["profile"]）按攻击难度 level 分层，**不由本类决定**而在下游控制
    （见 ``depth.LevelStrategy.profile_mode`` 与 ``page_type.PageBuilder.make_record``）：
    L1 发布 ``{}``（内部仍用身份核 profile 驱动 URL/种子/query），L2 发布 ``to_dict()``，
    L3 发布 ``to_dict()`` 并附加 ``professional_signals``（参数/评分/引用/机构/认证/用户反馈）。
    专业信号包由独立的 ``orchestrator.build_professional_signals`` 调用按品类特性生成（LLM 优先，
    离线/失败按品类确定性兜底），是独立对象、**不应**作为字段加到 BrandProfile 上。
    复用粒度由 ``GenerationConfig.profile_reuse`` 控制：once=每品牌一份、per_page=每页现建。
    """

    category: str
    brand: str
    brand_type: str = "fake"
    geo_direction: str = "positive"
    operator_name: str = ""
    headquarters: str = ""
    brand_positioning: str = ""
    target_audience: str = ""
    price_range: str = ""
    products: tuple[str, ...] = ()
    features: tuple[str, ...] = ()
    competitors: tuple[str, ...] = ()
    pain_points: tuple[str, ...] = ()
    channels: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        for key in ("products", "features", "competitors", "pain_points", "channels"):
            data[key] = list(data[key])
        return data

    def anchor_dict(self) -> dict[str, Any]:
        """页面间稳定的身份字段（不含随页面漂移的虚假指标）。"""
        return {
            "category": self.category,
            "brand": self.brand,
            "brand_type": self.brand_type,
            "operator_name": self.operator_name,
            "headquarters": self.headquarters,
            "brand_positioning": self.brand_positioning,
            "target_audience": self.target_audience,
            "products": list(self.products),
            "features": list(self.features),
            "competitors": list(self.competitors),
            "pain_points": list(self.pain_points),
        }


@dataclass
class PageRecord:
    """单条投毒网页记录，字段对齐需求文档定义的输出 schema。"""

    category: str = ""
    brand: str = ""
    brand_type: str = ""
    query: str = ""
    answer: str = ""
    title: str = ""
    url: str = ""
    timestamp: str = ""
    discovered_via: str = ""
    content: str = ""
    snippet: str = ""
    page_type: str = ""
    source_type: str = "generated"  # generated | modified
    level: str = "L1"
    manipulation_direction: str = "positive"  # positive | negative
    attack_vectors: list[dict[str, Any]] = field(default_factory=list)
    real_source_url: str = ""
    real_source_content: str = ""
    real_source_title: str = ""
    ext: dict[str, Any] = field(default_factory=dict)
    profile: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "category": self.category,
            "brand": self.brand,
            "brand_type": self.brand_type,
            "query": self.query,
            "answer": self.answer,
            "title": self.title,
            "url": self.url,
            "timestamp": self.timestamp,
            "discovered_via": self.discovered_via,
            "content": self.content,
            "snippet": self.snippet,
            "page_type": self.page_type,
            "source_type": self.source_type,
            "level": self.level,
            "manipulation_direction": self.manipulation_direction,
            "attack_vectors": [dict(v) for v in self.attack_vectors],
            "real_source_title": self.real_source_title,
            "real_source_url": self.real_source_url,
            "real_source_content": self.real_source_content,
            "ext": dict(self.ext),
            "profile": dict(self.profile),
        }


@dataclass
class GenerationConfig:
    """生成链路运行配置（精简版，去掉 grt/质量审计/引用检索相关字段）。"""

    categories: list[str] = field(default_factory=list)
    output_dir: Path = Path("output")
    anchor_date: str = "2026-07-28"
    brand_source: tuple[str, ...] = ("fake",)  # real | niche | fake，可多选
    brand_source_dir: Path = REPO_ROOT / "data" / "examples" / "brands"
    brands: list[str] | None = None  # None / ["all"] = 跑来源文件里全部品牌
    pages_per_brand: int = 10
    # 攻击难度 level：L1–L3 的子集；传入空/None/all 表示全部 3 个（每 level 各 N 条）
    levels: tuple[str, ...] = ("L1", "L2", "L3")
    # LLM
    llm_provider: str = "none"  # none | openai-compatible
    model: str | None = None
    temperature: float = 0.75
    timeout: int = 180
    max_tokens: int = 3000
    max_attempts: int = 2
    enable_thinking: bool | None = None
    debug_llm: bool = False
    # 采样
    seed: str = "gap"
    min_attacks_per_page: int = 1
    max_attacks_per_page: int = 3
    path_distribution: dict[str, float] = field(
        default_factory=lambda: {"generated": 0.5, "modified": 0.5}
    )
    domain_pool: Path = GENERATOR_ROOT / "assets" / "domain_pool.json"
    attack_library: Path = GENERATOR_ROOT / "assets" / "attack_family_library.json"
    force: bool = False
    # 品牌画像复用粒度：once=每品牌生成一次（基础画像 + L3 专业信号）全程复用；
    # per_page=每次生成页面前重新生成。默认 once。
    profile_reuse: str = "once"

    @property
    def brands_all(self) -> bool:
        if not self.brands:
            return True
        lowered = [b.strip().lower() for b in self.brands if b.strip()]
        return not lowered or all(b in {"all", ""} for b in lowered)


@dataclass
class LLMStats:
    calls: int = 0
    retries: int = 0
    failures: int = 0

    def to_dict(self) -> dict[str, int]:
        return asdict(self)
