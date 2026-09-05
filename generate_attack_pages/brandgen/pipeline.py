"""薄 re-export：保持旧 ``from .pipeline import generate, generate_one_brand`` 可用。

生成本体的实现见 :mod:`brandgen.orchestrator`；本需求文档见
``PIPELINE_REQUIREMENTS.md``（原 pipeline.py 内容）。
"""

from .orchestrator import (  # noqa: F401
    build_profile,
    fallback_profile,
    generate,
    generate_one_brand,
    has_failures,
    load_brand_names,
)

__all__ = [
    "build_profile",
    "fallback_profile",
    "generate",
    "generate_one_brand",
    "has_failures",
    "load_brand_names",
]