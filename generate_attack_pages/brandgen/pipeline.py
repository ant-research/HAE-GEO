"""Thin re-export preserving ``from .pipeline import generate, generate_one_brand``.

See :mod:`brandgen.orchestrator` for the generation implementation and
``PIPELINE_REQUIREMENTS.md`` for the requirements (formerly in pipeline.py).
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
