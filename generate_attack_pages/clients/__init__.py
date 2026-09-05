"""Public LLM clients for controlled benchmark-asset generation.

Import concrete clients from their modules, e.g. ``clients.openai_compatible``.
Keeping this package init light avoids requiring optional provider
dependencies when another provider is used.
"""

__all__ = ["base", "openai_compatible"]
