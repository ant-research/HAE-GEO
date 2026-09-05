"""Credential-free OpenAI-compatible client for synthetic page generation."""

from __future__ import annotations

import os
from typing import Any, Optional

from .base import BaseModelClient, ModelResponse


class OpenAICompatibleClient(BaseModelClient):
    provider_name = "openai-compatible"

    def __init__(self, **kwargs: Any):
        kwargs.setdefault("api_key", os.environ.get("GENERATOR_API_KEY"))
        kwargs.setdefault("base_url", os.environ.get("GENERATOR_BASE_URL"))
        kwargs.setdefault("model", os.environ.get("GENERATOR_MODEL", "your-generator-model"))
        super().__init__(**kwargs)

    def _validate_config(self) -> None:
        if not self.api_key or not self.base_url or not self.model:
            raise ValueError("Set GENERATOR_API_KEY, GENERATOR_BASE_URL, and GENERATOR_MODEL")

    def chat_messages(self, messages: list, temperature: float = 0.7,
                      max_tokens: Optional[int] = None, **kwargs: Any) -> ModelResponse:
        from openai import OpenAI

        client = OpenAI(api_key=self.api_key, base_url=self.base_url, timeout=self.timeout)
        request: dict[str, Any] = {
            "model": self.model, "messages": messages, "temperature": temperature,
        }
        if max_tokens is not None:
            request["max_tokens"] = max_tokens
        request.update(kwargs)
        try:
            response = client.chat.completions.create(**request)
            content = response.choices[0].message.content or ""
            usage = response.usage.model_dump() if response.usage else None
            return ModelResponse(True, content, self.model, usage=usage)
        except Exception as exc:  # noqa: BLE001
            return ModelResponse(False, "", self.model, error_message=f"{type(exc).__name__}: {exc}")

    def chat(self, message: str, **kwargs: Any) -> ModelResponse:
        return self.chat_messages([{"role": "user", "content": message}], **kwargs)
