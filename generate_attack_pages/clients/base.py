"""Shared model client interfaces."""

from __future__ import annotations

import socket
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, Generator, Optional


def create_httpx_client(timeout: int = 120):
    import httpx

    socket_options = [(socket.SOL_SOCKET, socket.SO_KEEPALIVE, 1)]
    if hasattr(socket, "TCP_KEEPIDLE"):
        socket_options.append((socket.IPPROTO_TCP, socket.TCP_KEEPIDLE, 30))
    elif hasattr(socket, "TCP_KEEPALIVE"):
        socket_options.append((socket.IPPROTO_TCP, socket.TCP_KEEPALIVE, 30))
    if hasattr(socket, "TCP_KEEPINTVL"):
        socket_options.append((socket.IPPROTO_TCP, socket.TCP_KEEPINTVL, 30))
    if hasattr(socket, "TCP_KEEPCNT"):
        socket_options.append((socket.IPPROTO_TCP, socket.TCP_KEEPCNT, 3))
    return httpx.Client(
        transport=httpx.HTTPTransport(socket_options=socket_options),
        timeout=timeout,
    )


@dataclass
class ModelResponse:
    success: bool
    content: str
    model: str
    raw_response: Optional[Dict[str, Any]] = None
    error_message: Optional[str] = None
    usage: Optional[Dict[str, Any]] = None


class BaseModelClient(ABC):
    provider_name: str = "base"

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        model: str = "default",
        timeout: int = 120,
    ):
        self.api_key = api_key
        self.base_url = base_url
        self.model = model
        self.timeout = timeout
        self._validate_config()

    @abstractmethod
    def _validate_config(self) -> None:
        """Validate client configuration."""

    @abstractmethod
    def chat(
        self,
        message: str,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        stream: bool = False,
        **kwargs,
    ) -> ModelResponse:
        """Send a single-turn chat request."""

    def chat_messages(
        self,
        messages: list,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        **kwargs,
    ) -> ModelResponse:
        parts = []
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            if role == "system":
                parts.append(f"[System] {content}")
            elif role == "assistant":
                parts.append(f"[Assistant] {content}")
            else:
                parts.append(content)
        return self.chat(
            "\n\n".join(parts),
            temperature=temperature,
            max_tokens=max_tokens,
            **kwargs,
        )

    def chat_stream(
        self,
        message: str,
        temperature: float = 0.7,
        **kwargs,
    ) -> Generator[str, None, None]:
        response = self.chat(message, temperature=temperature, stream=False, **kwargs)
        if response.success:
            yield response.content
        else:
            yield f"[错误: {response.error_message}]"
