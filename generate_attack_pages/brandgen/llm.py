from __future__ import annotations

import json
import re
import sys
import time
from typing import Any

from .models import LLMStats


class LLMError(RuntimeError):
    pass


_REFERENCE_EXCERPT_RE = re.compile(
    r'("reference_excerpt"\s*:\s*)"(?:\\.|[^"\\])*"'
)


def _redact_reference_excerpts(value: Any) -> Any:
    """Keep transient public excerpts out of debug output and redirected logs."""
    if isinstance(value, str):
        return _REFERENCE_EXCERPT_RE.sub(r'\1"<redacted-transient-reference>"', value)
    if isinstance(value, list):
        return [_redact_reference_excerpts(item) for item in value]
    if isinstance(value, dict):
        return {key: _redact_reference_excerpts(item) for key, item in value.items()}
    return value


def create_client(provider: str, model: str | None, timeout: int, enable_thinking: bool | None = None):
    if provider == "none":
        return None
    if provider == "openai-compatible":
        from clients.openai_compatible import OpenAICompatibleClient

        import os

        return OpenAICompatibleClient(
            model=model or os.environ.get("GENERATOR_MODEL", "your-generator-model"),
            timeout=timeout,
        )
    raise ValueError(f"unsupported llm provider: {provider}")


def client_label(client) -> str:
    if client is None:
        return "none"
    provider = getattr(client, "provider_name", "")
    model = getattr(client, "model", "")
    return "/".join(part for part in (provider, model) if part) or "unknown"


def _normalize_json_value(value: Any) -> Any:
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return value
        if stripped[0] in "{[":
            try:
                return _normalize_json_value(json.loads(stripped))
            except json.JSONDecodeError:
                return value
        return value
    if isinstance(value, list) and len(value) == 1 and isinstance(value[0], str):
        return _normalize_json_value(value[0])
    return value


def extract_json(text: str) -> Any:
    if isinstance(text, (dict, list)):
        return _normalize_json_value(text)
    text = (text or "").strip()
    if not text:
        raise ValueError("empty response")
    try:
        return _normalize_json_value(json.loads(text))
    except json.JSONDecodeError:
        pass

    fenced = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", text)
    if fenced:
        try:
            return _normalize_json_value(json.loads(fenced.group(1)))
        except json.JSONDecodeError:
            pass

    start_candidates = [idx for idx in (text.find("{"), text.find("[")) if idx >= 0]
    if not start_candidates:
        raise ValueError("response does not contain JSON")
    start = min(start_candidates)
    end = max(text.rfind("}"), text.rfind("]"))
    if end <= start:
        raise ValueError("response has incomplete JSON")
    return _normalize_json_value(json.loads(text[start : end + 1]))


def call_json(
    client,
    messages: list[dict[str, str]],
    *,
    context: str,
    temperature: float,
    max_tokens: int,
    max_attempts: int,
    stats: LLMStats,
    debug: bool = False,
) -> Any:
    if client is None:
        raise LLMError("LLM client is not configured")

    last_error = ""
    current_messages = list(messages)
    for attempt in range(1, max(1, max_attempts) + 1):
        stats.calls += 1
        if attempt > 1:
            stats.retries += 1
        started = time.monotonic()
        print(
            f"[info] LLM {context} attempt {attempt}/{max_attempts} "
            f"model={client_label(client)} max_tokens={max_tokens}",
            file=sys.stderr,
            flush=True,
        )
        if debug:
            print(
                f"[debug] LLM {context} request messages:\n"
                f"{json.dumps(_redact_reference_excerpts(current_messages), ensure_ascii=False, indent=2)}",
                file=sys.stderr,
                flush=True,
            )
        response = client.chat_messages(current_messages, temperature=temperature) #max_tokens=max_tokens
        # print(f"current_messages: {current_messages}")
        print(f"response: {response.content}")
        elapsed = time.monotonic() - started
        if not response.success:
            last_error = response.error_message or "unknown LLM error"
            print(f"\033[94m[warn]\033[0m LLM {context} failed in {elapsed:.1f}s: {last_error}", file=sys.stderr, flush=True)
        else:
            if debug:
                print(
                    f"[debug] LLM {context} raw response:\n{response.content}",
                    file=sys.stderr,
                    flush=True,
                )
            try:
                data = extract_json(response.content)
                if debug:
                    print(
                        f"[debug] LLM {context} parsed JSON:\n"
                        f"{json.dumps(data, ensure_ascii=False, indent=2)}",
                        file=sys.stderr,
                        flush=True,
                    )
                print(
                    f"[info] LLM {context} ok in {elapsed:.1f}s chars={len(response.content)}",
                    file=sys.stderr,
                    flush=True,
                )
                return data
            except Exception as err:
                last_error = f"invalid JSON: {err}"
                print(
                    f"\033[94m[warn]\033[0m LLM {context} returned invalid JSON in {elapsed:.1f}s: {err}",
                    file=sys.stderr,
                    flush=True,
                )

        current_messages = [
            *messages,
            {
                "role": "user",
                "content": (
                    "上一轮输出无法使用。请只输出严格合法 JSON，不要 Markdown、解释或额外字段。"
                    f"错误：{last_error}"
                ),
            },
        ]

    stats.failures += 1
    raise LLMError(f"{context} failed after {max_attempts} attempts: {last_error}")
