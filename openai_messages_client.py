from __future__ import annotations

import json
import urllib.error
import urllib.request
from collections.abc import Sequence
from typing import Any


Message = dict[str, str]


def normalize_base_url(base_url: str) -> str:
    url = base_url.rstrip("/")
    if url.endswith("/v1"):
        return url
    return f"{url}/v1"


def build_messages(prompt: str, system: str | None = None) -> list[Message]:
    messages: list[Message] = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})
    return messages


def validate_messages(messages: Sequence[Message]) -> list[Message]:
    if not messages:
        raise ValueError("messages must contain at least one OpenAI-format message.")

    normalized: list[Message] = []
    for index, message in enumerate(messages):
        role = message.get("role")
        content = message.get("content")
        if role not in {"system", "user", "assistant", "tool"}:
            raise ValueError(f"messages[{index}].role is invalid: {role!r}")
        if not isinstance(content, str) or not content:
            raise ValueError(f"messages[{index}].content must be a non-empty string.")
        normalized.append({"role": role, "content": content})
    return normalized


def chat_completion(
    *,
    base_url: str,
    api_key: str,
    model: str,
    messages: Sequence[Message],
    temperature: float,
    max_tokens: int,
    timeout: int,
    extra_payload: dict[str, Any] | None = None,
) -> str:
    if not api_key:
        raise RuntimeError("Missing API key for OpenAI-compatible chat completion.")
    if not model:
        raise RuntimeError("Missing model name for OpenAI-compatible chat completion.")

    payload: dict[str, Any] = {
        "model": model,
        "messages": validate_messages(messages),
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    if extra_payload:
        payload.update(extra_payload)

    request = urllib.request.Request(
        f"{normalize_base_url(base_url)}/chat/completions",
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            data = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {exc.code}: {body[:1200]}") from exc

    return data["choices"][0]["message"]["content"]
