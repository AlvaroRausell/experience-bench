from __future__ import annotations

import os
from typing import Any

import httpx

from experience_bench.adapters.retry import post_with_429_retries
from experience_bench.adapters.types import CompletionResult


class OpenRouterAdapter:
    def __init__(self) -> None:
        self.base_url = os.environ.get("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
        self.api_key = os.environ.get("OPENROUTER_API_KEY")
        self.http_referer = os.environ.get("OPENROUTER_HTTP_REFERER")
        self.x_title = os.environ.get("OPENROUTER_X_TITLE")

    def complete(
        self,
        *,
        model: str,
        system: str,
        user: str,
        max_output_tokens: int,
        temperature: float,
        timeout_s: float,
    ) -> CompletionResult:
        if not self.api_key:
            raise RuntimeError("OPENROUTER_API_KEY is not set")

        url = f"{self.base_url}/chat/completions"
        headers = {"Authorization": f"Bearer {self.api_key}"}
        if self.http_referer:
            headers["HTTP-Referer"] = self.http_referer
        if self.x_title:
            headers["X-Title"] = self.x_title

        payload: dict[str, Any] = {
            "model": model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "temperature": temperature,
            "max_tokens": max_output_tokens,
            "stream": False,
        }

        with httpx.Client(timeout=timeout_s) as client:
            r = post_with_429_retries(client, url, headers=headers, json=payload, timeout_s=timeout_s)
            r.raise_for_status()
            data = r.json()

        text = (
            (data.get("choices") or [{}])[0]
            .get("message", {})
            .get("content", "")
        )

        raw_usage = data.get("usage") if isinstance(data.get("usage"), dict) else None
        usage_derived = None
        if raw_usage:
            usage_derived = {
                "input_tokens": raw_usage.get("prompt_tokens"),
                "output_tokens": raw_usage.get("completion_tokens"),
                "total_tokens": raw_usage.get("total_tokens"),
            }

        return CompletionResult(text=text or "", raw_usage=raw_usage, usage_derived=usage_derived)
