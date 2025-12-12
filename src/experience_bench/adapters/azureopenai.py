from __future__ import annotations

import os
from typing import Any

import httpx

from experience_bench.adapters.retry import post_with_429_retries
from experience_bench.adapters.types import CompletionResult


class AzureOpenAIResponsesAdapter:
    def __init__(self) -> None:
        self.endpoint = os.environ.get("AZURE_OPENAI_ENDPOINT")
        self.api_key = os.environ.get("AZURE_OPENAI_API_KEY")
        self.api_version = os.environ.get("AZURE_OPENAI_API_VERSION")
        self.deployment = os.environ.get("AZURE_OPENAI_DEPLOYMENT")

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
        # For Azure, `model` is the deployment name from the model spec.
        deployment = model or self.deployment
        if not self.endpoint:
            raise RuntimeError("AZURE_OPENAI_ENDPOINT is not set")
        if not self.api_key:
            raise RuntimeError("AZURE_OPENAI_API_KEY is not set")
        if not self.api_version:
            raise RuntimeError("AZURE_OPENAI_API_VERSION is not set")
        if not deployment:
            raise RuntimeError("Azure deployment is not set (AZURE_OPENAI_DEPLOYMENT or azureopenai:<deployment>)")

        url = (
            f"{self.endpoint.rstrip('/')}/openai/deployments/{deployment}/responses"
            f"?api-version={self.api_version}"
        )
        headers = {"api-key": self.api_key}

        payload: dict[str, Any] = {
            "input": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "max_output_tokens": max_output_tokens,
            "temperature": temperature,
            "stream": False,
        }

        with httpx.Client(timeout=timeout_s) as client:
            r = post_with_429_retries(client, url, headers=headers, json=payload, timeout_s=timeout_s)
            r.raise_for_status()
            data = r.json()

        text = _extract_text_from_responses(data)

        raw_usage = data.get("usage") if isinstance(data.get("usage"), dict) else None
        usage_derived = None
        if raw_usage:
            usage_derived = {
                "input_tokens": raw_usage.get("input_tokens"),
                "output_tokens": raw_usage.get("output_tokens"),
                "total_tokens": raw_usage.get("total_tokens"),
            }

        return CompletionResult(text=text, raw_usage=raw_usage, usage_derived=usage_derived)


def _extract_text_from_responses(data: dict[str, Any]) -> str:
    # Try common shapes in OpenAI/Azure Responses API.
    if isinstance(data.get("output_text"), str):
        return data.get("output_text") or ""

    output = data.get("output")
    if isinstance(output, list):
        chunks: list[str] = []
        for item in output:
            if not isinstance(item, dict):
                continue
            content = item.get("content")
            if isinstance(content, list):
                for c in content:
                    if not isinstance(c, dict):
                        continue
                    if c.get("type") == "output_text" and isinstance(c.get("text"), str):
                        chunks.append(c.get("text") or "")
                    if c.get("type") == "text" and isinstance(c.get("text"), str):
                        chunks.append(c.get("text") or "")
            # Some variants embed message-like data
            if isinstance(item.get("text"), str):
                chunks.append(item.get("text") or "")
        return "".join(chunks)

    # Fallback: look for chat-completions-like shape (in case of gateway compatibility)
    try:
        choices = data.get("choices") or []
        if choices and isinstance(choices[0], dict):
            msg = choices[0].get("message") or {}
            if isinstance(msg, dict) and isinstance(msg.get("content"), str):
                return msg.get("content") or ""
    except Exception:
        pass

    return ""
