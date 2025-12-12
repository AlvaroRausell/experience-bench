from __future__ import annotations

import os
from typing import Any

import httpx

from experience_bench.adapters.types import CompletionResult


class OllamaAdapter:
    def __init__(self) -> None:
        self.base_url = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")

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
        url = f"{self.base_url}/api/generate"

        # Ollama doesn't have a native 'system' field for /generate; embed it.
        prompt = f"{system}\n\n{user}".strip() + "\n"

        payload: dict[str, Any] = {
            "model": model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": temperature,
                "num_predict": max_output_tokens,
            },
        }

        with httpx.Client(timeout=timeout_s) as client:
            r = client.post(url, json=payload)
            r.raise_for_status()
            data = r.json()

        text = str(data.get("response", "") or "")

        # Ollama reports counts as prompt_eval_count/eval_count when available.
        raw_usage = {
            "prompt_eval_count": data.get("prompt_eval_count"),
            "eval_count": data.get("eval_count"),
        }
        usage_derived = {
            "input_tokens": data.get("prompt_eval_count"),
            "output_tokens": data.get("eval_count"),
            "total_tokens": None,
        }

        return CompletionResult(text=text, raw_usage=raw_usage, usage_derived=usage_derived)
