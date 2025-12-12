from __future__ import annotations

import os
import random
import time
from dataclasses import dataclass
from typing import Any
from urllib.parse import urlsplit, urlunsplit

from azure.ai.inference import ChatCompletionsClient
from azure.ai.inference.models import SystemMessage, UserMessage
from azure.core.credentials import AzureKeyCredential
from azure.core.exceptions import HttpResponseError

from experience_bench.adapters.types import CompletionResult


class AzureOpenAIResponsesAdapter:
    def __init__(self) -> None:
        self.endpoint = _normalize_azure_endpoint(
            os.environ.get("AZURE_OPENAI_ENDPOINT"))
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
            raise RuntimeError(
                "Azure deployment is not set (AZURE_OPENAI_DEPLOYMENT or azureopenai:<deployment>)"
            )

        # azure-ai-inference expects the AOAI endpoint in the form:
        #   https://<resource>.openai.azure.com/openai/deployments/<deployment>
        aoai_endpoint = _azure_openai_deployment_endpoint(
            self.endpoint, deployment)
        client = ChatCompletionsClient(
            endpoint=aoai_endpoint,
            credential=AzureKeyCredential(self.api_key),
            api_version=self.api_version,
        )

        try:
            resp = _chat_complete_with_429_retries(
                client,
                messages=[
                    SystemMessage(content=system),
                    UserMessage(content=user),
                ],
                temperature=temperature,
                max_tokens=max_output_tokens,
                timeout_s=timeout_s,
            )
        finally:
            try:
                client.close()
            except Exception:
                pass

        text = ""
        try:
            if resp.choices and resp.choices[0].message and resp.choices[0].message.content:
                text = resp.choices[0].message.content
        except Exception:
            text = ""

        raw_usage = _usage_to_dict(getattr(resp, "usage", None))
        usage_derived = None
        if raw_usage:
            usage_derived = {
                "input_tokens": raw_usage.get("prompt_tokens") or raw_usage.get("input_tokens"),
                "output_tokens": raw_usage.get("completion_tokens") or raw_usage.get("output_tokens"),
                "total_tokens": raw_usage.get("total_tokens"),
            }

        return CompletionResult(text=text or "", raw_usage=raw_usage, usage_derived=usage_derived)


@dataclass(frozen=True)
class _Retry429Cfg:
    max_attempts: int
    base_delay_s: float
    max_delay_s: float


def _retry_429_cfg_from_env() -> _Retry429Cfg:
    return _Retry429Cfg(
        max_attempts=int(os.environ.get(
            "EXPERIENCE_BENCH_RETRY_429_MAX_ATTEMPTS", "5")),
        base_delay_s=float(os.environ.get(
            "EXPERIENCE_BENCH_RETRY_429_BASE_DELAY_S", "1.0")),
        max_delay_s=float(os.environ.get(
            "EXPERIENCE_BENCH_RETRY_429_MAX_DELAY_S", "30.0")),
    )


def _chat_complete_with_429_retries(
    client: ChatCompletionsClient,
    *,
    messages: list[Any],
    temperature: float,
    max_tokens: int,
    timeout_s: float,
):
    cfg = _retry_429_cfg_from_env()
    attempt = 0
    while True:
        try:
            return client.complete(
                messages=messages,
            )
        except HttpResponseError as e:
            status = getattr(e, "status_code", None)
            if status != 429:
                raise

            attempt += 1
            if attempt >= cfg.max_attempts:
                raise

            _sleep_for_retry(attempt=attempt, cfg=cfg, err=e)


def _sleep_for_retry(*, attempt: int, cfg: _Retry429Cfg, err: Exception) -> None:
    retry_after = _retry_after_seconds_from_error(err)
    if retry_after is None:
        retry_after = min(
            cfg.max_delay_s, cfg.base_delay_s * (2 ** (attempt - 1)))
    else:
        retry_after = min(cfg.max_delay_s, max(0.0, float(retry_after)))

    jitter = random.uniform(0.0, min(1.0, 0.25 * retry_after))
    time.sleep(retry_after + jitter)


def _retry_after_seconds_from_error(err: Exception) -> float | None:
    resp = getattr(err, "response", None)
    headers = getattr(resp, "headers", None) if resp is not None else None
    if not headers:
        return None

    value = headers.get("retry-after") or headers.get("Retry-After")
    if value is None:
        return None

    try:
        v = str(value).strip()
        if v.isdigit():
            return float(int(v))
    except Exception:
        return None
    return None


def _normalize_azure_endpoint(endpoint: str | None) -> str | None:
    """Normalize Azure OpenAI resource endpoint.

    Accepts values like:
    - https://<resource>.openai.azure.com/
    - https://<resource>.openai.azure.com/openai/
    - https://<resource>.openai.azure.com/openai/v1/

    Returns the base resource endpoint (without any `/openai...` path).
    """

    if not endpoint:
        return None

    e = endpoint.strip()
    if not e:
        return None

    try:
        parts = urlsplit(e)
        path = parts.path or ""
        idx = path.lower().find("/openai")
        if idx != -1:
            path = path[:idx]
        path = path.rstrip("/")
        return urlunsplit((parts.scheme, parts.netloc, path, "", ""))
    except Exception:
        return e.rstrip("/")


def _azure_openai_deployment_endpoint(resource_endpoint: str, deployment: str) -> str:
    base = resource_endpoint.rstrip("/")
    return f"{base}/openai/deployments/{deployment}"


def _usage_to_dict(usage: Any) -> dict[str, Any] | None:
    if usage is None:
        return None
    if isinstance(usage, dict):
        return usage
    if hasattr(usage, "model_dump"):
        try:
            return usage.model_dump()  # type: ignore[attr-defined]
        except Exception:
            return None
    if hasattr(usage, "__dict__"):
        try:
            return dict(usage.__dict__)
        except Exception:
            return None
    return None
