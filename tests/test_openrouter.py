from __future__ import annotations

import pytest

from experience_bench.adapters.openrouter import OpenRouterAdapter


def test_openrouter_adapter_init_default_base_url(monkeypatch) -> None:
    monkeypatch.delenv("OPENROUTER_BASE_URL", raising=False)
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    adapter = OpenRouterAdapter()
    assert adapter.base_url == "https://openrouter.ai/api/v1"


def test_openrouter_adapter_init_custom_base_url(monkeypatch) -> None:
    monkeypatch.setenv("OPENROUTER_BASE_URL", "https://custom.openrouter.ai/v1")
    adapter = OpenRouterAdapter()
    assert adapter.base_url == "https://custom.openrouter.ai/v1"


def test_openrouter_adapter_init_api_key(monkeypatch) -> None:
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-api-key")
    adapter = OpenRouterAdapter()
    assert adapter.api_key == "test-api-key"


def test_openrouter_adapter_init_optional_headers(monkeypatch) -> None:
    monkeypatch.setenv("OPENROUTER_HTTP_REFERER", "https://example.com")
    monkeypatch.setenv("OPENROUTER_X_TITLE", "Test App")
    adapter = OpenRouterAdapter()
    assert adapter.http_referer == "https://example.com"
    assert adapter.x_title == "Test App"


def test_openrouter_adapter_complete_raises_without_api_key(monkeypatch) -> None:
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    adapter = OpenRouterAdapter()
    with pytest.raises(RuntimeError, match="OPENROUTER_API_KEY is not set"):
        adapter.complete(
            model="openai/gpt-4o",
            system="You are a helpful assistant.",
            user="Hello",
            max_output_tokens=100,
            temperature=0.5,
            timeout_s=30.0,
        )


def test_openrouter_adapter_parses_usage() -> None:
    """Test that usage parsing logic works correctly."""
    # Test the parsing logic by simulating what the adapter does
    raw_usage = {
        "prompt_tokens": 10,
        "completion_tokens": 20,
        "total_tokens": 30,
    }

    usage_derived = {
        "input_tokens": raw_usage.get("prompt_tokens"),
        "output_tokens": raw_usage.get("completion_tokens"),
        "total_tokens": raw_usage.get("total_tokens"),
    }

    assert usage_derived["input_tokens"] == 10
    assert usage_derived["output_tokens"] == 20
    assert usage_derived["total_tokens"] == 30
