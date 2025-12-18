from __future__ import annotations

import pytest

from experience_bench.core.model_spec import ModelSpec, parse_model_spec


def test_parse_model_spec_openrouter() -> None:
    result = parse_model_spec("openrouter:openai/gpt-4o")
    assert result.provider == "openrouter"
    assert result.model == "openai/gpt-4o"
    assert result.model_spec == "openrouter:openai/gpt-4o"


def test_parse_model_spec_openrouter_with_free_suffix() -> None:
    result = parse_model_spec("openrouter:mistralai/devstral-2512:free")
    assert result.provider == "openrouter"
    assert result.model == "mistralai/devstral-2512:free"


def test_parse_model_spec_openrouter_uppercase() -> None:
    result = parse_model_spec("OPENROUTER:openai/gpt-4o")
    assert result.provider == "openrouter"
    assert result.model == "openai/gpt-4o"


def test_parse_model_spec_ollama() -> None:
    result = parse_model_spec("ollama:granite4:3b")
    assert result.provider == "ollama"
    assert result.model == "granite4:3b"


def test_parse_model_spec_azureopenai() -> None:
    result = parse_model_spec("azureopenai:my-deployment")
    assert result.provider == "azureopenai"
    assert result.model == "my-deployment"


def test_parse_model_spec_no_colon_raises() -> None:
    with pytest.raises(ValueError, match="provider:model"):
        parse_model_spec("openrouter-openai/gpt-4o")


def test_parse_model_spec_unknown_provider_raises() -> None:
    with pytest.raises(ValueError, match="Unknown provider"):
        parse_model_spec("unknown:model-name")


def test_parse_model_spec_empty_model_raises() -> None:
    with pytest.raises(ValueError, match="Model name is required"):
        parse_model_spec("openrouter:")


def test_model_spec_model_key() -> None:
    spec = ModelSpec(provider="openrouter", model="openai/gpt-4o")
    assert spec.model_key == "openrouter:openai/gpt-4o"
