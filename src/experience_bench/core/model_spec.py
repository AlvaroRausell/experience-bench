from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ModelSpec:
    provider: str
    model: str

    @property
    def model_spec(self) -> str:
        return f"{self.provider}:{self.model}"

    @property
    def model_key(self) -> str:
        # Token aggregation scope: per model_key.
        # For Azure, caller should provide `azureopenai:<deployment>` already.
        return self.model_spec


def parse_model_spec(text: str) -> ModelSpec:
    if ":" not in text:
        raise ValueError(
            "Model spec must be 'provider:model', e.g. openrouter:openai/gpt-oss-20b:free"
        )
    provider, model = text.split(":", 1)
    provider = provider.strip().lower()
    model = model.strip()
    if provider not in {"openrouter", "ollama", "azureopenai"}:
        raise ValueError(f"Unknown provider: {provider}")
    if not model:
        raise ValueError("Model name is required")
    return ModelSpec(provider=provider, model=model)
