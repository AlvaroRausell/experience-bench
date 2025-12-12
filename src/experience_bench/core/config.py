from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class BenchmarkExpected:
    part_a: str
    part_b: str


@dataclass(frozen=True)
class BenchmarkProblem:
    statement_path: Path
    input_path: Path


@dataclass(frozen=True)
class BenchmarkSpec:
    benchmark_id: str
    prompt_template_path: Path
    years: list[int]
    models: list[str]
    runs_per_setting: int
    warmup: int
    timeout_s: float
    max_output_tokens: int
    temperature: float
    problem: BenchmarkProblem
    expected: BenchmarkExpected


def load_benchmark_spec(path: Path) -> BenchmarkSpec:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("Benchmark file must be a YAML mapping")

    benchmark_id = str(raw.get("id") or path.stem)
    prompt_template_path = (path.parent / str(raw.get("prompt_template", ""))).resolve()
    if not prompt_template_path.exists():
        raise FileNotFoundError(f"prompt_template not found: {prompt_template_path}")

    years = _parse_int_list(raw.get("years"), default=[1, 5, 10, 25])
    models = _parse_str_list(raw.get("models"), default=[])

    defaults = raw.get("defaults") or {}
    runs_per_setting = int(defaults.get("runs_per_setting", 2))
    warmup = int(defaults.get("warmup", 1))
    timeout_s = float(defaults.get("timeout_s", 120.0))
    max_output_tokens = int(defaults.get("max_output_tokens", 1024))
    temperature = float(defaults.get("temperature", 0.0))

    problem_raw = raw.get("problem") or {}
    statement_path = (path.parent / str(problem_raw.get("statement_path", ""))).resolve()
    input_path = (path.parent / str(problem_raw.get("input_path", ""))).resolve()
    if not statement_path.exists():
        raise FileNotFoundError(f"problem.statement_path not found: {statement_path}")
    if not input_path.exists():
        raise FileNotFoundError(f"problem.input_path not found: {input_path}")

    expected_raw = raw.get("expected") or {}
    parts = expected_raw.get("parts") or {}
    part_a = str((parts.get("a") or {}).get("value", "")).strip()
    part_b = str((parts.get("b") or {}).get("value", "")).strip()
    if not part_a or not part_b:
        raise ValueError("expected.parts.a.value and expected.parts.b.value are required")

    return BenchmarkSpec(
        benchmark_id=benchmark_id,
        prompt_template_path=prompt_template_path,
        years=years,
        models=models,
        runs_per_setting=runs_per_setting,
        warmup=warmup,
        timeout_s=timeout_s,
        max_output_tokens=max_output_tokens,
        temperature=temperature,
        problem=BenchmarkProblem(statement_path=statement_path, input_path=input_path),
        expected=BenchmarkExpected(part_a=part_a, part_b=part_b),
    )


def _parse_int_list(value: Any, default: list[int]) -> list[int]:
    if value is None:
        return list(default)
    if isinstance(value, str):
        items = [x.strip() for x in value.split(",") if x.strip()]
        return [int(x) for x in items]
    if isinstance(value, list):
        return [int(x) for x in value]
    raise ValueError("Expected years to be a list or comma-separated string")


def _parse_str_list(value: Any, default: list[str]) -> list[str]:
    if value is None:
        return list(default)
    if isinstance(value, str):
        return [x.strip() for x in value.split(",") if x.strip()]
    if isinstance(value, list):
        return [str(x) for x in value]
    raise ValueError("Expected models to be a list or comma-separated string")
