from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class TrialRecord:
    run_id: str
    timestamp_utc: str

    benchmark_id: str
    years: int
    run_index: int

    provider: str
    model_spec: str
    model_key: str

    prompt_rendered_sha256: str

    status: str  # ok|error
    error_type: str | None
    error_message: str | None

    ttlt_ms: float | None
    exec_ms: float | None

    passed_a: bool | None
    passed_b: bool | None
    passed_all: bool | None

    output_a: str | None
    output_b: str | None

    expected_a: str
    expected_b: str

    # Token usage is model-scoped; do not compare across different model_key values.
    raw_usage: dict[str, Any] | None
    usage_derived: dict[str, Any] | None

    response_text_len: int | None
    extracted_code_len: int | None


def trial_record_to_json(rec: TrialRecord) -> dict[str, Any]:
    return rec.__dict__
