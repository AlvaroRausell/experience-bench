from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class EvalResult:
    passed_a: bool
    passed_b: bool
    passed_all: bool
    output_a: str | None
    output_b: str | None
    error_type: str | None
    error_message: str | None


def eval_two_line_stdout(*, stdout: str, expected_a: str, expected_b: str) -> EvalResult:
    text = stdout.strip("\n")
    lines = [ln.strip() for ln in text.splitlines() if ln.strip() != ""]
    if len(lines) < 2:
        return EvalResult(
            passed_a=False,
            passed_b=False,
            passed_all=False,
            output_a=lines[0] if len(lines) >= 1 else None,
            output_b=None,
            error_type="output_parse_error",
            error_message=f"Expected 2 non-empty lines, got {len(lines)}",
        )

    out_a, out_b = lines[0], lines[1]
    passed_a = out_a == expected_a
    passed_b = out_b == expected_b
    return EvalResult(
        passed_a=passed_a,
        passed_b=passed_b,
        passed_all=passed_a and passed_b,
        output_a=out_a,
        output_b=out_b,
        error_type=None,
        error_message=None,
    )
