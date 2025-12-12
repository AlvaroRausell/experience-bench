from __future__ import annotations


from datetime import datetime, timezone

from experience_bench.core.records import TrialRecord
from experience_bench.core.run import _safe_path_component, _write_artifacts


def test_safe_path_component_sanitizes() -> None:
    assert _safe_path_component("openai/gpt-4o") == "openai_gpt-4o"
    assert _safe_path_component("a b c") == "a_b_c"
    assert _safe_path_component("///") == "model"


def test_write_artifacts_writes_files(tmp_path) -> None:
    ts = datetime.now(timezone.utc).isoformat()
    rec = TrialRecord(
        run_id="run123",
        timestamp_utc=ts,
        benchmark_id="bench",
        years=5,
        run_index=0,
        provider="ollama",
        model_spec="ollama:openai/gpt-4o",
        model_key="openai/gpt-4o",
        prompt_rendered_sha256="sha",
        status="ok",
        error_type=None,
        error_message=None,
        ttlt_ms=1.23,
        exec_ms=4.56,
        passed_a=True,
        passed_b=True,
        passed_all=True,
        output_a="1",
        output_b="2",
        expected_a="1",
        expected_b="2",
        raw_usage={"total_tokens": 3},
        usage_derived={"total_tokens": 3},
        response_text_len=10,
        extracted_code_len=20,
    )

    _write_artifacts(
        output_root=tmp_path,
        provider=rec.provider,
        model_key=rec.model_key,
        years=rec.years,
        run_index=rec.run_index,
        prompt_system="sys",
        prompt_user="user",
        completion_text="completion",
        code_text="print('hi')",
        stdout_text="1\n2\n",
        stderr_text="",
        record=rec,
    )

    out_dir = (
        tmp_path
        / "ollama"
        / "openai_gpt-4o"
        / "years_5"
        / "run_000"
    )
    assert out_dir.is_dir()

    assert (out_dir / "prompt_system.txt").read_text(encoding="utf-8") == "sys"
    assert (out_dir / "prompt_user.txt").read_text(encoding="utf-8") == "user"
    assert (out_dir / "completion.txt").read_text(encoding="utf-8") == "completion"
    assert (out_dir / "solution.py").read_text(encoding="utf-8") == "print('hi')"
    assert (out_dir / "exec_stdout.txt").read_text(encoding="utf-8") == "1\n2\n"
    assert (out_dir / "exec_stderr.txt").read_text(encoding="utf-8") == ""

    record_text = (out_dir / "record.json").read_text(encoding="utf-8")
    assert '"run_id": "run123"' in record_text
