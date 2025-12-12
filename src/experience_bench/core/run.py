from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any

from rich.console import Console
from tqdm import tqdm

from experience_bench.adapters.azureopenai import AzureOpenAIResponsesAdapter
from experience_bench.adapters.ollama import OllamaAdapter
from experience_bench.adapters.openrouter import OpenRouterAdapter
from experience_bench.core.code_exec import run_python_code
from experience_bench.core.code_extract import extract_first_code_block
from experience_bench.core.config import load_benchmark_spec
from experience_bench.core.eval import eval_two_line_stdout
from experience_bench.core.model_spec import parse_model_spec
from experience_bench.core.prompts import render_prompt
from experience_bench.core.records import TrialRecord
from experience_bench.core.storage import append_jsonl


def run_benchmark(
    *,
    benchmark_path: Path,
    out_jsonl_path: Path,
    output_dir: Path,
    models_override: str | None,
    years_override: str | None,
    runs_per_setting: int,
    warmup: int,
    timeout_s: float,
    max_output_tokens: int,
    temperature: float,
    concurrency: int,
    console: Console,
) -> dict:
    if concurrency < 1:
        raise ValueError("concurrency must be >= 1")

    spec = load_benchmark_spec(benchmark_path)

    years = (
        [int(x) for x in years_override.split(",") if x.strip()]
        if years_override
        else spec.years
    )
    models = (
        [x.strip() for x in models_override.split(",") if x.strip()]
        if models_override
        else spec.models
    )
    if not models:
        raise ValueError("No models configured. Set models in YAML or pass --models")

    statement = spec.problem.statement_path.read_text(encoding="utf-8")
    input_payload = spec.problem.input_path.read_text(encoding="utf-8")
    prompt_template_path = spec.prompt_template_path

    adapters = {
        "openrouter": OpenRouterAdapter(),
        "ollama": OllamaAdapter(),
        "azureopenai": AzureOpenAIResponsesAdapter(),
    }

    # Date-based run id for human-friendly `.output/<run_id>/...` folder names.
    # Example: 20251212_142657_123456Z
    run_id = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%fZ")
    records: list[TrialRecord] = []

    total_steps = len(models) * len(years) * (max(0, warmup) + runs_per_setting)
    pbar = tqdm(total=total_steps, unit="req", desc="Benchmark", dynamic_ncols=True)

    output_root = output_dir / run_id
    output_root.mkdir(parents=True, exist_ok=True)

    def _run_warmup_trial(*, ms, y: int) -> None:
        pr = render_prompt(
            template_path=prompt_template_path,
            years_experience=y,
            problem_statement=statement,
        )
        adapter = adapters[ms.provider]
        adapter.complete(
            model=ms.model,
            system=pr.system,
            user=pr.user,
            max_output_tokens=max_output_tokens,
            temperature=temperature,
            timeout_s=timeout_s,
        )

    def _run_measured_trial(*, ms, y: int, i: int) -> TrialRecord:
        pr = render_prompt(
            template_path=prompt_template_path,
            years_experience=y,
            problem_statement=statement,
        )

        ts = datetime.now(timezone.utc).isoformat()
        t0 = time.perf_counter()
        resp_text: str | None = None
        code: str | None = None
        exec_stdout: str | None = None
        exec_stderr: str | None = None

        try:
            adapter = adapters[ms.provider]
            resp = adapter.complete(
                model=ms.model,
                system=pr.system,
                user=pr.user,
                max_output_tokens=max_output_tokens,
                temperature=temperature,
                timeout_s=timeout_s,
            )
            t1 = time.perf_counter()
            ttlt_ms = (t1 - t0) * 1000.0
            resp_text = resp.text

            code = extract_first_code_block(resp.text)
            if not code:
                rec = TrialRecord(
                    run_id=run_id,
                    timestamp_utc=ts,
                    benchmark_id=spec.benchmark_id,
                    years=y,
                    run_index=i,
                    provider=ms.provider,
                    model_spec=ms.model_spec,
                    model_key=ms.model_key,
                    prompt_rendered_sha256=pr.rendered_sha256,
                    status="error",
                    error_type="parse_error",
                    error_message="No fenced code block found",
                    ttlt_ms=ttlt_ms,
                    exec_ms=None,
                    passed_a=None,
                    passed_b=None,
                    passed_all=None,
                    output_a=None,
                    output_b=None,
                    expected_a=spec.expected.part_a,
                    expected_b=spec.expected.part_b,
                    raw_usage=resp.raw_usage,
                    usage_derived=resp.usage_derived,
                    response_text_len=len(resp.text or ""),
                    extracted_code_len=None,
                )
                _write_artifacts(
                    output_root=output_root,
                    provider=ms.provider,
                    model_key=ms.model_key,
                    years=y,
                    run_index=i,
                    prompt_system=pr.system,
                    prompt_user=pr.user,
                    completion_text=resp_text,
                    code_text=None,
                    stdout_text=None,
                    stderr_text=None,
                    record=rec,
                )
                return rec

            exec_res = run_python_code(
                code=code,
                stdin_text=input_payload,
                timeout_s=timeout_s,
            )
            exec_stdout = exec_res.stdout
            exec_stderr = exec_res.stderr

            if not exec_res.ok:
                rec = TrialRecord(
                    run_id=run_id,
                    timestamp_utc=ts,
                    benchmark_id=spec.benchmark_id,
                    years=y,
                    run_index=i,
                    provider=ms.provider,
                    model_spec=ms.model_spec,
                    model_key=ms.model_key,
                    prompt_rendered_sha256=pr.rendered_sha256,
                    status="error",
                    error_type=exec_res.error_type,
                    error_message=exec_res.stderr.strip()[:800] or None,
                    ttlt_ms=ttlt_ms,
                    exec_ms=exec_res.exec_ms,
                    passed_a=False,
                    passed_b=False,
                    passed_all=False,
                    output_a=None,
                    output_b=None,
                    expected_a=spec.expected.part_a,
                    expected_b=spec.expected.part_b,
                    raw_usage=resp.raw_usage,
                    usage_derived=resp.usage_derived,
                    response_text_len=len(resp.text or ""),
                    extracted_code_len=len(code),
                )
                _write_artifacts(
                    output_root=output_root,
                    provider=ms.provider,
                    model_key=ms.model_key,
                    years=y,
                    run_index=i,
                    prompt_system=pr.system,
                    prompt_user=pr.user,
                    completion_text=resp_text,
                    code_text=code,
                    stdout_text=exec_stdout,
                    stderr_text=exec_stderr,
                    record=rec,
                )
                return rec

            ev = eval_two_line_stdout(
                stdout=exec_stdout or "",
                expected_a=spec.expected.part_a,
                expected_b=spec.expected.part_b,
            )

            status = "ok" if ev.passed_all else "error"
            err_type = ev.error_type if not ev.passed_all else None
            err_msg = ev.error_message if not ev.passed_all else None

            rec = TrialRecord(
                run_id=run_id,
                timestamp_utc=ts,
                benchmark_id=spec.benchmark_id,
                years=y,
                run_index=i,
                provider=ms.provider,
                model_spec=ms.model_spec,
                model_key=ms.model_key,
                prompt_rendered_sha256=pr.rendered_sha256,
                status=status,
                error_type=err_type,
                error_message=err_msg,
                ttlt_ms=ttlt_ms,
                exec_ms=exec_res.exec_ms,
                passed_a=ev.passed_a,
                passed_b=ev.passed_b,
                passed_all=ev.passed_all,
                output_a=ev.output_a,
                output_b=ev.output_b,
                expected_a=spec.expected.part_a,
                expected_b=spec.expected.part_b,
                raw_usage=resp.raw_usage,
                usage_derived=resp.usage_derived,
                response_text_len=len(resp.text or ""),
                extracted_code_len=len(code),
            )
            _write_artifacts(
                output_root=output_root,
                provider=ms.provider,
                model_key=ms.model_key,
                years=y,
                run_index=i,
                prompt_system=pr.system,
                prompt_user=pr.user,
                completion_text=resp_text,
                code_text=code,
                stdout_text=exec_stdout,
                stderr_text=exec_stderr,
                record=rec,
            )
            return rec

        except Exception as e:
            t1 = time.perf_counter()
            ttlt_ms = (t1 - t0) * 1000.0
            rec = TrialRecord(
                run_id=run_id,
                timestamp_utc=ts,
                benchmark_id=spec.benchmark_id,
                years=y,
                run_index=i,
                provider=ms.provider,
                model_spec=ms.model_spec,
                model_key=ms.model_key,
                prompt_rendered_sha256=pr.rendered_sha256,
                status="error",
                error_type="provider_error",
                error_message=str(e)[:800],
                ttlt_ms=ttlt_ms,
                exec_ms=None,
                passed_a=None,
                passed_b=None,
                passed_all=None,
                output_a=None,
                output_b=None,
                expected_a=spec.expected.part_a,
                expected_b=spec.expected.part_b,
                raw_usage=None,
                usage_derived=None,
                response_text_len=len(resp_text or "") if resp_text is not None else None,
                extracted_code_len=len(code) if code is not None else None,
            )
            _write_artifacts(
                output_root=output_root,
                provider=ms.provider,
                model_key=ms.model_key,
                years=y,
                run_index=i,
                prompt_system=pr.system,
                prompt_user=pr.user,
                completion_text=resp_text,
                code_text=code,
                stdout_text=exec_stdout,
                stderr_text=exec_stderr,
                record=rec,
            )
            return rec

    try:
        # Warmup (not recorded)
        warmup_tasks: list[tuple[Any, int]] = []
        for model_text in models:
            ms = parse_model_spec(model_text)
            for y in years:
                for _w in range(max(0, warmup)):
                    warmup_tasks.append((ms, y))

        if warmup_tasks:
            pbar.set_postfix_str(f"warmup ({len(warmup_tasks)} calls, concurrency={concurrency})")
            if concurrency == 1:
                for ms, y in warmup_tasks:
                    try:
                        _run_warmup_trial(ms=ms, y=y)
                    except Exception:
                        pass
                    finally:
                        pbar.update(1)
            else:
                with ThreadPoolExecutor(max_workers=concurrency) as ex:
                    futures = [ex.submit(_run_warmup_trial, ms=ms, y=y) for (ms, y) in warmup_tasks]
                    for fut in as_completed(futures):
                        try:
                            fut.result()
                        except Exception:
                            pass
                        finally:
                            pbar.update(1)

        # Measured runs can be parallelized.
        tasks: list[tuple[Any, int, int]] = []
        for model_text in models:
            ms = parse_model_spec(model_text)
            for y in years:
                for i in range(runs_per_setting):
                    tasks.append((ms, y, i))

        pbar.set_postfix_str(f"running measured ({len(tasks)} tasks, concurrency={concurrency})")
        if concurrency == 1:
            for ms, y, i in tasks:
                rec = _run_measured_trial(ms=ms, y=y, i=i)
                records.append(rec)
                pbar.update(1)
        else:
            with ThreadPoolExecutor(max_workers=concurrency) as ex:
                future_to_task = {
                    ex.submit(_run_measured_trial, ms=ms, y=y, i=i): (ms, y, i)
                    for (ms, y, i) in tasks
                }
                for fut in as_completed(future_to_task):
                    ms, y, i = future_to_task[fut]
                    try:
                        rec = fut.result()
                    except Exception as e:
                        # Should be rare since _run_measured_trial catches most errors,
                        # but keep the runner robust.
                        ts = datetime.now(timezone.utc).isoformat()
                        rec = TrialRecord(
                            run_id=run_id,
                            timestamp_utc=ts,
                            benchmark_id=spec.benchmark_id,
                            years=y,
                            run_index=i,
                            provider=ms.provider,
                            model_spec=ms.model_spec,
                            model_key=ms.model_key,
                            prompt_rendered_sha256="",
                            status="error",
                            error_type="runner_error",
                            error_message=str(e)[:800],
                            ttlt_ms=None,
                            exec_ms=None,
                            passed_a=None,
                            passed_b=None,
                            passed_all=None,
                            output_a=None,
                            output_b=None,
                            expected_a=spec.expected.part_a,
                            expected_b=spec.expected.part_b,
                            raw_usage=None,
                            usage_derived=None,
                            response_text_len=None,
                            extracted_code_len=None,
                        )
                    records.append(rec)
                    pbar.update(1)

    finally:
        pbar.close()

    written = append_jsonl(out_jsonl_path, records)
    return {"run_id": run_id, "records_written": written}


def _write_artifacts(
    *,
    output_root: Path,
    provider: str,
    model_key: str,
    years: int,
    run_index: int,
    prompt_system: str,
    prompt_user: str,
    completion_text: str | None,
    code_text: str | None,
    stdout_text: str | None,
    stderr_text: str | None,
    record: TrialRecord,
) -> None:
    safe_model = _safe_path_component(model_key)
    out_dir = output_root / provider / safe_model / f"years_{years}" / f"run_{run_index:03d}"
    out_dir.mkdir(parents=True, exist_ok=True)

    (out_dir / "prompt_system.txt").write_text(prompt_system or "", encoding="utf-8")
    (out_dir / "prompt_user.txt").write_text(prompt_user or "", encoding="utf-8")
    if completion_text is not None:
        (out_dir / "completion.txt").write_text(completion_text, encoding="utf-8")
    if code_text is not None:
        (out_dir / "solution.py").write_text(code_text, encoding="utf-8")
    if stdout_text is not None:
        (out_dir / "exec_stdout.txt").write_text(stdout_text, encoding="utf-8")
    if stderr_text is not None:
        (out_dir / "exec_stderr.txt").write_text(stderr_text, encoding="utf-8")

    (out_dir / "record.json").write_text(
        json.dumps(record.__dict__, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def _safe_path_component(text: str) -> str:
    cleaned = []
    for ch in text:
        if ch.isalnum() or ch in {"-", "_", "."}:
            cleaned.append(ch)
        else:
            cleaned.append("_")
    s = "".join(cleaned).strip("._")
    return s or "model"
