from __future__ import annotations

import argparse
import json
from pathlib import Path

from rich.console import Console

from experience_bench.core.dotenv import load_dotenv_if_present
from experience_bench.core.run import run_benchmark
from experience_bench.reporting.html_report import build_html_report


def _cmd_run(args: argparse.Namespace) -> int:
    console = Console()
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    result = run_benchmark(
        benchmark_path=Path(args.benchmark),
        out_jsonl_path=out_path,
        output_dir=Path(args.output_dir),
        models_override=args.models,
        years_override=args.years,
        runs_per_setting=args.runs_per_setting,
        warmup=args.warmup,
        timeout_s=args.timeout_s,
        max_output_tokens=args.max_output_tokens,
        temperature=args.temperature,
        concurrency=args.concurrency,
        console=console,
    )

    console.print(f"Wrote {result['records_written']} records to {out_path}")
    return 0


def _cmd_report(args: argparse.Namespace) -> int:
    console = Console()
    in_path = Path(args.input)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    report = build_html_report(
        jsonl_path=in_path,
        out_html_path=out_path,
        console=console,
    )

    console.print(
        f"Wrote report to {out_path} (records={report['n_records']}, groups={report['n_groups']})"
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    # Load .env early so provider API keys (e.g., OPENROUTER_API_KEY) work
    # without requiring users to export them in the shell.
    load_dotenv_if_present()

    parser = argparse.ArgumentParser(prog="experience-bench")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_run = sub.add_parser("run", help="Run benchmark and write JSONL results")
    p_run.add_argument("--benchmark", required=True, help="Path to benchmark YAML")
    p_run.add_argument("--out", required=True, help="Output JSONL path")
    p_run.add_argument(
        "--output-dir",
        default=".output",
        help="Directory to write per-trial artifacts (completion/code/stdout/stderr)",
    )
    p_run.add_argument(
        "--models",
        default=None,
        help="Comma-separated provider:model specs; overrides benchmark config",
    )
    p_run.add_argument(
        "--years",
        default=None,
        help="Comma-separated years of experience; overrides benchmark config",
    )
    p_run.add_argument("--runs-per-setting", type=int, default=2)
    p_run.add_argument("--warmup", type=int, default=1)
    p_run.add_argument("--timeout-s", type=float, default=120.0)
    p_run.add_argument("--max-output-tokens", type=int, default=1024)
    p_run.add_argument("--temperature", type=float, default=0.0)
    p_run.add_argument("--concurrency", type=int, default=1)
    p_run.set_defaults(func=_cmd_run)

    p_report = sub.add_parser("report", help="Generate an interactive HTML report")
    p_report.add_argument("--in", dest="input", required=True, help="Input JSONL path")
    p_report.add_argument("--out", required=True, help="Output HTML path")
    p_report.set_defaults(func=_cmd_report)

    args = parser.parse_args(argv)
    return int(args.func(args))
