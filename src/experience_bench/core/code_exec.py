from __future__ import annotations

import subprocess
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
import os


@dataclass(frozen=True)
class ExecResult:
    ok: bool
    stdout: str
    stderr: str
    exit_code: int
    exec_ms: float
    error_type: str | None


def run_python_code(*, code: str, stdin_text: str, timeout_s: float) -> ExecResult:
    with tempfile.TemporaryDirectory(prefix="experience-bench-") as td:
        workdir = Path(td)
        script_path = workdir / "solution.py"
        script_path.write_text(code, encoding="utf-8")

        cmd = ["python", str(script_path)]
        env = dict(os.environ)
        env["PYTHONNOUSERSITE"] = "1"

        t0 = time.perf_counter()
        try:
            proc = subprocess.run(
                cmd,
                input=stdin_text,
                text=True,
                capture_output=True,
                cwd=str(workdir),
                env=env,
                timeout=timeout_s,
            )
            t1 = time.perf_counter()
        except subprocess.TimeoutExpired as e:
            t1 = time.perf_counter()
            return ExecResult(
                ok=False,
                stdout=(e.stdout or ""),
                stderr=(e.stderr or ""),
                exit_code=124,
                exec_ms=(t1 - t0) * 1000.0,
                error_type="timeout",
            )

        ok = proc.returncode == 0
        error_type = None if ok else "runtime_error"
        return ExecResult(
            ok=ok,
            stdout=proc.stdout,
            stderr=proc.stderr,
            exit_code=proc.returncode,
            exec_ms=(t1 - t0) * 1000.0,
            error_type=error_type,
        )
