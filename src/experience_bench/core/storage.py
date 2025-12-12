from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

from experience_bench.core.records import TrialRecord, trial_record_to_json


def append_jsonl(path: Path, records: Iterable[TrialRecord]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with path.open("a", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(trial_record_to_json(rec), ensure_ascii=False))
            f.write("\n")
            n += 1
    return n


def read_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)
