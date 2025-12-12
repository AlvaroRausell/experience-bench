from __future__ import annotations

import email.utils
import os
import random
import time
from datetime import datetime, timezone
from typing import Any

import httpx


def post_with_429_retries(
    client: httpx.Client,
    url: str,
    *,
    headers: dict[str, str] | None = None,
    json: dict[str, Any] | None = None,
    timeout_s: float | None = None,
) -> httpx.Response:
    """POST with retries on HTTP 429.

    Behavior:
    - Retries only on HTTP 429
    - Honors `Retry-After` when present (seconds or HTTP-date)
    - Otherwise uses exponential backoff with jitter

    Tuning via env vars:
    - EXPERIENCE_BENCH_RETRY_429_MAX_ATTEMPTS (default 5)
    - EXPERIENCE_BENCH_RETRY_429_BASE_DELAY_S (default 1.0)
    - EXPERIENCE_BENCH_RETRY_429_MAX_DELAY_S (default 30.0)
    """

    max_attempts = int(os.environ.get("EXPERIENCE_BENCH_RETRY_429_MAX_ATTEMPTS", "5"))
    base_delay_s = float(os.environ.get("EXPERIENCE_BENCH_RETRY_429_BASE_DELAY_S", "1.0"))
    max_delay_s = float(os.environ.get("EXPERIENCE_BENCH_RETRY_429_MAX_DELAY_S", "30.0"))

    attempt = 0
    while True:
        r = client.post(url, headers=headers, json=json, timeout=timeout_s)
        if r.status_code != 429:
            return r

        attempt += 1
        if attempt >= max_attempts:
            return r

        retry_after = _retry_after_seconds(r.headers.get("retry-after"))
        if retry_after is None:
            # attempt=1 -> base_delay_s
            retry_after = min(max_delay_s, base_delay_s * (2 ** (attempt - 1)))
        else:
            retry_after = min(max_delay_s, max(0.0, float(retry_after)))

        # Add small jitter to avoid synchronization across runs.
        jitter = random.uniform(0.0, min(1.0, 0.25 * retry_after))
        time.sleep(retry_after + jitter)


def _retry_after_seconds(value: str | None) -> float | None:
    if not value:
        return None

    v = value.strip()
    # RFC: Retry-After can be delta-seconds.
    if v.isdigit():
        return float(int(v))

    # Or an HTTP-date.
    try:
        dt = email.utils.parsedate_to_datetime(v)
        if dt is None:
            return None
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        now = datetime.now(timezone.utc)
        return max(0.0, (dt - now).total_seconds())
    except Exception:
        return None
