from __future__ import annotations


from experience_bench.adapters.retry import _retry_after_seconds


def test_retry_after_seconds_parses_delta_seconds() -> None:
    assert _retry_after_seconds("0") == 0.0
    assert _retry_after_seconds("  12 ") == 12.0


def test_retry_after_seconds_returns_none_on_invalid() -> None:
    assert _retry_after_seconds(None) is None
    assert _retry_after_seconds("") is None
    assert _retry_after_seconds("nonsense") is None
