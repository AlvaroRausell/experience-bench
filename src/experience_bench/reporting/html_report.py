from __future__ import annotations

from pathlib import Path
from typing import Any

import plotly.colors
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from rich.console import Console

from experience_bench.core.storage import read_jsonl


def build_html_report(*, jsonl_path: Path, out_html_path: Path, console: Console) -> dict[str, Any]:
    rows = list(read_jsonl(jsonl_path))
    if not rows:
        raise ValueError("No records found")

    groups: dict[tuple[str, str, int], list[dict[str, Any]]] = {}
    for r in rows:
        try:
            key = (str(r.get("provider")), str(r.get("model_key")), int(r.get("years")))
        except Exception:
            continue
        groups.setdefault(key, []).append(r)

    # Aggregate per (provider, model_key, years)
    agg: dict[tuple[str, str], dict[str, Any]] = {}
    for (provider, model_key, years), items in groups.items():
        s_key = (provider, model_key)
        s = agg.setdefault(
            s_key,
            {
                "years": [],
                "n": [],
                "n_perf": [],
                "pass_a": [],
                "pass_b": [],
                "pass_all": [],
                "ttlt_p50": [],
                "ttlt_p90": [],
                "exec_p50": [],
                "exec_p90": [],
            },
        )

        n = len(items)
        pass_a_cnt = sum(1 for it in items if it.get("passed_a") is True)
        pass_b_cnt = sum(1 for it in items if it.get("passed_b") is True)
        pass_all_cnt = sum(1 for it in items if it.get("passed_all") is True)

        # Performance charts should only include runs where at least one part passed.
        items_perf = [
            it
            for it in items
            if (it.get("passed_a") is True) or (it.get("passed_b") is True)
        ]
        n_perf = len(items_perf)

        ttlt_vals = [
            float(it["ttlt_ms"]) for it in items_perf if it.get("ttlt_ms") is not None
        ]
        exec_vals = [
            float(it["exec_ms"]) for it in items_perf if it.get("exec_ms") is not None
        ]

        s["years"].append(years)
        s["n"].append(n)
        s["n_perf"].append(n_perf)
        s["pass_a"].append((pass_a_cnt / n) if n else 0.0)
        s["pass_b"].append((pass_b_cnt / n) if n else 0.0)
        s["pass_all"].append((pass_all_cnt / n) if n else 0.0)
        s["ttlt_p50"].append(_pctl(ttlt_vals, 50))
        s["ttlt_p90"].append(_pctl(ttlt_vals, 90))
        s["exec_p50"].append(_pctl(exec_vals, 50))
        s["exec_p90"].append(_pctl(exec_vals, 90))

    def _label(provider: str, model_key: str) -> str:
        return f"{provider}:{model_key}"

    # Consistent color mapping across all charts
    palette = (
        plotly.colors.qualitative.Plotly
        + plotly.colors.qualitative.Safe
        + plotly.colors.qualitative.D3
    )
    series_keys = sorted(agg.keys(), key=lambda x: (x[0], x[1]))
    label_to_color: dict[str, str] = {}
    for idx, (provider, model_key) in enumerate(series_keys):
        label_to_color[_label(provider, model_key)] = palette[idx % len(palette)]

    # Figure 1: Pass rates (line charts) with Part A / Part B / All clearly separated
    fig_pass = make_subplots(
        rows=1,
        cols=3,
        subplot_titles=("Pass rate — Part A", "Pass rate — Part B", "Pass rate — All parts"),
        shared_yaxes=True,
    )

    for (provider, model_key), s in sorted(agg.items(), key=lambda x: (x[0][0], x[0][1])):
        order = sorted(range(len(s["years"])), key=lambda i: s["years"][i])
        years_sorted = [s["years"][i] for i in order]
        n_sorted = [s["n"][i] for i in order]
        label = _label(provider, model_key)
        color = label_to_color.get(label)

        pass_a = [s["pass_a"][i] for i in order]
        pass_b = [s["pass_b"][i] for i in order]
        pass_all = [s["pass_all"][i] for i in order]

        fig_pass.add_trace(
            go.Scatter(
                x=years_sorted,
                y=pass_a,
                mode="lines+markers",
                name=label,
                legendgroup=label,
                line=dict(color=color) if color else None,
                marker=dict(color=color) if color else None,
                hovertemplate=(
                    "Model=%{fullData.name}<br>Years=%{x}<br>Pass rate=%{y:.0%}<br>n=%{customdata}<extra></extra>"
                ),
                customdata=n_sorted,
            ),
            row=1,
            col=1,
        )
        fig_pass.add_trace(
            go.Scatter(
                x=years_sorted,
                y=pass_b,
                mode="lines+markers",
                name=label,
                legendgroup=label,
                showlegend=False,
                line=dict(color=color) if color else None,
                marker=dict(color=color) if color else None,
                hovertemplate=(
                    "Model=%{fullData.name}<br>Years=%{x}<br>Pass rate=%{y:.0%}<br>n=%{customdata}<extra></extra>"
                ),
                customdata=n_sorted,
            ),
            row=1,
            col=2,
        )
        fig_pass.add_trace(
            go.Scatter(
                x=years_sorted,
                y=pass_all,
                mode="lines+markers",
                name=label,
                legendgroup=label,
                showlegend=False,
                line=dict(color=color) if color else None,
                marker=dict(color=color) if color else None,
                hovertemplate=(
                    "Model=%{fullData.name}<br>Years=%{x}<br>Pass rate=%{y:.0%}<br>n=%{customdata}<extra></extra>"
                ),
                customdata=n_sorted,
            ),
            row=1,
            col=3,
        )

    fig_pass.update_layout(
        title=f"experience-bench report: {jsonl_path.name}",
        template="plotly_white",
        height=520,
        font=dict(size=14),
        legend=dict(orientation="h", yanchor="bottom", y=1.05, xanchor="left", x=0),
        margin=dict(l=60, r=20, t=90, b=60),
        hovermode="x unified",
    )
    fig_pass.update_yaxes(tickformat=".0%", rangemode="tozero")
    for c in (1, 2, 3):
        fig_pass.update_xaxes(title_text="Years of experience", row=1, col=c)

    # Figure 2: Performance (line charts). We plot p50 as solid and p90 as dotted.
    fig_perf = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=("TTLT (ms) — p50 & p90", "Execution (ms) — p50 & p90"),
        shared_yaxes=False,
    )

    for (provider, model_key), s in sorted(agg.items(), key=lambda x: (x[0][0], x[0][1])):
        order = sorted(range(len(s["years"])), key=lambda i: s["years"][i])
        years_sorted = [s["years"][i] for i in order]
        n_perf_sorted = [s["n_perf"][i] for i in order]
        label = _label(provider, model_key)
        color = label_to_color.get(label)

        ttlt_p50 = [s["ttlt_p50"][i] for i in order]
        ttlt_p90 = [s["ttlt_p90"][i] for i in order]
        exec_p50 = [s["exec_p50"][i] for i in order]
        exec_p90 = [s["exec_p90"][i] for i in order]

        fig_perf.add_trace(
            go.Scatter(
                x=years_sorted,
                y=ttlt_p50,
                mode="lines+markers",
                name=label,
                legendgroup=label,
                marker=dict(color=color) if color else None,
                line=dict(color=color) if color else None,
                hovertemplate=(
                    "Model=%{fullData.name}<br>Years=%{x}<br>p50=%{y:.1f} ms<br>n_used=%{meta}<extra></extra>"
                ),
                meta=n_perf_sorted,
            ),
            row=1,
            col=1,
        )

        fig_perf.add_trace(
            go.Scatter(
                x=years_sorted,
                y=ttlt_p90,
                mode="lines+markers",
                name=f"{label} p90",
                legendgroup=label,
                showlegend=False,
                marker=dict(color=color) if color else None,
                line=(dict(color=color, dash="dot") if color else dict(dash="dot")),
                hovertemplate=(
                    "Model=%{legendgroup}<br>Years=%{x}<br>p90=%{y:.1f} ms<br>n_used=%{meta}<extra></extra>"
                ),
                meta=n_perf_sorted,
            ),
            row=1,
            col=1,
        )

        fig_perf.add_trace(
            go.Scatter(
                x=years_sorted,
                y=exec_p50,
                mode="lines+markers",
                name=label,
                legendgroup=label,
                showlegend=False,
                marker=dict(color=color) if color else None,
                line=dict(color=color) if color else None,
                hovertemplate=(
                    "Model=%{legendgroup}<br>Years=%{x}<br>p50=%{y:.1f} ms<br>n_used=%{meta}<extra></extra>"
                ),
                meta=n_perf_sorted,
            ),
            row=1,
            col=2,
        )

        fig_perf.add_trace(
            go.Scatter(
                x=years_sorted,
                y=exec_p90,
                mode="lines+markers",
                name=f"{label} p90",
                legendgroup=label,
                showlegend=False,
                marker=dict(color=color) if color else None,
                line=(dict(color=color, dash="dot") if color else dict(dash="dot")),
                hovertemplate=(
                    "Model=%{legendgroup}<br>Years=%{x}<br>p90=%{y:.1f} ms<br>n_used=%{meta}<extra></extra>"
                ),
                meta=n_perf_sorted,
            ),
            row=1,
            col=2,
        )

    fig_perf.update_layout(
        template="plotly_white",
        height=520,
        font=dict(size=14),
        legend=dict(orientation="h", yanchor="bottom", y=1.05, xanchor="left", x=0),
        margin=dict(l=60, r=20, t=70, b=60),
        hovermode="x unified",
    )
    fig_perf.update_xaxes(title_text="Years of experience", row=1, col=1)
    fig_perf.update_xaxes(title_text="Years of experience", row=1, col=2)
    fig_perf.update_yaxes(title_text="ms", row=1, col=1)
    fig_perf.update_yaxes(title_text="ms", row=1, col=2)

    # Compose single HTML (two stacked figures) with basic legibility defaults.
    html = "".join(
        [
            "<!doctype html>",
            "<html><head><meta charset='utf-8' />",
            f"<title>experience-bench report: {jsonl_path.name}</title>",
            "<meta name='viewport' content='width=device-width, initial-scale=1' />",
            "<style>body{max-width:1200px;margin:0 auto;padding:16px;font-family:system-ui,-apple-system,Segoe UI,Roboto,Helvetica,Arial,sans-serif}</style>",
            "</head><body>",
            "<h1 style='margin:0 0 12px 0'>experience-bench report</h1>",
            f"<div style='color:#444;margin:0 0 18px 0'>Source: {jsonl_path.name} &nbsp;·&nbsp; Records: {len(rows)} &nbsp;·&nbsp; Groups: {len(groups)}</div>",
            fig_pass.to_html(full_html=False, include_plotlyjs="cdn"),
            "<div style='height:10px'></div>",
            fig_perf.to_html(full_html=False, include_plotlyjs=False),
            "</body></html>",
        ]
    )
    out_html_path.write_text(html, encoding="utf-8")

    return {"n_records": len(rows), "n_groups": len(groups)}


def _pctl(values: list[float], p: int) -> float | None:
    if not values:
        return None
    values = sorted(values)
    if p <= 0:
        return values[0]
    if p >= 100:
        return values[-1]
    k = (len(values) - 1) * (p / 100.0)
    f = int(k)
    c = min(f + 1, len(values) - 1)
    if f == c:
        return values[f]
    d0 = values[f] * (c - k)
    d1 = values[c] * (k - f)
    return d0 + d1
