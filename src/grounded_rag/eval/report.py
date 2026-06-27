"""Report serialisation + a human-readable summary.

A :class:`Report` is versioned and self-describing (schema version, git sha, seed,
mock flag, config snapshot) so any ``report.json`` is reproducible and auditable.
``summary.md`` is generated from the same object — one source of truth.
"""

from __future__ import annotations

from pathlib import Path

from grounded_rag.core.types import Report

REPORT_SCHEMA_VERSION = "1"


def _fmt(value: float | None) -> str:
    return "N/A" if value is None else f"{value:.3f}"


def report_to_markdown(report: Report) -> str:
    """Render a human-readable summary of a run."""
    s = report.summary
    recall_cols = " · ".join(f"recall@{k}={_fmt(v)}" for k, v in sorted(s.recall_at_k.items()))
    lines = [
        "# Evaluation report",
        "",
        f"- schema: `{report.schema_version}` · git: `{report.git_sha}` · "
        f"mock_mode: `{report.mock_mode}` · seed: `{report.seed}`",
        f"- timestamp: `{report.timestamp}`",
        "",
        "## Headline",
        "",
        f"| queries | correctness | groundedness | refusal acc | {recall_cols} | "
        "p50 ms | p95 ms | cost/query |",
        "|--:|--:|--:|--:|--:|--:|--:|--:|",
        f"| {s.n} | {_fmt(s.correctness)} | {_fmt(s.groundedness)} | "
        f"{_fmt(s.refusal_accuracy)} | — | {s.p50_latency_ms:.1f} | {s.p95_latency_ms:.1f} | "
        f"${s.cost_per_query_usd:.4f} |",
        "",
        "## Per query",
        "",
        "| id | type | correct | grounded | refusal ok | score | agreement |",
        "|---|---|:--:|:--:|:--:|--:|--:|",
    ]
    for q in report.per_query:
        v = q.verdict
        lines.append(
            f"| {q.record_id} | {q.type.value} | {'✓' if v.correct else '✗'} | "
            f"{'✓' if v.grounded else '✗'} | "
            f"{'✓' if q.refusal_correct else '✗'} | {_fmt(v.score)} | {_fmt(v.agreement)} |"
        )
    return "\n".join(lines) + "\n"


def write_report(report: Report, out_dir: str | Path) -> Path:
    """Write ``report.json`` + ``summary.md`` into ``out_dir``; return the json path."""
    directory = Path(out_dir)
    directory.mkdir(parents=True, exist_ok=True)
    json_path = directory / "report.json"
    json_path.write_text(report.model_dump_json(indent=2), encoding="utf-8")
    (directory / "summary.md").write_text(report_to_markdown(report), encoding="utf-8")
    return json_path
