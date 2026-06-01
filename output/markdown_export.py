"""Markdown export — annotated table for manual review."""

from __future__ import annotations


def to_markdown(token_states: list) -> str:
    """Render token states as an annotated markdown table.

    Arabic text is wrapped in <span dir="rtl"> for RTL rendering.
    Accepted tokens (decision='accept') are shown minimally.
    """
    lines = [
        "| Original | Corrected | Confidence | Decision | Sources |",
        "|---|---|---|---|---|",
    ]
    for ts in token_states:
        orig = _rtl(ts.original)
        sel = _rtl(ts.selected)
        conf = f"{ts.confidence:.2f}"
        dec = ts.decision
        srcs = ", ".join(ts.sources) if ts.sources else "—"
        lines.append(f"| {orig} | {sel} | {conf} | {dec} | {srcs} |")
    return "\n".join(lines)


def _rtl(text: str) -> str:
    return f'<span dir="rtl">{text}</span>'
