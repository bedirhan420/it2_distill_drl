"""Rule-base report writers (Markdown / LaTeX table) for the paper's qualitative section."""
from __future__ import annotations

from pathlib import Path

from .rule_extractor import ExtractedRule


def write_markdown(rules: list[ExtractedRule], path: Path) -> None:
    lines = [f"{i+1}. {r}   _(support={r.support:.2f})_" for i, r in enumerate(rules)]
    Path(path).write_text("\n".join(lines))
