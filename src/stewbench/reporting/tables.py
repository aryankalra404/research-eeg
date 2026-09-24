"""Write tables as CSV, booktabs LaTeX and Markdown from one DataFrame."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


def fmt_mean_sd(mean: float | None, sd: float | None, digits: int = 3) -> str:
    if mean is None or pd.isna(mean):
        return "--"
    if sd is None or pd.isna(sd):
        return f"{mean:.{digits}f}"
    return f"{mean:.{digits}f} ± {sd:.{digits}f}"


def fmt_ci(ci) -> str:
    if not ci or ci[0] is None:
        return "--"
    return f"[{ci[0]:.3f}, {ci[1]:.3f}]"


def _latex_escape(text: str) -> str:
    return (str(text).replace("\\", r"\textbackslash{}").replace("&", r"\&").replace("%", r"\%")
            .replace("_", r"\_").replace("#", r"\#").replace("±", r"$\pm$").replace("→", r"$\rightarrow$")
            .replace("Δ", r"$\Delta$"))


def to_latex(df: pd.DataFrame, caption: str, label: str, bold_max: dict | None = None) -> str:
    """Booktabs table. ``bold_max`` maps column -> row index to embolden."""
    columns = list(df.columns)
    align = "l" + "r" * (len(columns) - 1)
    lines = [r"\begin{table*}[t]", r"\centering", r"\small", rf"\caption{{{_latex_escape(caption)}}}",
             rf"\label{{{label}}}", rf"\begin{{tabular}}{{{align}}}", r"\toprule",
             " & ".join(_latex_escape(c) for c in columns) + r" \\", r"\midrule"]
    for i, (_, row) in enumerate(df.iterrows()):
        cells = []
        for column in columns:
            cell = _latex_escape(row[column])
            if bold_max and bold_max.get(column) == i:
                cell = rf"\textbf{{{cell}}}"
            cells.append(cell)
        lines.append(" & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table*}"]
    return "\n".join(lines) + "\n"


def to_markdown(df: pd.DataFrame) -> str:
    header = "| " + " | ".join(map(str, df.columns)) + " |"
    rule = "|" + "|".join("---" for _ in df.columns) + "|"
    rows = ["| " + " | ".join(str(v) for v in row) + " |" for row in df.itertuples(index=False)]
    return "\n".join([header, rule, *rows]) + "\n"


def write_table(df_display: pd.DataFrame, path: Path, caption: str, label: str,
                df_numeric: pd.DataFrame | None = None, bold_max: dict | None = None) -> None:
    """Writes <path>.csv (numeric if given), <path>.tex and <path>.md."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    (df_numeric if df_numeric is not None else df_display).to_csv(path.with_suffix(".csv"), index=False)
    path.with_suffix(".tex").write_text(to_latex(df_display, caption, label, bold_max))
    path.with_suffix(".md").write_text(f"**{caption}**\n\n" + to_markdown(df_display))
