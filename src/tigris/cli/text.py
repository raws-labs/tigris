"""Plain terminal output: aligned columns, color only for meaning.

Click drops the color codes when the output is not a terminal, and NO_COLOR
turns them off everywhere.
"""

import os

import click


def _style(text: str, **how) -> str:
    return text if os.environ.get("NO_COLOR") else click.style(text, **how)


def good(text: str) -> str:
    return _style(text, fg="green", bold=True)


def bad(text: str) -> str:
    return _style(text, fg="red", bold=True)


def warn(text: str) -> str:
    return _style(text, fg="yellow")


def dim(text: str) -> str:
    return _style(text, dim=True)


def bold(text: str) -> str:
    return _style(text, bold=True)


def echo(text: str = "", err: bool = False) -> None:
    if not err:
        _printed()["lines"] += 1
    click.echo(text, err=err)


def _printed() -> dict:
    """What this command has written to stdout so far."""
    context = click.get_current_context(silent=True)
    meta = context.meta if context is not None else _NO_CONTEXT
    return meta.setdefault("tigris_output", {"lines": 0})


_NO_CONTEXT: dict = {}


def gap() -> None:
    """One blank line between blocks, none before the first."""
    if _printed()["lines"]:
        echo()


def columns(rows: list[list[str]], align: str = "", indent: int = 2, gap: int = 3) -> list[str]:
    """Rows of cells aligned in columns. `align` holds one '<' or '>' per
    column; columns beyond it align left. Widths ignore color codes."""
    if not rows:
        return []
    count = max(len(row) for row in rows)
    # A cell that renders empty is empty, so a row ends where its text does.
    rows = [[cell if click.unstyle(cell) else "" for cell in row] + [""] * (count - len(row))
            for row in rows]
    widths = [max(len(click.unstyle(row[k])) for row in rows) for k in range(count)]
    lines = []
    for row in rows:
        cells = []
        for k, cell in enumerate(row):
            pad = " " * (widths[k] - len(click.unstyle(cell)))
            right = k < len(align) and align[k] == ">"
            cells.append(pad + cell if right else cell + (pad if k < count - 1 else ""))
        lines.append(" " * indent + (" " * gap).join(cells).rstrip())
    return lines


def section(title: str, rows: list[list[str]], align: str = "") -> None:
    """A section title flush left and its rows aligned beneath it, one blank
    line from what came before."""
    gap()
    echo(bold(title))
    for line in columns(rows, align):
        echo(line)


def block(rows: list[list[str]], align: str = "") -> None:
    """Key and value rows flush left, one blank line from what came before."""
    gap()
    for line in columns(rows, align, indent=0):
        echo(line)
