"""Remove comments and docstrings from source files.

Docstrings are located with `ast` so that only strings in statement position are
removed and string literals used as values survive. Comments are located with
`tokenize` so that a `#` inside a string is never mistaken for one. Every result
is re-parsed before being written, and files whose behaviour would change are
skipped rather than mangled.
"""
from __future__ import annotations

import argparse
import ast
import io
import re
import sys
import tokenize
from pathlib import Path

DOC_NODES = (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)


def _docstring_line_ranges(tree: ast.AST) -> set[int]:
    lines: set[int] = set()
    for node in ast.walk(tree):
        if not isinstance(node, DOC_NODES):
            continue
        body = getattr(node, "body", None)
        if not body:
            continue
        first = body[0]
        if (
            isinstance(first, ast.Expr)
            and isinstance(first.value, ast.Constant)
            and isinstance(first.value.value, str)
        ):
            # a lone docstring in an otherwise empty body must leave a `pass`
            if len(body) == 1:
                continue
            lines.update(range(first.lineno, (first.end_lineno or first.lineno) + 1))
    return lines


def _comment_spans(source: str) -> dict[int, int]:
    """Map line number to the column where a comment starts on that line."""
    spans: dict[int, int] = {}
    try:
        for token in tokenize.generate_tokens(io.StringIO(source).readline):
            if token.type == tokenize.COMMENT:
                spans[token.start[0]] = token.start[1]
    except tokenize.TokenError:
        pass
    return spans


def strip_source(source: str) -> str:
    tree = ast.parse(source)
    doc_lines = _docstring_line_ranges(tree)
    comments = _comment_spans(source)

    out: list[str] = []
    for number, line in enumerate(source.splitlines(), start=1):
        if number in doc_lines:
            continue
        if number in comments:
            line = line[: comments[number]].rstrip()
            if not line:
                continue
        out.append(line.rstrip())

    text = "\n".join(out)
    text = re.sub(r"\n{3,}", "\n\n\n", text)
    return text.strip() + "\n"


def _behaviour_preserved(original: str, stripped: str) -> bool:
    """Compare ASTs with docstrings removed. If they differ, the strip changed
    more than documentation and must not be written."""

    def normalise(code: str) -> str:
        tree = ast.parse(code)
        for node in ast.walk(tree):
            if isinstance(node, DOC_NODES) and getattr(node, "body", None):
                first = node.body[0]
                if (
                    isinstance(first, ast.Expr)
                    and isinstance(first.value, ast.Constant)
                    and isinstance(first.value.value, str)
                    and len(node.body) > 1
                ):
                    node.body.pop(0)
        return ast.dump(tree)

    return normalise(original) == normalise(stripped)


def process(path: Path, dry_run: bool = False) -> bool:
    original = path.read_text()
    try:
        stripped = strip_source(original)
        ast.parse(stripped)
    except SyntaxError as exc:
        print(f"  SKIP {path.name}: would not parse ({exc})", file=sys.stderr)
        return False

    if not _behaviour_preserved(original, stripped):
        print(f"  SKIP {path.name}: behaviour would change", file=sys.stderr)
        return False

    delta = len(stripped) - len(original)
    print(f"  {path.name:24s} {len(original):6d} -> {len(stripped):6d} ({delta:+d})")
    if not dry_run:
        path.write_text(stripped)
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description="Strip comments and docstrings.")
    parser.add_argument("paths", nargs="+")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    targets: list[Path] = []
    for raw in args.paths:
        path = Path(raw)
        targets.extend(sorted(path.rglob("*.py")) if path.is_dir() else [path])

    ok = sum(process(path, args.dry_run) for path in targets)
    print(f"{ok}/{len(targets)} files processed")
    if ok != len(targets):
        sys.exit(1)


if __name__ == "__main__":
    main()
