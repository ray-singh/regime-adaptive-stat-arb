from __future__ import annotations

import re

PAIR_ID_SEP = "–"
_PAIR_SPLIT_RE = re.compile(r"[/–]")


def make_pair_id(t1: str, t2: str) -> str:
    """Return canonical pair identifier in alphabetical order: 'AAA–BBB'."""
    a = str(t1).strip().upper()
    b = str(t2).strip().upper()
    return f"{min(a, b)}{PAIR_ID_SEP}{max(a, b)}"


def split_pair_id(pair_id: str) -> list[str]:
    """Split a pair id that may use slash or en-dash separators."""
    if not pair_id:
        return []
    parts = [p.strip().upper() for p in _PAIR_SPLIT_RE.split(str(pair_id)) if p.strip()]
    return parts
