from pathlib import Path
from collections import deque


def tail_text(path: Path, n_lines: int) -> str:
    if not path.exists():
        return ""
    with path.open("r", encoding="utf-8", errors="replace") as f:
        lines = deque(f, maxlen=max(1, n_lines))
    return "".join(lines)
