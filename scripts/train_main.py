"""兼容入口：将 scripts/train_main.py 路由到 src/train_main.py。"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.train_main import main


if __name__ == "__main__":
    main()
