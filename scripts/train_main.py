"""兼容入口：将 scripts/train_main.py 路由到 src/train_main.py。"""

from src.train_main import main


if __name__ == "__main__":
    main()
