import argparse
import json
from pathlib import Path
from collections import defaultdict

from PIL import Image, ImageDraw, ImageFont

# 你可以沿用你之前的 configs/config.py
try:
    from configs.config import DATA_ROOT, OUT_DIR
except Exception:
    # 如果你还没写 config，就先用当前脚本所在目录向上推一层
    BASE = Path(__file__).resolve().parents[1]
    DATA_ROOT = BASE / "wildtrack"
    OUT_DIR = BASE / "outputs"
    OUT_DIR.mkdir(parents=True, exist_ok=True)


def find_image_path(images_root: Path, cam_folder: str, stem: str) -> Path | None:
    """
    Wildtrack 常见命名是: Image_subsets/C1/000000.png 这类。
    如果扩展名不确定，就依次尝试。
    """
    cam_dir = images_root / cam_folder
    if not cam_dir.exists():
        return None

    for ext in [".png", ".jpg", ".jpeg", ".bmp"]:
        p = cam_dir / f"{stem}{ext}"
        if p.exists():
            return p
    return None


def load_font(size=14):
    """
    Windows 上一般有 Arial；如果找不到就用默认字体。
    """
    try:
        return ImageFont.truetype("arial.ttf", size=size)
    except Exception:
        return ImageFont.load_default()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--json",
        type=str,
        default="",
        help="指定某一帧的json文件路径，例如 wildtrack/annotations_positions/000000.json"
    )
    parser.add_argument(
        "--index",
        type=int,
        default=0,
        help="如果不指定--json，就用annotations_positions里排序后的第index个json"
    )
    parser.add_argument(
        "--mapping",
        type=str,
        default="plus1",
        choices=["plus1", "identity"],
        help="viewNum到相机文件夹的映射：plus1表示0->C1,...,6->C7；identity表示0->C0（一般用不上）"
    )
    parser.add_argument(
        "--max_people",
        type=int,
        default=0,
        help="限制最多画多少个人；0表示不限制（调试时可设为10）"
    )
    args = parser.parse_args()

    ann_dir = DATA_ROOT / "annotations_positions"
    images_root = DATA_ROOT / "Image_subsets"
    out_dir = OUT_DIR / "debug_bbox"
    out_dir.mkdir(parents=True, exist_ok=True)

    # 选定json文件
    if args.json:
        json_path = Path(args.json)
    else:
        files = sorted(ann_dir.glob("*.json"))
        assert files, f"No json found in {ann_dir}"
        assert 0 <= args.index < len(files), f"index out of range: {args.index}, total={len(files)}"
        json_path = files[args.index]

    data = json.loads(json_path.read_text(encoding="utf-8"))
    assert isinstance(data, list), "你的annotations_positions帧文件顶层应该是list（每个元素一个人）"

    stem = json_path.stem  # 通常与图像文件名一致
    font = load_font(14)

    # 将标注按相机分组：cam -> list of (personID, bbox)
    by_cam = defaultdict(list)

    # 可能有人在某些相机 notvisible，所以views里数量不固定
    count = 0
    for obj in data:
        if not isinstance(obj, dict):
            continue
        pid = obj.get("personID", None)
        views = obj.get("views", [])
        if pid is None or not isinstance(views, list):
            continue

        for v in views:
            if not isinstance(v, dict):
                continue
            view_num = v.get("viewNum", None)
            if view_num is None:
                continue

            # bbox字段名按你数据：xmin,xmax,ymin,ymax
            xmin = v.get("xmin", None)
            ymin = v.get("ymin", None)
            xmax = v.get("xmax", None)
            ymax = v.get("ymax", None)
            if None in (xmin, ymin, xmax, ymax):
                continue

            by_cam[int(view_num)].append((int(pid), int(xmin), int(ymin), int(xmax), int(ymax)))

        count += 1
        if args.max_people > 0 and count >= args.max_people:
            break

    # 逐相机画图
    for view_num, items in sorted(by_cam.items()):
        if args.mapping == "plus1":
            cam_folder = f"C{view_num + 1}"
        else:
            cam_folder = f"C{view_num}"

        img_path = find_image_path(images_root, cam_folder, stem)
        if img_path is None:
            print(f"[WARN] image not found for viewNum={view_num}, folder={cam_folder}, stem={stem}")
            continue

        img = Image.open(img_path).convert("RGB")
        draw = ImageDraw.Draw(img)

        # 画框 + 标ID
        for (pid, xmin, ymin, xmax, ymax) in items:
            # 框
            draw.rectangle([xmin, ymin, xmax, ymax], width=2)
            # 标注文字
            label = f"ID:{pid}"
            # 给文字加个底色块，便于看
            tw, th = draw.textbbox((0, 0), label, font=font)[2:]
            bx0, by0 = xmin, max(0, ymin - th - 2)
            draw.rectangle([bx0, by0, bx0 + tw + 4, by0 + th + 2], fill=(0, 0, 0))
            draw.text((bx0 + 2, by0 + 1), label, fill=(255, 255, 255), font=font)

        out_path = out_dir / f"{stem}_{cam_folder}.png"
        img.save(out_path)
        print(f"[OK] saved: {out_path}   (viewNum={view_num}, people={len(items)})")


if __name__ == "__main__":
    main()
