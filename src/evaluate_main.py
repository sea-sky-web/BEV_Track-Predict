"""
评估轻量化多视角检测（MVDet）模型性能的脚本
"""

import argparse
import sys
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from config import (
    DEFAULT_DATA_ROOT, DEFAULT_OUTPUT_DIR,
    DEFAULT_MAX_FRAMES, DEFAULT_BATCH_SIZE,
    DEFAULT_BEV_DOWN, DEFAULT_FEAT_H, DEFAULT_FEAT_W,
    DEFAULT_IMG_H, DEFAULT_IMG_W, DEFAULT_PERSON_H,
    CAM_NAMES,
    IMG_ORI_W, IMG_ORI_H,
)
from calibration import CalibrationLoader, decide_unit_scale, parse_rectangles_pom
from geometry import make_worldgrid2worldcoord_mat, build_mvdet_proj_mat, compute_valid_ratio_from_homography
from dataset import create_wildtrack_dataset
from models import create_model
from trainer import MVDetTrainer
from utils import build_gaussian_kernel_2d


def parse_args():
    """解析命令行参数"""
    ap = argparse.ArgumentParser(
        description="MVDet 模型性能评估脚本"
    )
    
    # 数据相关
    ap.add_argument("--data_root", type=str, default=DEFAULT_DATA_ROOT,
                    help="Wildtrack 数据集根目录")
    ap.add_argument("--views", type=str, default="0,1,2",
                    help="使用的视角 ID")
    ap.add_argument("--drop_bad_views", action="store_true",
                    help="是否丢弃低有效性的视角")
    ap.add_argument("--valid_thr", type=float, default=0.10,
                    help="投影有效性阈值")
    
    # 模型相关
    ap.add_argument("--model_path", type=str, 
                    default=str(Path(DEFAULT_OUTPUT_DIR) / "model_final.pth"),
                    help="模型文件路径")
    ap.add_argument("--device", type=str, default="cuda",
                    choices=["cuda", "cpu"],
                    help="计算设备")
    
    # 评估参数
    ap.add_argument("--max_frames", type=int, default=DEFAULT_MAX_FRAMES,
                    help="评估时使用的最大帧数")
    ap.add_argument("--batch", type=int, default=DEFAULT_BATCH_SIZE,
                    help="批大小")
    
    return ap.parse_args()


def evaluate_model(model, dataloader, device):
    """评估模型性能"""
    model.eval()
    total_loss = 0.0
    count = 0
    
    with torch.no_grad():
        for batch in dataloader:
            stems, x_views, map_gt, imgs_gt = batch
            x_views = x_views.to(device)
            map_gt = map_gt.to(device)
            imgs_gt = imgs_gt.to(device)
            
            # 前向传播
            map_logits, imgs_logits = model(x_views)
            
            # 计算损失
            from loss import compute_loss
            loss, loss_info = compute_loss(
                map_logits, map_gt,
                imgs_logits, imgs_gt
            )
            
            total_loss += loss.item()
            count += 1
    
    avg_loss = total_loss / count
    print(f"平均损失值: {avg_loss:.4f}")
    return avg_loss


def test_inference_speed(model, device, batch_size=1):
    """测试模型推理速度"""
    model.eval()
    
    # 创建虚拟输入
    B = batch_size
    V = 3
    C = 3
    H = 720
    W = 1280
    
    dummy_input = torch.randn(B, V, C, H, W).to(device)
    
    # 预热
    for _ in range(10):
        with torch.no_grad():
            _ = model(dummy_input)
    
    # 测试
    import time
    start_time = time.time()
    num_runs = 100
    
    with torch.no_grad():
        for _ in range(num_runs):
            _ = model(dummy_input)
    
    end_time = time.time()
    
    avg_time = (end_time - start_time) / num_runs
    fps = 1 / avg_time * batch_size
    
    print(f"平均推理时间: {avg_time*1000:.2f} ms")
    print(f"推理速度: {fps:.2f} FPS")
    
    return avg_time, fps


def count_parameters(model):
    """统计模型参数量"""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"总参数量: {total_params / 1e6:.2f}M")
    print(f"可训练参数量: {trainable_params / 1e6:.2f}M")
    return total_params, trainable_params


def main():
    """主函数"""
    args = parse_args()
    
    # 设置设备
    dev = torch.device(args.device)
    print(f"[DEV] device={dev}, cuda_available={torch.cuda.is_available()}")
    if dev.type == "cuda":
        print(f"[DEV] gpu={torch.cuda.get_device_name(0)}")
        torch.backends.cudnn.benchmark = True
    
    # 检查模型文件是否存在
    model_path = Path(args.model_path)
    if not model_path.exists():
        print(f"错误: 模型文件 {model_path} 不存在")
        return
    
    # ========== 1. 加载和处理标定 ==========
    print("\n[CALIB] Loading calibration...")
    
    # 解析 rectangles.pom
    data_root = Path(args.data_root)
    pom = parse_rectangles_pom(data_root / "rectangles.pom")
    
    # 解析视角
    views = [int(x.strip()) for x in args.views.split(",") if x.strip().isdigit()]
    assert len(views) > 0, "至少需要一个视角"
    print(f"[CALIB] views={views}")
    
    # 加载标定数据
    calib_loader = CalibrationLoader(data_root / "calibrations", CAM_NAMES)
    calib_cache, t_norms = calib_loader.load_all(views)
    
    # 推断单位制
    step_m = float(pom.get("STEP", 0.025))
    unit_scale = decide_unit_scale(step_m, t_norms)
    print(f"[UNIT] step={step_m}, median||t||={torch.median(torch.tensor(t_norms)):.2f} => unit_scale={unit_scale}")
    
    # ========== 2. 构建投影矩阵 ==========
    print("\n[GRID] Building projection matrices...")
    
    Hb = int(pom.get("NB_HEIGHT", 1440)) // args.bev_down
    Wb = int(pom.get("NB_WIDTH", 480)) // args.bev_down
    Hf, Wf = args.feat_h, args.feat_w
    
    print(f"[CFG] img={args.img_h}x{args.img_w}, "
          f"feat={Hf}x{Wf}, bev(reduced)={Hb}x{Wb}")
    
    # 缩放内参
    sx_f = Wf / IMG_ORI_W
    sy_f = Hf / IMG_ORI_H
    
    from calibration import scale_intrinsics
    for v in views:
        K0 = calib_cache[v]["K0"]
        K_feat = scale_intrinsics(K0, sx=sx_f, sy=sy_f)
        calib_cache[v]["K_feat"] = K_feat
    
    # 构建投影矩阵
    origin_x_m = float(pom.get("ORIGINE_X", -3.0))
    origin_y_m = float(pom.get("ORIGINE_Y", -9.0))
    step = (step_m * args.bev_down) * unit_scale
    ox = origin_x_m * unit_scale
    oy = origin_y_m * unit_scale
    
    w2w_mat = make_worldgrid2worldcoord_mat(ox, oy, step)
    
    proj_mats = []
    kept_views = []
    for v in views:
        K_feat = calib_cache[v]["K_feat"]
        R = calib_cache[v]["R"]
        t = calib_cache[v]["t"]
        
        try:
            proj = build_mvdet_proj_mat(K_feat, R, t, w2w_mat)
        except Exception as e:
            print(f"[GRID] view={v} cam={calib_cache[v]['cam']} 错误: {e}")
            if args.drop_bad_views:
                continue
            else:
                raise RuntimeError("投影矩阵构建失败")
        
        vr = compute_valid_ratio_from_homography(proj, (Hf, Wf), (Hb, Wb))
        print(f"[GRID] view={v} cam={calib_cache[v]['cam']} valid_ratio={vr:.4f}")
        
        if args.drop_bad_views and vr < args.valid_thr:
            print(f"[GRID] drop view={v}")
            continue
        
        proj_mats.append(torch.from_numpy(proj).float())
        kept_views.append(v)
    
    assert len(kept_views) > 0, "没有有效的视角"
    print(f"[CFG] kept_views={kept_views}")
    
    proj_mats_t = torch.stack(proj_mats, dim=0).to(dev)
    
    # ========== 3. 创建数据集 ==========
    print("\n[DATA] Creating dataset...")
    
    ds = create_wildtrack_dataset(
        data_root=data_root,
        views=kept_views,
        max_frames=args.max_frames,
        img_hw=(args.img_h, args.img_w),
        feat_hw=(Hf, Wf),
        bev_down=args.bev_down,
        person_h_m=args.person_h,
        unit_scale=unit_scale,
        calib_cache=calib_cache,
    )
    
    def collate_fn(batch):
        stems, x_views, map_gt, imgs_gt = zip(*batch)
        return list(stems), torch.stack(x_views, 0), torch.stack(map_gt, 0), torch.stack(imgs_gt, 0)
    
    loader = DataLoader(
        ds,
        batch_size=args.batch,
        shuffle=True,
        num_workers=0,
        pin_memory=(dev.type == "cuda"),
        drop_last=True,
        collate_fn=collate_fn,
    )
    
    print(f"[DATA] len(ds)={len(ds)}, len(loader)={len(loader)}")
    
    # ========== 4. 创建模型 ==========
    print("\n[MODEL] Creating model...")
    
    model = create_model(
        num_views=len(kept_views),
        proj_mats=proj_mats_t,
        reduced_hw=(Hb, Wb),
        feat_hw=(Hf, Wf),
        device=dev,
        pretrained=False,
        feat_ch=512,
        add_coord=True,
    )
    
    print(f"[MODEL] {type(model).__name__}")
    
    # 加载模型权重
    print(f"[MODEL] Loading weights from {model_path}...")
    state_dict = torch.load(model_path, map_location=dev)
    model.load_state_dict(state_dict)
    print("[OK] 模型权重加载成功")
    
    # ========== 5. 模型评估 ==========
    print("\n[EVAL] Starting evaluation...")
    
    # 计算损失
    avg_loss = evaluate_model(model, loader, dev)
    
    # 测试推理速度
    avg_time, fps = test_inference_speed(model, dev)
    
    # 统计模型参数量
    total_params, trainable_params = count_parameters(model)
    
    # 计算模型大小
    torch.save(model.state_dict(), "temp_model.pth")
    model_size = Path("temp_model.pth").stat().st_size / (1024*1024)  # MB
    print(f"模型大小: {model_size:.2f} MB")
    Path("temp_model.pth").unlink()
    
    print("\n" + "="*50)
    print("评估结果总结")
    print("="*50)
    print(f"平均损失值: {avg_loss:.4f}")
    print(f"推理速度: {fps:.2f} FPS")
    print(f"平均推理时间: {avg_time*1000:.2f} ms")
    print(f"模型大小: {model_size:.2f} MB")
    print(f"总参数量: {total_params / 1e6:.2f}M")
    print(f"可训练参数量: {trainable_params / 1e6:.2f}M")
    print("="*50)
    
    return {
        "avg_loss": avg_loss,
        "fps": fps,
        "avg_time": avg_time,
        "model_size": model_size,
        "total_params": total_params,
        "trainable_params": trainable_params
    }


if __name__ == "__main__":
    main()
