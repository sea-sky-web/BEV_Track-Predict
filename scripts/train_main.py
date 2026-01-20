# scripts/train_main.py
"""
主训练脚本：参数解析、组件初始化、训练循环调用
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from config import (
    DEFAULT_DATA_ROOT, DEFAULT_OUTPUT_DIR,
    DEFAULT_MAX_FRAMES, DEFAULT_EPOCHS, DEFAULT_BATCH_SIZE,
    DEFAULT_BEV_DOWN, DEFAULT_FEAT_H, DEFAULT_FEAT_W,
    DEFAULT_IMG_H, DEFAULT_IMG_W, DEFAULT_PERSON_H,
    DEFAULT_ALPHA, DEFAULT_MAP_KSIZE, DEFAULT_MAP_SIGMA,
    DEFAULT_IMG_KSIZE, DEFAULT_IMG_SIGMA, DEFAULT_PRETRAINED,
    DEFAULT_FREEZE_BN, DEFAULT_AMP_ENABLED, DEFAULT_DEVICE,
    DEFAULT_MAX_LR, DEFAULT_NUM_WORKERS, DEFAULT_LOG_EVERY,
    DEFAULT_SAVE_STEPS, DEFAULT_FIXED_STEM, DEFAULT_VALID_THR,
    CAM_NAMES,
    IMG_ORI_W, IMG_ORI_H,
)
from calibration import CalibrationLoader, decide_unit_scale, parse_rectangles_pom
from geometry import make_worldgrid2worldcoord_mat, build_mvdet_proj_mat, compute_valid_ratio_from_homography
from dataset import create_wildtrack_dataset
from models import create_model
from trainer import MVDetTrainer, create_optimizer, create_scheduler
from utils import build_gaussian_kernel_2d


def parse_args():
    """解析命令行参数"""
    ap = argparse.ArgumentParser(
        description="MVDet 风格的多视角 BEV 检测训练脚本"
    )
    
    # 数据相关
    ap.add_argument("--data_root", type=str, default=DEFAULT_DATA_ROOT,
                    help="Wildtrack 数据集根目录")
    ap.add_argument("--views", type=str, default="0,1,2",
                    help="使用的视角 ID，例如 0,1,2 或 0,3,5")
    ap.add_argument("--drop_bad_views", action="store_true",
                    help="是否丢弃低有效性的视角")
    ap.add_argument("--valid_thr", type=float, default=DEFAULT_VALID_THR,
                    help="投影有效性阈值")
    
    # 训练超参数
    ap.add_argument("--max_frames", type=int, default=DEFAULT_MAX_FRAMES,
                    help="最大帧数（-1 表示全部）")
    ap.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS,
                    help="训练轮数")
    ap.add_argument("--batch", type=int, default=DEFAULT_BATCH_SIZE,
                    help="批大小")
    
    # 网络架构
    ap.add_argument("--bev_down", type=int, default=DEFAULT_BEV_DOWN,
                    help="BEV 下采样倍数")
    ap.add_argument("--feat_h", type=int, default=DEFAULT_FEAT_H,
                    help="特征平面高度")
    ap.add_argument("--feat_w", type=int, default=DEFAULT_FEAT_W,
                    help="特征平面宽度")
    ap.add_argument("--img_h", type=int, default=DEFAULT_IMG_H,
                    help="输入图像高度")
    ap.add_argument("--img_w", type=int, default=DEFAULT_IMG_W,
                    help="输入图像宽度")
    
    # 人体模型和损失
    ap.add_argument("--person_h", type=float, default=DEFAULT_PERSON_H,
                    help="人体高度（米）")
    ap.add_argument("--alpha", type=float, default=DEFAULT_ALPHA,
                    help="图像损失权重")
    ap.add_argument("--map_ksize", type=int, default=DEFAULT_MAP_KSIZE,
                    help="BEV 热图高斯核大小")
    ap.add_argument("--map_sigma", type=float, default=DEFAULT_MAP_SIGMA,
                    help="BEV 热图高斯标准差")
    ap.add_argument("--img_ksize", type=int, default=DEFAULT_IMG_KSIZE,
                    help="图像热图高斯核大小")
    ap.add_argument("--img_sigma", type=float, default=DEFAULT_IMG_SIGMA,
                    help="图像热图高斯标准差")
    
    # 优化器和训练策略
    ap.add_argument("--pretrained", action="store_true", default=DEFAULT_PRETRAINED,
                    help="是否使用预训练权重")
    ap.add_argument("--freeze_bn", action="store_true", default=DEFAULT_FREEZE_BN,
                    help="是否冻结 BatchNorm")
    ap.add_argument("--amp", action="store_true", default=DEFAULT_AMP_ENABLED,
                    help="是否启用自动混合精度")
    ap.add_argument("--device", type=str, default=DEFAULT_DEVICE,
                    choices=["cuda", "cpu"],
                    help="计算设备")
    ap.add_argument("--max_lr", type=float, default=DEFAULT_MAX_LR,
                    help="最大学习率")
    ap.add_argument("--num_workers", type=int, default=DEFAULT_NUM_WORKERS,
                    help="数据加载线程数")
    
    # 日志和检查点
    ap.add_argument("--log_every", type=int, default=DEFAULT_LOG_EVERY,
                    help="每多少步打印日志")
    ap.add_argument("--save_steps", type=str, default=DEFAULT_SAVE_STEPS,
                    help="保存可视化的步数列表")
    ap.add_argument("--fixed_stem", type=str, default=DEFAULT_FIXED_STEM,
                    help="固定可视化的样本名称")
    
    return ap.parse_args()


def main():
    """主函数"""
    args = parse_args()
    
    # 设置设备
    dev = torch.device(args.device)
    print(f"[DEV] device={dev}, cuda_available={torch.cuda.is_available()}")
    if dev.type == "cuda":
        print(f"[DEV] gpu={torch.cuda.get_device_name(0)}")
        torch.backends.cudnn.benchmark = True
    
    # 设置输出目录
    data_root = Path(args.data_root)
    out_dir = Path(DEFAULT_OUTPUT_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # ========== 1. 加载和处理标定 ==========
    print("\n[CALIB] Loading calibration...")
    
    # 解析 rectangles.pom
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
    print(f"[UNIT] step={step_m}, median||t||={np.median(t_norms):.2f} => unit_scale={unit_scale}")
    
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
    
    for v in views:
        K0 = calib_cache[v]["K0"]
        from calibration import scale_intrinsics
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
        except np.linalg.LinAlgError:
            print(f"[GRID] view={v} cam={calib_cache[v]['cam']} singular")
            if args.drop_bad_views:
                continue
            else:
                raise RuntimeError("投影矩阵奇异")
        
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
        num_workers=args.num_workers,
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
        pretrained=args.pretrained,
        feat_ch=512,
        add_coord=True,
    )
    
    print(f"[MODEL] {type(model).__name__}")
    
    # ========== 5. 创建优化器和调度器 ==========
    print("\n[OPT] Creating optimizer and scheduler...")
    
    optimizer = create_optimizer(model, lr=1e-3, momentum=0.5, weight_decay=5e-4)
    scheduler = create_scheduler(
        optimizer,
        max_lr=args.max_lr,
        epochs=args.epochs,
        steps_per_epoch=len(loader)
    )
    
    # ========== 6. 创建训练器 ==========
    print("\n[TRAIN] Creating trainer...")
    
    trainer = MVDetTrainer(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        device=dev,
        output_dir=out_dir,
        amp_enabled=args.amp,
        freeze_bn=args.freeze_bn,
    )
    
    # 构建高斯核
    map_kernel = build_gaussian_kernel_2d(args.map_ksize, args.map_sigma, device=dev)
    img_kernel = build_gaussian_kernel_2d(args.img_ksize, args.img_sigma, device=dev)
    
    save_steps = set(int(s.strip()) for s in args.save_steps.split(",") if s.strip().isdigit())
    
    # ========== 7. 训练循环 ==========
    print("\n[TRAIN] Starting training...\n")
    
    for ep in range(args.epochs):
        print(f"=== Epoch {ep+1}/{args.epochs} ===")
        
        epoch_stats = trainer.train_epoch(
            loader,
            map_kernel,
            img_kernel,
            alpha=args.alpha,
            log_every=args.log_every,
        )
        
        print(f"[Epoch {ep}] "
              f"loss={epoch_stats['loss']:.6f} "
              f"bev={epoch_stats['bev_loss']:.6f} "
              f"img={epoch_stats['img_loss']:.6f}")
        
        # 保存检查点
        if (ep + 1) % 5 == 0:
            trainer.save_checkpoint(ep)
    
    # ========== 8. 保存最终模型 ==========
    print("\n[SAVE] Saving final model...")
    torch.save(model.state_dict(), out_dir / "model_final.pth")
    print(f"[OK] saved {out_dir / 'model_final.pth'}")


if __name__ == "__main__":
    main()
