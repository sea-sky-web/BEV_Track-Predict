"""Training entry point for spatiotemporal field prediction.

Usage:
    PYTHONPATH=src python3 src/temporal/train_temporal_main.py \
        --annotations_dir wildtrack/annotations_positions \
        --output_dir outputs/temporal/run_seed0 \
        --seed 0 --device cuda

    # Three ablation modes:
    --ablation occ_only   # occupancy loss only
    --ablation occ_vel    # + velocity loss
    --ablation full       # + trace consistency (default)
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from temporal.convlstm import SpatioTemporalPredictor
from temporal.temporal_loss import CombinedTemporalLoss
from temporal.temporal_dataset import FieldSequenceDataset
from temporal.temporal_trainer import TemporalTrainer, set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train ConvLSTM spatiotemporal predictor")
    parser.add_argument("--annotations_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="outputs/temporal/run")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default="cuda")

    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=10)

    parser.add_argument("--history_len", type=int, default=4)
    parser.add_argument("--future_len", type=int, default=4)
    parser.add_argument("--sigma_m", type=float, default=0.2)

    parser.add_argument("--hidden_channels", type=int, default=32)
    parser.add_argument("--kernel_size", type=int, default=3)
    parser.add_argument("--n_encoder_layers", type=int, default=2)

    parser.add_argument("--lambda_vel", type=float, default=0.5)
    parser.add_argument("--lambda_trace", type=float, default=0.1)
    parser.add_argument("--ablation", type=str, default="full",
                        choices=["occ_only", "occ_vel", "full"])

    parser.add_argument("--log_every", type=int, default=10)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    device = torch.device(args.device)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"[CFG] seed={args.seed} device={device} ablation={args.ablation}")
    print(f"[CFG] epochs={args.epochs} batch={args.batch} lr={args.lr} wd={args.weight_decay}")
    print(f"[CFG] history={args.history_len} future={args.future_len} sigma_m={args.sigma_m}")

    ann_dir = Path(args.annotations_dir)
    print("[DATA] Loading train split...")
    train_ds = FieldSequenceDataset(ann_dir, split="train",
                                    history_len=args.history_len,
                                    future_len=args.future_len,
                                    sigma_m=args.sigma_m)
    print(f"[DATA] Train: {len(train_ds)} windows")

    print("[DATA] Loading val split...")
    val_ds = FieldSequenceDataset(ann_dir, split="val",
                                  history_len=args.history_len,
                                  future_len=args.future_len,
                                  sigma_m=args.sigma_m)
    print(f"[DATA] Val: {len(val_ds)} windows")

    train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True,
                              drop_last=True, pin_memory=device.type == "cuda")
    val_loader = DataLoader(val_ds, batch_size=args.batch, shuffle=False,
                            pin_memory=device.type == "cuda")

    model = SpatioTemporalPredictor(
        in_channels=5,
        hidden_channels=args.hidden_channels,
        kernel_size=args.kernel_size,
        n_encoder_layers=args.n_encoder_layers,
        n_future=args.future_len,
        out_channels=3,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"[MODEL] SpatioTemporalPredictor: {n_params:,} parameters")

    criterion = CombinedTemporalLoss(
        lambda_vel=args.lambda_vel,
        lambda_trace=args.lambda_trace,
        ablation=args.ablation,
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    trainer = TemporalTrainer(
        model=model, criterion=criterion, optimizer=optimizer,
        scheduler=scheduler, device=device, output_dir=output_dir,
        patience=args.patience,
    )

    for epoch in range(args.epochs):
        train_metrics = trainer.train_epoch(train_loader, epoch, log_every=args.log_every)
        val_metrics = trainer.validate(val_loader)

        print(
            f"[epoch {epoch}] "
            f"train_loss={train_metrics['total']:.6f} "
            f"val_loss={val_metrics['total']:.6f} "
            f"val_auprc={val_metrics['auprc']:.4f} "
            f"best_auprc={trainer.best_auprc:.4f}"
        )

        trainer.save_checkpoint(epoch, val_metrics)

        if val_metrics["auprc"] > trainer.best_auprc or trainer.best_auprc < 0:
            trainer.save_checkpoint(epoch, val_metrics, best=True)

        if trainer.check_early_stop(val_metrics["auprc"]):
            print(f"[STOP] Early stopping at epoch {epoch} (patience={args.patience})")
            break

    print(f"[DONE] Best val AUPRC: {trainer.best_auprc:.4f}")


if __name__ == "__main__":
    main()
