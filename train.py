import argparse
import csv
import json
import platform
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from change_detection.data import LevirDataset
from change_detection.metrics import evaluate
from change_detection.model import ChangeUNet


def arguments():
    parser = argparse.ArgumentParser(description="Train a building change detector on LEVIR-CD")
    parser.add_argument("--data", type=Path, default=Path("data/LEVIR-CD"))
    parser.add_argument("--run", type=Path, default=Path("runs/early"))
    parser.add_argument("--architecture", choices=("early", "siamese"), default="early")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--patch-size", type=int, default=256)
    parser.add_argument("--base", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main():
    args = arguments()
    if args.patch_size < 16 or args.patch_size % 16:
        raise ValueError("--patch-size must be a multiple of 16")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train = LevirDataset(args.data, "train", args.patch_size, augment=True)
    val = LevirDataset(args.data, "val", args.patch_size)
    loader = DataLoader(train, batch_size=args.batch_size, shuffle=True, num_workers=args.workers)
    val_loader = DataLoader(val, batch_size=args.batch_size, num_workers=args.workers)
    model = ChangeUNet(args.base, args.architecture).to(device)
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    bce = nn.BCEWithLogitsLoss()
    args.run.mkdir(parents=True, exist_ok=True)
    config = vars(args) | {"data": str(args.data), "run": str(args.run), "device": str(device),
                           "python": platform.python_version(), "torch": torch.__version__,
                           "cuda_device": torch.cuda.get_device_name(0) if device.type == "cuda" else None}
    (args.run / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")

    best = -1.0
    best_epoch = None
    started = time.perf_counter()
    with (args.run / "history.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=("epoch", "loss", "val_iou", "val_f1", "val_precision", "val_recall"))
        writer.writeheader()
        for epoch in range(1, args.epochs + 1):
            model.train()
            total_loss = 0.0
            for a, b, mask, _ in tqdm(loader, desc=f"Epoch {epoch}/{args.epochs}"):
                a, b, mask = a.to(device), b.to(device), mask.to(device)
                optimizer.zero_grad(set_to_none=True)
                logits = model(a, b)
                bce_loss = bce(logits, mask)
                probability = logits.sigmoid()
                dice_loss = 1 - ((2 * (probability * mask).sum((1, 2, 3)) + 1) /
                                 (probability.sum((1, 2, 3)) + mask.sum((1, 2, 3)) + 1)).mean()
                loss = bce_loss + dice_loss
                loss.backward()
                optimizer.step()
                total_loss += loss.item() * a.size(0)
            metrics = evaluate(model, val_loader, device)
            row = {"epoch": epoch, "loss": total_loss / len(train), **{f"val_{k}": metrics[k] for k in ("iou", "f1", "precision", "recall")}}
            writer.writerow(row)
            file.flush()
            print(json.dumps(row))
            if metrics["f1"] > best:
                best = metrics["f1"]
                best_epoch = epoch
                torch.save({"model": model.state_dict(), "architecture": args.architecture,
                            "base": args.base, "patch_size": args.patch_size,
                            "epoch": epoch, "val_metrics": metrics}, args.run / "best.pt")
    summary = {"architecture": args.architecture, "parameters": parameter_count,
               "training_seconds": round(time.perf_counter() - started, 2),
               "best_epoch": best_epoch, "best_val_f1_at_0.5": best,
               "train_tiles": len(train), "val_tiles": len(val)}
    (args.run / "training_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Best validation F1: {best:.4f}; checkpoint: {args.run / 'best.pt'}")


if __name__ == "__main__":
    main()
