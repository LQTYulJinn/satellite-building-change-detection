import argparse
import csv
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from change_detection.data import LevirDataset
from change_detection.metrics import counts, scores
from change_detection.model import ChangeUNet
from change_detection.visualize import save_comparison


@torch.no_grad()
def choose_threshold(model, loader, device):
    """Select one operating point using validation pixels only."""
    candidates = [i / 100 for i in range(10, 91, 5)]
    totals = {t: [0, 0, 0] for t in candidates}
    model.eval()
    for a, b, mask, _ in loader:
        probability = model(a.to(device), b.to(device)).sigmoid()
        truth = mask.to(device) > 0.5
        for threshold in candidates:
            result = counts(probability >= threshold, truth)
            totals[threshold] = [old + new for old, new in zip(totals[threshold], result)]
    threshold = max(candidates, key=lambda t: scores(*totals[t])["f1"])
    return threshold, scores(*totals[threshold])


@torch.no_grad()
def score_split(model, loader, device, threshold):
    model.eval()
    totals = [0, 0, 0]
    rows = []
    for a, b, mask, names in loader:
        probability = model(a.to(device), b.to(device)).sigmoid()
        for index, name in enumerate(names):
            sample_counts = counts(probability[index] >= threshold, mask[index].to(device) > 0.5)
            totals = [old + new for old, new in zip(totals, sample_counts)]
            rows.append({"tile": name, **scores(*sample_counts)})
    return scores(*totals), rows


def select_examples(rows, maximum):
    """Include both error types and representative successes; avoid cherry-picking."""
    if maximum <= 0:
        return []
    choices = []
    rankings = [
        sorted(rows, key=lambda r: r["fn"], reverse=True),
        sorted(rows, key=lambda r: r["fp"], reverse=True),
        sorted((r for r in rows if r["tp"] + r["fn"]), key=lambda r: r["f1"]),
        sorted((r for r in rows if r["tp"] + r["fn"]), key=lambda r: r["f1"], reverse=True),
        sorted((r for r in rows if not r["tp"] and not r["fn"]), key=lambda r: r["fp"], reverse=True),
    ]
    for ranking in rankings:
        for row in ranking:
            if row["tile"] not in choices:
                choices.append(row["tile"])
                break
    for row in rows:
        if len(choices) >= maximum:
            break
        if row["tile"] not in choices:
            choices.append(row["tile"])
    return choices[:maximum]


def main():
    parser = argparse.ArgumentParser(description="Evaluate a saved LEVIR-CD model")
    parser.add_argument("--data", type=Path, default=Path("data/LEVIR-CD"))
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--split", choices=("val", "test"), default="test")
    parser.add_argument("--threshold", type=float, help="Use a preselected threshold; omit to tune on validation")
    parser.add_argument("--examples", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=8)
    args = parser.parse_args()
    if args.threshold is not None and not 0 < args.threshold < 1:
        raise ValueError("--threshold must be between 0 and 1")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=True)
    model = ChangeUNet(checkpoint["base"], checkpoint["architecture"]).to(device)
    model.load_state_dict(checkpoint["model"])
    patch_size = checkpoint["patch_size"]
    if args.threshold is None:
        validation = DataLoader(LevirDataset(args.data, "val", patch_size), batch_size=args.batch_size)
        threshold, val_metrics = choose_threshold(model, validation, device)
        source = "validation F1 search (0.10 to 0.90, step 0.05)"
    else:
        threshold, val_metrics = args.threshold, None
        source = "user supplied"
    dataset = LevirDataset(args.data, args.split, patch_size)
    loader = DataLoader(dataset, batch_size=args.batch_size)
    metrics, rows = score_split(model, loader, device, threshold)
    output = args.checkpoint.parent / args.split
    output.mkdir(parents=True, exist_ok=True)
    report = {"split": args.split, "architecture": checkpoint["architecture"],
              "checkpoint_epoch": checkpoint["epoch"], "threshold": threshold,
              "threshold_source": source, "validation_at_selected_threshold": val_metrics,
              "metrics": metrics, "tiles": len(rows)}
    (output / "metrics.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    with (output / "per_tile.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=("tile", "iou", "f1", "precision", "recall", "tp", "fp", "fn"))
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps(report, indent=2))

    model.eval()
    selected = select_examples(rows, args.examples)
    tile_lookup = {dataset.tiles[i][5] + f"_{dataset.tiles[i][3]}_{dataset.tiles[i][4]}": i for i in range(len(dataset))}
    with torch.no_grad():
        for name in selected:
            a, b, mask, _ = dataset[tile_lookup[name]]
            probability = model(a[None].to(device), b[None].to(device)).sigmoid()[0].cpu()
            save_comparison(a, b, mask, probability, output / "examples" / f"{name}.png", threshold)
    print(f"Saved {len(selected)} comparison figures to {output / 'examples'}")


if __name__ == "__main__":
    main()
