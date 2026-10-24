"""Create a results table from actual experiment outputs; never invent metrics."""

import argparse
import json
from pathlib import Path


def read(path):
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def main():
    parser = argparse.ArgumentParser(description="Summarize completed experiments")
    parser.add_argument("--runs", type=Path, default=Path("runs"))
    args = parser.parse_args()
    baseline = read(args.runs / "baseline.json")
    early = read(args.runs / "early" / "test" / "metrics.json")
    siamese = read(args.runs / "siamese" / "test" / "metrics.json")
    early_training = read(args.runs / "early" / "training_summary.json")
    siamese_training = read(args.runs / "siamese" / "training_summary.json")
    if not any((baseline, early, siamese)):
        raise FileNotFoundError("No completed experiment output found in runs/")

    lines = ["# Experimental results", "", "All values below are read from saved experiment outputs.", "",
             "| Method | Test IoU | Test F1 | Precision | Recall | Threshold | Parameters | Train time (min) |",
             "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for label, result, training in (("RGB difference", baseline, None),
                                    ("Early fusion U-Net", early, early_training),
                                    ("Siamese U-Net", siamese, siamese_training)):
        if result is None:
            continue
        metrics = result["test"] if label == "RGB difference" else result["metrics"]
        parameters = f"{training['parameters']:,}" if training else "—"
        minutes = f"{training['training_seconds'] / 60:.1f}" if training else "—"
        lines.append(f"| {label} | {metrics['iou']:.4f} | {metrics['f1']:.4f} | "
                     f"{metrics['precision']:.4f} | {metrics['recall']:.4f} | "
                     f"{result['threshold']:.2f} | {parameters} | {minutes} |")
    lines.extend(["", "Thresholds were chosen on validation data only. Test metrics are global pixel counts (micro average).", ""])
    target = args.runs / "results.md"
    target.write_text("\n".join(lines), encoding="utf-8")
    print(target.read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
