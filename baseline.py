import argparse
import json
from pathlib import Path

from change_detection.data import LevirDataset
from change_detection.metrics import counts, scores


def measure(dataset, thresholds):
    totals = {threshold: [0, 0, 0] for threshold in thresholds}
    for a, b, mask, _ in dataset:
        difference = (a - b).abs().mean(dim=0, keepdim=True)
        for threshold in thresholds:
            result = counts(difference >= threshold, mask > 0.5)
            totals[threshold] = [old + new for old, new in zip(totals[threshold], result)]
    return {threshold: scores(*totals[threshold]) for threshold in thresholds}


def main():
    parser = argparse.ArgumentParser(description="RGB difference baseline; tune threshold on validation only")
    parser.add_argument("--data", type=Path, default=Path("data/LEVIR-CD"))
    parser.add_argument("--patch-size", type=int, default=256)
    parser.add_argument("--output", type=Path, default=Path("runs/baseline.json"))
    args = parser.parse_args()
    val = LevirDataset(args.data, "val", args.patch_size)
    candidates = [i / 100 for i in range(5, 51, 5)]
    validation = measure(val, candidates)
    threshold = max(candidates, key=lambda t: validation[t]["f1"])
    result = {"threshold": threshold, "val": validation[threshold],
              "test": measure(LevirDataset(args.data, "test", args.patch_size), [threshold])[threshold]}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
