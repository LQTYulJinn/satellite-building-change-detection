"""Recover saved text and displayed PNGs from a downloaded Kaggle notebook."""

import argparse
import base64
import csv
import json
from pathlib import Path


def text_of(output):
    return "".join(output.get("text", []))


def object_after(text, marker):
    index = text.index(marker)
    start = text.rfind("{", 0, index)
    return json.JSONDecoder().raw_decode(text[start:])[0]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("notebook", type=Path)
    parser.add_argument("destination", type=Path)
    args = parser.parse_args()
    notebook = json.loads(args.notebook.read_text(encoding="utf-8"))
    args.destination.mkdir(parents=True, exist_ok=True)

    outputs = [output for cell in notebook["cells"] for output in cell.get("outputs", [])]
    streams = [text_of(output) for output in outputs if output.get("output_type") == "stream"]
    baseline = next(object_after(value, '"threshold"') for value in streams
                    if '"threshold"' in value and '"val"' in value and '"test"' in value)
    evaluation = next(object_after(value, '"split": "test"') for value in streams
                      if '"split": "test"' in value and '"checkpoint_epoch"' in value)
    audit = next(object_after(value, '"train"') for value in streams
                 if '"train"' in value and '"changed_pixels"' in value)
    for name, value in (("baseline.json", baseline), ("test_metrics.json", evaluation),
                        ("data_audit.json", audit)):
        (args.destination / name).write_text(json.dumps(value, indent=2), encoding="utf-8")

    epoch_rows = []
    for value in streams:
        for line in value.splitlines():
            if line.startswith('{"epoch":'):
                epoch_rows.append(json.loads(line))
    if epoch_rows:
        with (args.destination / "history.csv").open("w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=epoch_rows[0].keys())
            writer.writeheader()
            writer.writerows(epoch_rows)

    count = 0
    for output in outputs:
        value = output.get("data", {}).get("image/png")
        if not value:
            continue
        image_bytes = base64.b64decode("".join(value))
        name = f"comparison_{count + 1}.png"
        (args.destination / name).write_bytes(image_bytes)
        count += 1

    report = (
        "# Recovered Kaggle notebook outputs\n\n"
        f"Source: {args.notebook.name}. These files were extracted from saved notebook "
        "cell outputs after the Kaggle draft session restarted. The checkpoint `best.pt` "
        "is not embedded in the notebook and could not be recovered here.\n\n"
        f"Training epochs recovered: {len(epoch_rows)}. Comparison images: {count}.\n\n"
        f"Test IoU: {evaluation['metrics']['iou']:.4f}; test F1: "
        f"{evaluation['metrics']['f1']:.4f}.\n"
    )
    (args.destination / "README.md").write_text(report, encoding="utf-8")
    print(report)


if __name__ == "__main__":
    main()
