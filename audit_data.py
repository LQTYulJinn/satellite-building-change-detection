"""Check LEVIR-CD triplets, split overlap and change-pixel prevalence."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image


def image_hash(path):
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def audit_split(root, split):
    folder = root / split
    paths = {part: {p.stem: p for p in (folder / part).glob("*.png")}
             for part in ("A", "B", "label")}
    names = set(paths["A"])
    if not names or any(set(paths[part]) != names for part in ("B", "label")):
        raise ValueError(f"Missing or unmatched A/B/label PNG files in {folder}")
    changed = total = 0
    hashes = set()
    for name in sorted(names):
        with Image.open(paths["A"][name]) as image:
            size = image.size
        with Image.open(paths["B"][name]) as image:
            if image.size != size:
                raise ValueError(f"A/B size mismatch: {split}/{name}")
        with Image.open(paths["label"][name]) as image:
            if image.size != size:
                raise ValueError(f"Label size mismatch: {split}/{name}")
            mask = np.asarray(image.convert("L"))
        levels = set(np.unique(mask).tolist())
        if not levels.issubset({0, 1, 255}):
            raise ValueError(f"Label must be binary 0/1 or 0/255: {split}/{name}, levels={sorted(levels)[:10]}")
        changed += int((mask > 0).sum())
        total += mask.size
        hashes.add((image_hash(paths["A"][name]), image_hash(paths["B"][name])))
    return {"images": len(names), "pixels": total, "changed_pixels": changed,
            "change_prevalence": changed / total}, names, hashes


def main():
    parser = argparse.ArgumentParser(description="Audit LEVIR-CD splits before training")
    parser.add_argument("--data", type=Path, default=Path("data/LEVIR-CD"))
    parser.add_argument("--output", type=Path, default=Path("runs/data_audit.json"))
    args = parser.parse_args()
    results = {}
    split_names = {}
    split_hashes = {}
    for split in ("train", "val", "test"):
        results[split], split_names[split], split_hashes[split] = audit_split(args.data, split)
    for left, right in (("train", "val"), ("train", "test"), ("val", "test")):
        duplicate_pairs = split_hashes[left] & split_hashes[right]
        if duplicate_pairs:
            raise ValueError(f"Possible split leakage between {left} and {right}: "
                             f"{len(duplicate_pairs)} identical image pairs")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
