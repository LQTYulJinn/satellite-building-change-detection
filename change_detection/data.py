from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


class LevirDataset(Dataset):
    """Read LEVIR-CD A/B/label triplets and tile them without crossing splits."""

    def __init__(self, root: str | Path, split: str, patch_size: int = 256, augment: bool = False):
        self.split_dir = Path(root) / split
        self.patch_size = patch_size
        self.augment = augment
        for folder in ("A", "B", "label"):
            if not (self.split_dir / folder).is_dir():
                raise FileNotFoundError(f"Missing {self.split_dir / folder}")

        a_files = {p.stem: p for p in (self.split_dir / "A").glob("*.png")}
        b_files = {p.stem: p for p in (self.split_dir / "B").glob("*.png")}
        l_files = {p.stem: p for p in (self.split_dir / "label").glob("*.png")}
        if not a_files or a_files.keys() != b_files.keys() or a_files.keys() != l_files.keys():
            raise ValueError(f"A, B and label must contain matching PNG filenames in {self.split_dir}")

        self.tiles = []
        for stem in sorted(a_files):
            with Image.open(a_files[stem]) as im:
                width, height = im.size
            for other in (b_files[stem], l_files[stem]):
                with Image.open(other) as im:
                    if im.size != (width, height):
                        raise ValueError(f"Image dimensions differ for {stem}")
            for top in range(0, height, patch_size):
                for left in range(0, width, patch_size):
                    if top + patch_size <= height and left + patch_size <= width:
                        self.tiles.append((a_files[stem], b_files[stem], l_files[stem], left, top, stem))
        if not self.tiles:
            raise ValueError(f"No complete {patch_size}x{patch_size} tiles in {self.split_dir}")

    def __len__(self):
        return len(self.tiles)

    def __getitem__(self, index):
        a_path, b_path, label_path, left, top, stem = self.tiles[index]
        box = (left, top, left + self.patch_size, top + self.patch_size)
        with Image.open(a_path) as image:
            a = np.asarray(image.convert("RGB").crop(box), dtype=np.float32) / 255.0
        with Image.open(b_path) as image:
            b = np.asarray(image.convert("RGB").crop(box), dtype=np.float32) / 255.0
        with Image.open(label_path) as image:
            mask = np.asarray(image.convert("L").crop(box), dtype=np.uint8) > 0

        if self.augment:
            if torch.rand(()) < 0.5:
                a, b, mask = np.flip(a, 1), np.flip(b, 1), np.flip(mask, 1)
            if torch.rand(()) < 0.5:
                a, b, mask = np.flip(a, 0), np.flip(b, 0), np.flip(mask, 0)
            turns = int(torch.randint(0, 4, ()).item())
            a, b, mask = np.rot90(a, turns), np.rot90(b, turns), np.rot90(mask, turns)

        a = torch.from_numpy(np.ascontiguousarray(a.transpose(2, 0, 1)))
        b = torch.from_numpy(np.ascontiguousarray(b.transpose(2, 0, 1)))
        mask = torch.from_numpy(np.ascontiguousarray(mask[None].astype(np.float32)))
        return a, b, mask, f"{stem}_{left}_{top}"
