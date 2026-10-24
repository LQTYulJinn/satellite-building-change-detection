"""Run the whole pipeline on tiny synthetic data to catch integration errors.

Synthetic scores are intentionally discarded and must never be reported as project results.
"""

import os
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image


def main():
    project = Path(__file__).resolve().parent
    with tempfile.TemporaryDirectory(dir=project) as directory:
        root = Path(directory)
        data = root / "data"
        for index, split in enumerate(("train", "val", "test")):
            for part in ("A", "B", "label"):
                (data / split / part).mkdir(parents=True)
            rng = np.random.default_rng(index + 1)
            before = rng.integers(0, 120, (32, 32, 3), dtype=np.uint8)
            after = before.copy()
            after[8:20, 8:20] = (220, 220, 220)
            mask = np.zeros((32, 32), dtype=np.uint8)
            mask[8:20, 8:20] = 255
            name = f"{split}.png"
            Image.fromarray(before).save(data / split / "A" / name)
            Image.fromarray(after).save(data / split / "B" / name)
            Image.fromarray(mask).save(data / split / "label" / name)

        env = os.environ.copy()
        env["MPLCONFIGDIR"] = str(root / "mpl")
        runs = root / "runs"

        def run(*args):
            subprocess.run([sys.executable, str(project / args[0]), *map(str, args[1:])],
                           cwd=project, env=env, check=True, stdout=subprocess.DEVNULL)

        run("audit_data.py", "--data", data, "--output", runs / "data_audit.json")
        run("baseline.py", "--data", data, "--patch-size", 32, "--output", runs / "baseline.json")
        for architecture in ("early", "siamese"):
            destination = runs / architecture
            run("train.py", "--data", data, "--run", destination, "--architecture", architecture,
                "--epochs", 1, "--batch-size", 1, "--patch-size", 32, "--base", 4)
            run("evaluate.py", "--data", data, "--checkpoint", destination / "best.pt",
                "--batch-size", 1, "--examples", 1)
            assert (destination / "test" / "metrics.json").exists()
            assert len(list((destination / "test" / "examples").glob("*.png"))) == 1
        run("summarize.py", "--runs", runs)
        assert (runs / "results.md").exists()
    print("End-to-end smoke test passed; synthetic data and scores were deleted.")


if __name__ == "__main__":
    main()
