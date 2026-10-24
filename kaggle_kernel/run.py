"""Kaggle GPU run of the real LEVIR-CD benchmark."""

import json
import subprocess
import sys
from pathlib import Path

import torch


REPOSITORY = "https://github.com/LQTYulJinn/satellite-building-change-detection.git"
REVISION = "b2eb1b9"
PROJECT = Path("/kaggle/working/satellite-building-change-detection")
INPUT_ROOT = Path("/kaggle/input")


def find_data():
    matches = [folder.parent for folder in INPUT_ROOT.rglob("train")
               if folder.is_dir() and (folder / "A").is_dir()
               and (folder.parent / "val" / "A").is_dir()
               and (folder.parent / "test" / "A").is_dir()]
    if len(matches) != 1:
        visible = [str(p) for p in INPUT_ROOT.glob("*/*")][:30]
        raise FileNotFoundError(f"Expected one LEVIR-CD root, found {matches}; visible paths: {visible}")
    return matches[0]


def run(*arguments):
    print("Running:", " ".join(map(str, arguments)), flush=True)
    subprocess.run(list(map(str, arguments)), check=True, cwd=PROJECT)


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("Kaggle notebook must have a GPU accelerator enabled")
    data = find_data()
    if not PROJECT.exists():
        subprocess.run(["git", "clone", REPOSITORY, str(PROJECT)], check=True)
    subprocess.run(["git", "checkout", REVISION], check=True, cwd=PROJECT)

    print(json.dumps({"data": str(data), "revision": REVISION,
                      "gpu": torch.cuda.get_device_name(0),
                      "torch": torch.__version__}, indent=2), flush=True)
    run(sys.executable, "audit_data.py", "--data", data)
    run(sys.executable, "baseline.py", "--data", data)
    for architecture in ("early", "siamese"):
        destination = Path("runs") / architecture
        run(sys.executable, "train.py", "--data", data, "--run", destination,
            "--architecture", architecture, "--epochs", "30", "--batch-size", "8",
            "--base", "32", "--workers", "4", "--seed", "42")
        run(sys.executable, "evaluate.py", "--data", data,
            "--checkpoint", destination / "best.pt", "--split", "test")
    run(sys.executable, "summarize.py")
    print((PROJECT / "runs" / "results.md").read_text(encoding="utf-8"), flush=True)


if __name__ == "__main__":
    main()
