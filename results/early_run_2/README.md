# LEVIR-CD early-fusion U-Net: second execution

Source: `levir_cd_results.zip` supplied on 2026-10-25. The archive was checked for corruption, and its `best.pt` loaded successfully with `torch.load(..., weights_only=True)`. Original dataset files are not included.

This is a **repeat of the existing experiment**, not a new model or training method: early-fusion U-Net, 30 epochs, batch size 8, base width 32, AdamW learning rate 0.001, seed 42, 256×256 patches, BCE + Dice, Tesla T4. There were 7,120 train, 1,024 validation, and 2,048 test tiles.

The validation-selected checkpoint is epoch 29. The prediction threshold 0.30 was selected on validation. Test results: IoU **0.595058**, F1 **0.746127**, precision **0.778412**, recall **0.716414**; TP 4,898,411, FP 1,394,414, FN 1,938,993. These are global pixel counts. `metrics.json` is the numeric source of truth.

The checkpoint is kept locally at `runs/early/best.pt` and inside the supplied ZIP, outside ordinary Git history. Its SHA-256 is `d5c6ed046a60f4699b14e817c71186b80c2d3f92fba0f8d94d37f17c6d137199`. The folder here tracks the reproducible metadata, per-tile scores, history, and eight generated comparison images.

The first execution is preserved in [`../../recovered_results/`](../../recovered_results/). Both test results have already been observed; future experiments on this test split should be described as exploratory comparisons.
