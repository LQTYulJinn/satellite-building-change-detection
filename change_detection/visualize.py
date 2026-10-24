from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def save_comparison(a, b, mask, probability, destination: str | Path, threshold=0.5):
    a = a.permute(1, 2, 0).cpu().numpy()
    b = b.permute(1, 2, 0).cpu().numpy()
    truth = mask.squeeze().cpu().numpy() > 0.5
    pred = probability.squeeze().cpu().numpy() >= threshold
    fig, axes = plt.subplots(1, 5, figsize=(17, 4))
    for ax, image, title in zip(
        axes, (a, b, truth, pred, None),
        ("Before", "After", "Ground truth", "Prediction", "Errors"),
    ):
        if image is not None:
            ax.imshow(image, cmap="gray" if image.ndim == 2 else None, vmin=0, vmax=1)
        else:
            errors = b.copy() * 0.35
            errors[(pred == 1) & (truth == 1)] = (0.1, 0.85, 0.1)
            errors[(pred == 1) & (truth == 0)] = (1.0, 0.2, 0.1)
            errors[(pred == 0) & (truth == 1)] = (0.2, 0.3, 1.0)
            ax.imshow(errors)
        ax.set_title(title)
        ax.axis("off")
    fig.suptitle("Errors: green = TP, red = FP, blue = FN")
    fig.tight_layout()
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(destination, dpi=150)
    plt.close(fig)
