import torch


def counts(prediction, target):
    prediction = prediction.bool()
    target = target.bool()
    return (
        int((prediction & target).sum()),
        int((prediction & ~target).sum()),
        int((~prediction & target).sum()),
    )


def scores(tp, fp, fn):
    def divide(num, den):
        return num / den if den else 0.0

    return {
        "iou": divide(tp, tp + fp + fn),
        "f1": divide(2 * tp, 2 * tp + fp + fn),
        "precision": divide(tp, tp + fp),
        "recall": divide(tp, tp + fn),
        "tp": tp,
        "fp": fp,
        "fn": fn,
    }


@torch.no_grad()
def evaluate(model, loader, device, threshold=0.5):
    model.eval()
    tp = fp = fn = 0
    for a, b, mask, _ in loader:
        probability = model(a.to(device), b.to(device)).sigmoid()
        x, y, z = counts(probability >= threshold, mask.to(device) > 0.5)
        tp, fp, fn = tp + x, fp + y, fn + z
    return scores(tp, fp, fn)
