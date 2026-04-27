"""Evaluation metrics for liver segmentation."""

import numpy as np
import torch
from monai.metrics import DiceMetric, HausdorffDistanceMetric
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score


@torch.no_grad()
def compute_per_segment_dice(
    model: torch.nn.Module,
    loader,
    device: torch.device,
    num_classes: int = 9,
) -> np.ndarray:
    """Per-segment Dice scores used for FedMorph quality-weighted aggregation."""
    model.eval()
    dm = DiceMetric(include_background=True, reduction="mean")
    totals = np.zeros(num_classes)
    counts = np.zeros(num_classes)

    use_amp = device.type == "cuda"
    for batch in loader:
        images = batch["image"].to(device)
        masks = batch["mask"]
        with torch.amp.autocast("cuda", enabled=use_amp):
            seg_logits, _, _, _ = model(images)
        seg_pred = (seg_logits.sigmoid().cpu() > 0.5).float()
        B = images.shape[0]
        for b in range(B):
            for c in range(num_classes):
                gt = masks[b, c]
                if gt.sum() == 0:
                    continue
                dm.reset()
                dm(
                    y_pred=seg_pred[b, c].unsqueeze(0).unsqueeze(0),
                    y=gt.unsqueeze(0).unsqueeze(0),
                )
                totals[c] += dm.aggregate().item()
                counts[c] += 1

    for c in range(num_classes):
        if counts[c] > 0:
            totals[c] /= counts[c]
    return totals


@torch.no_grad()
def compute_morph_diversity(
    model: torch.nn.Module,
    loader,
    device: torch.device,
) -> float:
    """Volume-ratio variance used for FedMorph diversity-weighted aggregation."""
    model.eval()
    all_vr: list[torch.Tensor] = []
    use_amp = device.type == "cuda"
    for batch in loader:
        images = batch["image"].to(device)
        with torch.amp.autocast("cuda", enabled=use_amp):
            _, _, _, vol_ratios = model(images)
        all_vr.append(vol_ratios.cpu())
    if not all_vr:
        return 0.0
    vr = torch.cat(all_vr, dim=0)
    return float(vr.var(dim=0).mean().item())


@torch.no_grad()
def evaluate(
    model: torch.nn.Module,
    loader,
    device: torch.device,
    num_classes: int = 9,
) -> tuple:
    """Full evaluation: per-segment Dice/HD95, classification, VR error.

    Returns (dice_per_class, hd95_per_class, cls_metrics_dict, mean_vr_err).
    """
    model.eval()
    dm = DiceMetric(include_background=True, reduction="mean")
    hm = HausdorffDistanceMetric(
        include_background=True, percentile=95.0, reduction="mean"
    )

    pcd: list[list[float]] = [[] for _ in range(num_classes)]
    phd: list[list[float]] = [[] for _ in range(num_classes)]
    cls_preds: list[float] = []
    cls_labels: list[int] = []
    vr_errors: list[torch.Tensor] = []

    use_amp = device.type == "cuda"
    for batch in loader:
        images = batch["image"].to(device)
        masks = batch["mask"]
        cirrhosis = batch["cirrhosis"]

        with torch.amp.autocast("cuda", enabled=use_amp):
            seg_logits, cls_logits, _morph, vol_ratios = model(images)

        seg_pred = (seg_logits.sigmoid().cpu() > 0.5).float()
        pred_vr = vol_ratios.cpu()
        gt_vol = masks.sum(dim=(2, 3, 4))
        gt_total = gt_vol.sum(dim=1, keepdim=True).clamp(min=1e-6)
        gt_vr = gt_vol / gt_total
        vr_errors.append((pred_vr - gt_vr).abs().mean(dim=1))

        B = images.shape[0]
        for b in range(B):
            for c in range(num_classes):
                gt = masks[b, c]
                if gt.sum() == 0:
                    continue
                pred = seg_pred[b, c]
                g5 = gt.unsqueeze(0).unsqueeze(0)
                p5 = pred.unsqueeze(0).unsqueeze(0)
                dm.reset()
                dm(y_pred=p5, y=g5)
                pcd[c].append(dm.aggregate().item())
                try:
                    hm.reset()
                    hm(y_pred=p5, y=g5)
                    phd[c].append(hm.aggregate().item())
                except Exception:
                    pass

            if cirrhosis[b] >= 0:
                p_val = cls_logits[b].float().sigmoid().cpu().item()
                cls_preds.append(p_val)
                cls_labels.append(int(cirrhosis[b].item()))

    dv = torch.full((num_classes,), float("nan"))
    hv = torch.full((num_classes,), float("nan"))
    for c in range(num_classes):
        if pcd[c]:
            dv[c] = np.nanmean(pcd[c])
        if phd[c]:
            hv[c] = np.nanmean(phd[c])

    cls_m: dict[str, float] = {
        "auc": float("nan"),
        "acc": float("nan"),
        "f1": float("nan"),
    }
    valid = [
        (l, p)
        for l, p in zip(cls_labels, cls_preds)
        if not (np.isnan(p) or np.isinf(p))
    ]
    if len(valid) >= 2:
        cls_labels_v, cls_preds_v = zip(*valid)
        if len(set(cls_labels_v)) > 1:
            cls_m["auc"] = roc_auc_score(cls_labels_v, cls_preds_v)
        cls_bin = [int(p > 0.5) for p in cls_preds_v]
        cls_m["acc"] = accuracy_score(cls_labels_v, cls_bin)
        cls_m["f1"] = f1_score(cls_labels_v, cls_bin, zero_division=0)

    mean_vr_err = float("nan")
    if vr_errors:
        mean_vr_err = float(torch.cat(vr_errors).mean().item())

    return dv, hv, cls_m, mean_vr_err
