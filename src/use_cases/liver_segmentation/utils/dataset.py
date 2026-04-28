"""Liver segmentation dataset for federated learning."""

import os
import random

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from tqdm import tqdm


def discover_patients(data_dir: str) -> list[str]:
    """Scan data_dir for patient folders containing image.npy + mask.npy."""
    pids = []
    for name in sorted(os.listdir(data_dir)):
        patient_dir = os.path.join(data_dir, name)
        if not os.path.isdir(patient_dir):
            continue
        if (
            os.path.isfile(os.path.join(patient_dir, "image.npy"))
            and os.path.isfile(os.path.join(patient_dir, "mask.npy"))
        ):
            pids.append(name)
    return pids


def auto_split(
    patient_ids: list[str],
    train_ratio: float = 0.85,
    seed: int = 42,
) -> tuple[list[str], list[str]]:
    """Split patient IDs into train / val sets."""
    rng = random.Random(seed)
    ids = list(patient_ids)
    rng.shuffle(ids)
    n_train = max(1, int(round(len(ids) * train_ratio)))
    return ids[:n_train], ids[n_train:]


def resize_volume(vol: np.ndarray, size: int, is_gt: bool = False) -> np.ndarray:
    """Resize a 3-D volume (D, H, W) to (D, size, size) with square padding."""
    H, W = vol.shape[1], vol.shape[2]
    if H > W:
        pad = (H - W) // 2
        vol = np.pad(vol, ((0, 0), (0, 0), (pad, pad)), mode="constant")
    elif H < W:
        pad = (W - H) // 2
        vol = np.pad(vol, ((0, 0), (pad, pad), (0, 0)), mode="constant")
    D = vol.shape[0]
    out = np.zeros((D, size, size), dtype=vol.dtype)
    interp = cv2.INTER_NEAREST if is_gt else cv2.INTER_CUBIC
    for i in range(D):
        sl = vol[i].astype(np.uint8 if is_gt else np.float32)
        out[i] = cv2.resize(sl, (size, size), interpolation=interp)
    return out


class LiverSeg9Dataset(Dataset):
    """9-segment liver CT volume dataset with cirrhosis labels."""

    def __init__(
        self,
        combined_dir: str,
        patient_ids: list,
        cirrhosis_dict: dict | None = None,
        image_size: int = 128,
        volume_depth: int = 64,
        mode: str = "train",
        num_classes: int = 9,
    ):
        self.image_size = image_size
        self.volume_depth = volume_depth
        self.mode = mode
        self.num_classes = num_classes
        self.samples: list[dict] = []
        if cirrhosis_dict is None:
            cirrhosis_dict = {}

        for pid in tqdm(patient_ids, desc=f"Building {mode}", leave=False):
            vol_dir = os.path.join(combined_dir, pid)
            img_path = os.path.join(vol_dir, "image.npy")
            mask_path = os.path.join(vol_dir, "mask.npy")
            if not os.path.exists(img_path):
                continue
            mask = np.load(mask_path)
            seg9 = mask[1 : num_classes + 1]
            active = [c for c in range(num_classes) if seg9[c].sum() > 0]
            cirrhosis_label = cirrhosis_dict.get(pid, -1)
            if isinstance(cirrhosis_label, str):
                cirrhosis_label = int(cirrhosis_label)
            self.samples.append(
                {
                    "img_path": img_path,
                    "mask_path": mask_path,
                    "pid": pid,
                    "active_classes": active,
                    "cirrhosis": cirrhosis_label,
                }
            )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        s = self.samples[idx]
        img = np.load(s["img_path"]).astype(np.float32)
        mask_all = np.load(s["mask_path"])
        seg = mask_all[1 : self.num_classes + 1].astype(np.uint8)

        D = img.shape[0]
        vd = self.volume_depth

        if self.mode == "train" and D > vd:
            start = random.randint(0, D - vd)
            img = img[start : start + vd]
            seg = seg[:, start : start + vd]
        elif D > vd:
            start = (D - vd) // 2
            img = img[start : start + vd]
            seg = seg[:, start : start + vd]

        cur_d = img.shape[0]
        if cur_d < vd:
            pad_d = vd - cur_d
            img = np.pad(img, ((0, pad_d), (0, 0), (0, 0)), mode="constant")
            seg = np.pad(
                seg, ((0, 0), (0, pad_d), (0, 0), (0, 0)), mode="constant"
            )

        img_r = resize_volume(img, self.image_size, is_gt=False) / 255.0
        seg_r = np.stack(
            [
                resize_volume(seg[c], self.image_size, is_gt=True)
                for c in range(self.num_classes)
            ],
            axis=0,
        )

        return {
            "image": torch.from_numpy(img_r).float().unsqueeze(0),
            "mask": torch.from_numpy(seg_r.astype(np.float32)),
            "cirrhosis": torch.tensor(s["cirrhosis"], dtype=torch.float32),
            "pid": s["pid"],
        }


def seg9_collate(batch: list[dict]) -> dict:
    """Custom collate for LiverSeg9Dataset."""
    return {
        "image": torch.stack([b["image"] for b in batch]),
        "mask": torch.stack([b["mask"] for b in batch]),
        "cirrhosis": torch.stack([b["cirrhosis"] for b in batch]),
        "pids": [b["pid"] for b in batch],
    }
