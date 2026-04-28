"""
FedMorph Liver Segmentation — Federated Client
================================================

Each hospital PC runs one instance of this client.
The client:
  1. Scans its local data directory for CT volumes (image.npy + mask.npy)
  2. Auto-splits into train/val sets
  3. Receives global model from the server, trains locally
  4. Returns updated model weights + seg quality metrics to server

No central data distribution needed — each site only accesses its own data.

Usage:
  python main_client.py --data-dir D:\\data\\liver_ct --server-address 192.168.1.100:9000
"""

import argparse
import json
import math
import os
import random
import sys

os.environ["TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD"] = "1"

import flwr as fl
import numpy as np
import torch
import torch.nn as nn
import yaml
from collections import OrderedDict
from torch.utils.data import DataLoader

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "..", ".."))

from src.fed_core.fed_client import FedFlowerClient
from src.use_cases.liver_segmentation.models.segresnet_cirrhosis import build_model
from src.use_cases.liver_segmentation.utils.dataset import (
    LiverSeg9Dataset,
    auto_split,
    discover_patients,
    seg9_collate,
)
from src.use_cases.liver_segmentation.utils.loss import compute_loss
from src.use_cases.liver_segmentation.utils.metrics import (
    compute_morph_diversity,
    compute_per_segment_dice,
    evaluate,
)


def _is_norm(name: str) -> bool:
    return "norm" in name


class LiverSegmentationClient(FedFlowerClient):
    """FedMorph client for 9-segment liver CT segmentation."""

    def __init__(self, client_id: int, config: dict):
        super().__init__(client_id, config)

        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        # ── Model ──
        self.model = build_model(config, self.device)

        # ── Data: auto-discover from local directory ──
        data_dir = config["data_dir"]
        all_pids = discover_patients(data_dir)
        if not all_pids:
            raise RuntimeError(
                f"No patients found in {data_dir}. "
                "Each subfolder must contain image.npy and mask.npy."
            )

        train_ids, val_ids = auto_split(
            all_pids,
            train_ratio=config.get("train_ratio", 0.85),
            seed=config.get("seed", 42),
        )
        nc = config["num_classes"]

        self.train_ds = LiverSeg9Dataset(
            data_dir, train_ids, None,
            config["image_size"], config["volume_depth"],
            mode="train", num_classes=nc,
        )
        self.val_ds = LiverSeg9Dataset(
            data_dir, val_ids, None,
            config["image_size"], config["volume_depth"],
            mode="val", num_classes=nc,
        )

        nw = config.get("num_workers", 0)
        pin = torch.cuda.is_available()
        self.train_loader = DataLoader(
            self.train_ds, batch_size=config["batch_size"], shuffle=True,
            num_workers=nw, collate_fn=seg9_collate,
            pin_memory=pin, drop_last=True,
        )
        self.val_loader = DataLoader(
            self.val_ds, batch_size=config["batch_size"], shuffle=False,
            num_workers=nw, collate_fn=seg9_collate,
            pin_memory=pin,
        )

        # ── Mixed precision (CUDA only) ──
        self.scaler = (
            torch.amp.GradScaler("cuda") if torch.cuda.is_available() else None
        )

        # ── FL state ──
        self.local_norm_state: OrderedDict | None = None
        self.global_params_for_prox: dict | None = None
        self.current_round = 0

        print(f"[Client {client_id}] Device: {self.device}")
        print(f"[Client {client_id}] Data: {data_dir}")
        print(
            f"[Client {client_id}] Patients: {len(all_pids)} total "
            f"(train {len(self.train_ds)}, val {len(self.val_ds)})"
        )

    # ------------------------------------------------------------------
    # Flower interface
    # ------------------------------------------------------------------
    def get_model_parameters(self) -> list[np.ndarray]:
        return [v.cpu().numpy() for _, v in self.model.state_dict().items()]

    def set_model_parameters(self, parameters: list[np.ndarray]) -> None:
        keys = list(self.model.state_dict().keys())
        state_dict = OrderedDict()
        for k, v in zip(keys, parameters):
            state_dict[k] = torch.from_numpy(np.copy(v)).to(self.device)

        method = self.config.get("method", "FedMorph")

        if method == "FedBN" and self.local_norm_state is not None:
            for k, v in self.local_norm_state.items():
                state_dict[k] = v.to(self.device)

        self.model.load_state_dict(state_dict)

        if method == "FedProx":
            self.global_params_for_prox = {
                k: v.clone()
                for k, v in state_dict.items()
                if not k.endswith("num_batches_tracked")
            }

    def fit(
        self,
        parameters: list[np.ndarray],
        config: dict,
    ) -> tuple[list[np.ndarray], int, dict]:
        """Flower fit callback — train locally, return params + FedMorph metadata."""
        self.current_round = config.get("server_round", self.current_round + 1)
        print(
            f"\n[Client {self.client_id}] === Round {self.current_round} ==="
        )

        self.set_model_parameters(parameters)

        epochs = config.get(
            "local_epochs", self.config.get("local_epochs", 10)
        )
        train_metrics = self.train_model(epochs)

        method = self.config.get("method", "FedMorph")
        if method == "FedMorph":
            seg_dices = compute_per_segment_dice(
                self.model, self.val_loader, self.device,
                self.config["num_classes"],
            )
            morph_div = compute_morph_diversity(
                self.model, self.val_loader, self.device,
            )
            train_metrics["seg_dices_json"] = json.dumps(seg_dices.tolist())
            train_metrics["morph_diversity"] = float(morph_div)
            print(
                f"[Client {self.client_id}] "
                f"Seg Dice mean: {seg_dices.mean():.4f}, "
                f"Morph div: {morph_div:.6f}"
            )

        return (
            self.get_model_parameters(),
            self._get_dataset_size(),
            train_metrics,
        )

    def train_model(self, epochs: int) -> dict[str, float]:
        config = self.config
        method = config.get("method", "FedMorph")
        fl_rounds = config.get("fl_rounds", 50)

        lr_scale = 0.5 * (1 + math.cos(math.pi * self.current_round / fl_rounds))
        cur_lr = config["learning_rate"] * max(lr_scale, 0.3)

        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=cur_lr,
            weight_decay=config["weight_decay"],
        )

        fedprox_mu = config.get("fedprox_mu", 0.0) if method == "FedProx" else 0.0

        morph_start = int(fl_rounds * 0.5)
        if method == "FedMorph" and self.current_round >= morph_start:
            alpha = (self.current_round - morph_start) / max(fl_rounds - morph_start, 1)
            mc = 0.005 * alpha
        else:
            mc = 0.0
        local_config = {**config, "morph_coeff": mc, "cls_coeff": 0.0}

        total_loss = 0.0
        for epoch in range(epochs):
            global_epoch = self.current_round * epochs + epoch
            loss = self._train_one_epoch(
                optimizer, local_config, fedprox_mu, global_epoch
            )
            total_loss += loss

            if (epoch + 1) % max(1, epochs // 3) == 0:
                print(
                    f"[Client {self.client_id}] "
                    f"Epoch {epoch + 1}/{epochs}, Loss: {loss:.4f}, "
                    f"LR: {cur_lr:.6f}"
                )

        if method == "FedBN":
            self.local_norm_state = OrderedDict(
                (k, v.cpu().clone())
                for k, v in self.model.state_dict().items()
                if _is_norm(k)
            )

        avg_loss = total_loss / max(epochs, 1)
        return {"train_loss": float(avg_loss)}

    def _train_one_epoch(self, optimizer, config, fedprox_mu, epoch):
        self.model.train()
        total_loss, n = 0.0, 0

        warmup = config.get("seg_warmup_epochs", 30)
        cc = 0.0 if epoch < warmup else config.get("cls_coeff", 0.0)
        mc = 0.0 if epoch < warmup else config.get("morph_coeff", 0.0)

        gp = None
        if fedprox_mu > 0 and self.global_params_for_prox is not None:
            gp = self.global_params_for_prox

        use_amp = self.scaler is not None
        for batch in self.train_loader:
            images = batch["image"].to(self.device)
            masks = batch["mask"].to(self.device)
            cirrhosis = batch["cirrhosis"].to(self.device)

            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=use_amp):
                seg_logits, cls_logits, morph_feats, vol_ratios = self.model(
                    images
                )
                loss, _sl, _cl, _ml = compute_loss(
                    seg_logits, cls_logits, morph_feats, vol_ratios,
                    masks, cirrhosis, cc, mc,
                )

                if fedprox_mu > 0 and gp is not None:
                    prox = sum(
                        ((p - gp[k]) ** 2).sum()
                        for k, p in self.model.named_parameters()
                        if p.requires_grad and k in gp
                    )
                    loss = loss + (fedprox_mu / 2.0) * prox

            if use_amp:
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.scaler.step(optimizer)
                self.scaler.update()
            else:
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()

            total_loss += loss.item()
            n += 1

        return total_loss / max(n, 1)

    def evaluate_model(self) -> tuple[float, float, dict]:
        dv, hv, cls_m, vr_err = evaluate(
            self.model, self.val_loader, self.device, self.config["num_classes"]
        )

        mean_dice = float(torch.nanmean(dv).item())
        loss_proxy = 1.0 - mean_dice

        metrics = {
            "dice": mean_dice,
            "hd95": float(torch.nanmean(hv).item()),
            "cls_auc": float(cls_m.get("auc", 0.0)),
            "cls_acc": float(cls_m.get("acc", 0.0)),
            "vr_err": float(vr_err) if not np.isnan(vr_err) else 0.0,
        }
        print(
            f"[Client {self.client_id}] Eval — "
            f"Dice: {mean_dice:.4f} | "
            f"AUC: {metrics['cls_auc']:.4f} | "
            f"Acc: {metrics['cls_acc']:.4f}"
        )
        return loss_proxy, mean_dice, metrics

    def _get_dataset_size(self) -> int:
        return len(self.train_ds)


# ======================================================================
# Entry point
# ======================================================================
def load_config(config_path: str) -> dict:
    with open(config_path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser(
        description="FedMorph Liver Segmentation — Federated Client"
    )
    parser.add_argument(
        "--client-id", type=int, default=0,
        help="Client ID (for logging only, data is auto-discovered)",
    )
    parser.add_argument(
        "--server-address", type=str, default=None,
        help="Server address (overrides config)",
    )
    parser.add_argument(
        "--config", type=str,
        default="src/use_cases/liver_segmentation/configs/base.yaml",
    )
    parser.add_argument(
        "--data-dir", type=str, default=None,
        help="Local CT data directory (overrides config)",
    )
    args = parser.parse_args()

    config = load_config(args.config)

    config["data_dir"] = os.environ.get(
        "FEDMORPH_DATA_DIR", args.data_dir or config.get("data_dir", ".")
    )
    server_addr = (
        args.server_address
        or os.environ.get("FEDMORPH_SERVER_ADDRESS")
        or config["server_address"]
    )

    seed = config.get("seed", 42)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        props = torch.cuda.get_device_properties(0)
        print(f"GPU: {props.name}, {props.total_memory / 1024**3:.1f} GB")
        if props.major >= 8:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True

    print(f"FedMorph - Liver Segmentation Client {args.client_id}")
    print(f"  Method:  {config.get('method', 'FedMorph')}")
    print(f"  Data:    {config['data_dir']}")
    print(f"  Server:  {server_addr}")
    print("=" * 60)

    client = LiverSegmentationClient(args.client_id, config)

    print(f"Connecting to server at {server_addr} ...")
    print("=" * 60)

    fl.client.start_numpy_client(
        server_address=server_addr, client=client,
    )


if __name__ == "__main__":
    main()
