"""
FedMorph Liver Segmentation — Federated Server
================================================

Starts the Flower gRPC server with FedMorph aggregation strategy.
Waits for ``min_clients`` to connect, then runs ``fl_rounds`` rounds.

Usage:
  python main_server.py
  python main_server.py --config configs/base.yaml
"""

import argparse
import math
import os
import sys

os.environ["TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD"] = "1"

import torch
import yaml

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "..", ".."))

from src.fed_core.fed_server import FedFlowerServer
from src.fed_core.fedmorph_strategy import FedMorphStrategy
from src.use_cases.liver_segmentation.models.segresnet_cirrhosis import build_model


def load_config(
    config_path: str = "src/use_cases/liver_segmentation/configs/base.yaml",
) -> dict:
    with open(config_path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def get_model_state_keys(config: dict) -> list[str]:
    """Create a temporary model on CPU to get ordered state_dict keys."""
    device = torch.device("cpu")
    model = build_model(config, device)
    keys = list(model.state_dict().keys())
    del model
    return keys


def main():
    parser = argparse.ArgumentParser(
        description="FedMorph Liver Segmentation — Federated Server"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="src/use_cases/liver_segmentation/configs/base.yaml",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    method = config.get("method", "FedMorph")
    fl_rounds = config["fl_rounds"]
    min_clients = config["min_clients"]
    local_epochs = config["local_epochs"]

    print("FedMorph - Liver Segmentation Server")
    print("=" * 60)
    print(f"  Method:       {method}")
    print(f"  Rounds:       {fl_rounds}")
    print(f"  Min Clients:  {min_clients}")
    print(f"  Local Epochs: {local_epochs}")
    print(f"  Model:        SegResNetWithCirrhosis "
          f"(filters={config['init_filters']})")

    model_keys = get_model_state_keys(config)
    print(f"  Param tensors: {len(model_keys)}")

    def fit_config_fn(server_round: int) -> dict:
        lr_scale = 0.5 * (1 + math.cos(math.pi * server_round / fl_rounds))
        return {
            "server_round": server_round,
            "local_epochs": local_epochs,
            "method": method,
            "lr_scale": lr_scale,
        }

    def evaluate_metrics_agg_fn(metrics: list) -> dict:
        if not metrics:
            return {}
        dices = [n * m.get("dice", 0.0) for n, m in metrics]
        examples = [n for n, _ in metrics]
        total = sum(examples)
        if total == 0:
            return {}
        avg_dice = sum(dices) / total
        print(f"  [Server] Aggregated eval — Dice: {avg_dice:.4f}")
        return {"dice": avg_dice}

    strategy = FedMorphStrategy(
        model_state_keys=model_keys,
        num_classes=config["num_classes"],
        method=method,
        fraction_fit=1.0,
        fraction_evaluate=1.0,
        min_fit_clients=min_clients,
        min_evaluate_clients=min_clients,
        min_available_clients=min_clients,
        on_fit_config_fn=fit_config_fn,
        evaluate_metrics_aggregation_fn=evaluate_metrics_agg_fn,
    )

    server = FedFlowerServer(
        num_rounds=fl_rounds,
        min_clients=min_clients,
        strategy=strategy,
        config=config,
    )

    print("=" * 60)
    server_address = config.get("server_address", "0.0.0.0:9000")
    print(f"Listening on {server_address}")
    print(f"Waiting for {min_clients} clients to connect ...")
    server.start(server_address)


if __name__ == "__main__":
    main()
