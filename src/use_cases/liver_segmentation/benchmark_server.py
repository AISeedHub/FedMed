"""
FedMorph Benchmark Server
=========================

Runs FL experiments for all 4 methods sequentially:
  FedAvg → FedProx → FedBN → FedMorph

Each method runs a full FL session (N rounds).
Clients must reconnect for each method using benchmark_client.py.

Usage:
  uv run python src/use_cases/liver_segmentation/benchmark_server.py
  uv run python src/use_cases/liver_segmentation/benchmark_server.py --methods FedAvg FedMorph
"""

import argparse
import math
import os
import sys
import time

os.environ["TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD"] = "1"

import flwr as fl
import torch
import yaml

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "..", ".."))

from src.fed_core.fedmorph_strategy import FedMorphStrategy
from src.use_cases.liver_segmentation.models.segresnet_morph import build_model

ALL_METHODS = ["FedAvg", "FedProx", "FedBN", "FedMorph"]


def load_config(path="src/use_cases/liver_segmentation/configs/base.yaml"):
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def get_model_state_keys(config):
    model = build_model(config, torch.device("cpu"))
    keys = list(model.state_dict().keys())
    del model
    return keys


def run_one_method(method, config, model_keys, server_address):
    """Run a single FL method for fl_rounds."""
    fl_rounds = config["fl_rounds"]
    min_clients = config["min_clients"]
    local_epochs = config["local_epochs"]

    def fit_config_fn(server_round):
        lr_scale = 0.5 * (1 + math.cos(math.pi * server_round / fl_rounds))
        return {
            "server_round": server_round,
            "local_epochs": local_epochs,
            "method": method,
            "lr_scale": lr_scale,
            "benchmark_method": method,
        }

    def evaluate_metrics_agg_fn(metrics):
        if not metrics:
            return {}
        dices = [n * m.get("dice", 0.0) for n, m in metrics]
        examples = [n for n, _ in metrics]
        total = sum(examples)
        if total == 0:
            return {}
        avg_dice = sum(dices) / total
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

    fl.server.start_server(
        server_address=server_address,
        config=fl.server.ServerConfig(num_rounds=fl_rounds),
        strategy=strategy,
    )


def main():
    parser = argparse.ArgumentParser(description="FedMorph Benchmark Server")
    parser.add_argument("--config", type=str,
                        default="src/use_cases/liver_segmentation/configs/base.yaml")
    parser.add_argument("--methods", nargs="+", default=ALL_METHODS,
                        choices=ALL_METHODS,
                        help="Methods to benchmark (default: all 4)")
    args = parser.parse_args()

    config = load_config(args.config)
    server_address = config.get("server_address", "0.0.0.0:9000")
    model_keys = get_model_state_keys(config)

    methods = args.methods
    total = len(methods)

    print("=" * 60)
    print("  FedMorph Benchmark Server")
    print("=" * 60)
    print(f"  Methods:     {', '.join(methods)}")
    print(f"  Rounds/method: {config['fl_rounds']}")
    print(f"  Min Clients: {config['min_clients']}")
    print(f"  Address:     {server_address}")
    print("=" * 60)

    for i, method in enumerate(methods, 1):
        print(f"\n{'#' * 60}")
        print(f"  [{i}/{total}] Starting method: {method}")
        print(f"{'#' * 60}")
        print(f"  Waiting for {config['min_clients']} clients to connect...")

        t0 = time.time()
        run_one_method(method, config, model_keys, server_address)
        elapsed = time.time() - t0

        print(f"\n  [{i}/{total}] {method} completed in {elapsed:.0f}s")

        if i < total:
            wait = 5
            print(f"  Next method in {wait}s... "
                  f"(clients should auto-reconnect)")
            time.sleep(wait)

    print(f"\n{'=' * 60}")
    print("  Benchmark complete!")
    print(f"  Methods tested: {', '.join(methods)}")
    print("=" * 60)


if __name__ == "__main__":
    main()
