"""
Prepare client data assignments for federated learning.
=========================================================

Splits the full patient list into per-client subsets and writes JSON files
that each client reads at startup.

Run this ONCE on the machine that has the full dataset, then distribute
the output ``fl_clients/`` folder to every client PC.

Usage:
  python prepare_client_data.py \\
      --patient-json /data/jin/data/combined/patient.json \\
      --n-clients 3 \\
      --output-dir /data/jin/FedFace/fl_clients
"""

import argparse
import json
import os
import random


def train_val_test_split(
    json_path: str,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    seed: int = 42,
):
    with open(json_path) as f:
        findings = json.load(f)

    all_pids = list(findings.keys())
    rng = random.Random(seed)
    rng.shuffle(all_pids)

    n = len(all_pids)
    nt = max(1, int(round(n * train_ratio)))
    nv = max(1, int(round(n * val_ratio)))
    nte = n - nt - nv
    if nte < 0:
        nte = 0
        nv = n - nt

    train_ids = all_pids[:nt]
    val_ids = all_pids[nt : nt + nv]
    test_ids = all_pids[nt + nv :]

    print(
        f"Split: Train {len(train_ids)} | Val {len(val_ids)} "
        f"| Test {len(test_ids)}"
    )
    return train_ids, val_ids, test_ids


def split_into_centers(ids, n_centers, seed=42):
    rng = random.Random(seed)
    shuffled = list(ids)
    rng.shuffle(shuffled)
    centers: list[list] = [[] for _ in range(n_centers)]
    for i, pid in enumerate(shuffled):
        centers[i % n_centers].append(pid)
    for c in centers:
        rng.shuffle(c)
    return centers


def main():
    parser = argparse.ArgumentParser(
        description="Prepare client data assignments for FedMorph FL"
    )
    parser.add_argument(
        "--patient-json", type=str, default="patient.json",
        help="Path to patient.json",
    )
    parser.add_argument("--n-clients", type=int, default=3)
    parser.add_argument("--output-dir", type=str, default="fl_clients")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    train_ids, val_ids, test_ids = train_val_test_split(
        args.patient_json, seed=args.seed
    )

    center_train = split_into_centers(train_ids, args.n_clients, args.seed)
    center_val = split_into_centers(val_ids, args.n_clients, args.seed + 1)

    os.makedirs(args.output_dir, exist_ok=True)

    with open(os.path.join(args.output_dir, "test_patients.json"), "w") as f:
        json.dump({"test": test_ids}, f, indent=2)

    print(f"\n{'='*60}")
    print("Client Data Assignments")
    print(f"{'='*60}")

    for cid in range(args.n_clients):
        data = {
            "train": center_train[cid],
            "val": center_val[cid],
        }
        path = os.path.join(args.output_dir, f"client_{cid}_patients.json")
        with open(path, "w") as f:
            json.dump(data, f, indent=2)

        print(
            f"  Client {cid}: train={len(data['train']):3d}, "
            f"val={len(data['val'])}"
        )

    print(f"\n  Test set (shared): {len(test_ids)} patients")
    print(f"\n  Files saved to: {os.path.abspath(args.output_dir)}/")

    print(f"\n{'='*60}")
    print("Deployment Instructions (Windows 3-PC)")
    print(f"{'='*60}")
    print("""
  1. Copy the CT data folder to each PC:
       Source: <original CT data path>
       Target: D:\\data\\combined   (or any path)

  2. Copy the fl_clients/ folder to each PC:
       Source: {out_dir}
       Target: D:\\data\\fl_clients

  3. On the SERVER PC, run:
       run_liver_server.bat

  4. On each CLIENT PC, run:
       run_liver_client.bat <CLIENT_ID> <SERVER_IP>:9000

     Example:
       PC-A (client 0): run_liver_client.bat 0 192.168.1.100:9000
       PC-B (client 1): run_liver_client.bat 1 192.168.1.100:9000
       PC-C (client 2): run_liver_client.bat 2 192.168.1.100:9000

  5. Make sure port 9000 is open in Windows Firewall on the server PC.
""".format(out_dir=os.path.abspath(args.output_dir)))


if __name__ == "__main__":
    main()
