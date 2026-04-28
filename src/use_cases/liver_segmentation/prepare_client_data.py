"""
Prepare client data assignments for federated learning.
=========================================================

Splits the full patient list into per-client subsets and writes JSON files
that each client reads at startup.

Patients are auto-detected by scanning the data directory for folders
that contain both ``image.npy`` and ``mask.npy``.

Run this ONCE on the machine that has the full dataset, then distribute
the output ``fl_clients/`` folder to every client PC.

Usage:
  python prepare_client_data.py \\
      --data-dir /path/to/combined \\
      --n-clients 3 \\
      --output-dir fl_clients
"""

import argparse
import json
import os
import random


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
    print(f"Discovered {len(pids)} patients in {data_dir}")
    return pids


def train_val_test_split(
    all_pids: list[str],
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    seed: int = 42,
):
    rng = random.Random(seed)
    pids = list(all_pids)
    rng.shuffle(pids)

    n = len(pids)
    nt = max(1, int(round(n * train_ratio)))
    nv = max(1, int(round(n * val_ratio)))
    nte = n - nt - nv
    if nte < 0:
        nte = 0
        nv = n - nt

    train_ids = pids[:nt]
    val_ids = pids[nt : nt + nv]
    test_ids = pids[nt + nv :]

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
        "--data-dir", type=str, required=True,
        help="Path to CT data directory (each subfolder = one patient with image.npy + mask.npy)",
    )
    parser.add_argument("--n-clients", type=int, default=3)
    parser.add_argument("--output-dir", type=str, default="fl_clients")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    all_pids = discover_patients(args.data_dir)
    if not all_pids:
        print(f"ERROR: No patients found in {args.data_dir}")
        print("  Each patient folder must contain image.npy and mask.npy")
        return

    train_ids, val_ids, test_ids = train_val_test_split(
        all_pids, seed=args.seed
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
