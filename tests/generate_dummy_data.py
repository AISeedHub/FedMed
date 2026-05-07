"""
Generate dummy CT data for testing FedMorph federated learning.

Creates small synthetic volumes that match the expected data format:
  image.npy  — (D, H, W)   float32, CT-like values [0, 255]
  mask.npy   — (C, D, H, W) uint8, C>=10 (background + 9 segments)

Usage:
  python tests/generate_dummy_data.py                      # default: tests/dummy_data/
  python tests/generate_dummy_data.py --out-dir /tmp/data  # custom output
  python tests/generate_dummy_data.py --n-patients 10      # more patients
"""

import argparse
import os

import numpy as np


def make_sphere_mask(shape, center, radius):
    """Create a binary sphere in a 3D volume."""
    D, H, W = shape
    d, h, w = np.ogrid[:D, :H, :W]
    dist = np.sqrt(
        (d - center[0]) ** 2 + (h - center[1]) ** 2 + (w - center[2]) ** 2
    )
    return (dist <= radius).astype(np.uint8)


def generate_patient(out_dir: str, pid: str, depth=80, height=128, width=128):
    """Generate one patient with synthetic CT volume and 9-segment mask."""
    patient_dir = os.path.join(out_dir, pid)
    os.makedirs(patient_dir, exist_ok=True)

    rng = np.random.default_rng(hash(pid) % (2**31))

    image = rng.normal(loc=120, scale=40, size=(depth, height, width)).clip(0, 255)
    image = image.astype(np.float32)

    num_segments = 9
    num_channels = num_segments + 1  # background + 9 segments
    mask = np.zeros((num_channels, depth, height, width), dtype=np.uint8)

    centers_h = [height * 0.3, height * 0.7, height * 0.5,
                 height * 0.3, height * 0.7, height * 0.3,
                 height * 0.7, height * 0.5, height * 0.5]
    centers_w = [width * 0.3, width * 0.3, width * 0.5,
                 width * 0.7, width * 0.7, width * 0.5,
                 width * 0.5, width * 0.3, width * 0.7]
    centers_d = [depth * (0.2 + 0.06 * i) for i in range(num_segments)]

    for seg_idx in range(num_segments):
        center = (
            int(centers_d[seg_idx] + rng.integers(-3, 4)),
            int(centers_h[seg_idx] + rng.integers(-5, 6)),
            int(centers_w[seg_idx] + rng.integers(-5, 6)),
        )
        radius = int(min(depth, height, width) * 0.08 + rng.integers(0, 5))
        sphere = make_sphere_mask((depth, height, width), center, radius)
        mask[seg_idx + 1] = sphere

        brightness = 160 + seg_idx * 8 + rng.normal(0, 5)
        image[sphere > 0] = brightness

    any_seg = mask[1:].max(axis=0)
    mask[0] = 1 - any_seg

    np.save(os.path.join(patient_dir, "image.npy"), image)
    np.save(os.path.join(patient_dir, "mask.npy"), mask)

    return image.shape, mask.shape


def main():
    parser = argparse.ArgumentParser(description="Generate dummy data for FedMorph")
    parser.add_argument(
        "--out-dir", type=str, default="tests/dummy_data",
        help="Output directory (default: tests/dummy_data/)",
    )
    parser.add_argument("--n-patients", type=int, default=6)
    parser.add_argument("--depth", type=int, default=80)
    parser.add_argument("--height", type=int, default=128)
    parser.add_argument("--width", type=int, default=128)
    args = parser.parse_args()

    print(f"Generating {args.n_patients} dummy patients in {args.out_dir}")
    for i in range(args.n_patients):
        pid = f"patient_{i:03d}"
        img_shape, mask_shape = generate_patient(
            args.out_dir, pid, args.depth, args.height, args.width,
        )
        print(f"  {pid}: image {img_shape}, mask {mask_shape}")

    print(f"\nDone. {args.n_patients} patients in {os.path.abspath(args.out_dir)}")


if __name__ == "__main__":
    main()
