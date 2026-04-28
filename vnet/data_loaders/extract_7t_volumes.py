#!/usr/bin/env python3
import os
import argparse
import h5py
import numpy as np

from ..data_loaders.dataset_7t import load_7t_volumes


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--slices-per-volume", type=int, default=80)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    volumes, segmentations = load_7t_volumes(args.data, args.slices_per_volume)

    for i, (vol, seg) in enumerate(zip(volumes, segmentations)):
        im_path = os.path.join(args.out_dir, f"volume_{i:02d}.im")
        seg_path = os.path.join(args.out_dir, f"volume_{i:02d}.seg")

        with h5py.File(im_path, "w") as f:
            f.create_dataset("data", data=vol.astype(np.float32), compression="gzip")

        with h5py.File(seg_path, "w") as f:
            f.create_dataset("data", data=seg.astype(np.int64), compression="gzip")

        print(f"Saved {im_path}")
        print(f"Saved {seg_path}")


if __name__ == "__main__":
    main()