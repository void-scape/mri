import os
import argparse
import h5py
import numpy as np
from pathlib import Path
from normalize import normalize


"""
Very specific parser for the Neal_7T_Cartilages_20200504.hdf5 dataset.

Generates MRI data 'im' files and binary 'seg' files. Normalizes the
MRI data in accordance with the low resolution data, see `preprocess/low-res.py`.
"""


def parse_args():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("path", help="path to the 7T dataset")
    parser.add_argument("data_path", help="path to save volume data")
    parser.add_argument(
        "-r", help="ratio of data used for validation", default=0.2, type=float
    )
    parser.add_argument("-v", help="print generated filenames", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()

    filenames = []
    images = []
    masks = []

    hf = h5py.File(args.path)

    for key in list(hf.keys())[3:-1]:
        filename = str(hf[key]["filename"][:], encoding="utf-8")
        if filename not in filenames:
            filenames.append(filename)
        images.append(hf[key]["normalizedImage"][:])
        mask = hf[key]["exportedSegMask"][:]
        mask = mask > 0
        masks.append(mask)

    train = os.path.join(args.data_path, "train")
    valid = os.path.join(args.data_path, "valid")

    os.makedirs(train, exist_ok=True)
    os.makedirs(valid, exist_ok=True)

    for i, filename in enumerate(filenames):
        if i < round(len(filenames) * (1.0 - args.r)):
            parent = train
        else:
            parent = valid

        slices = 80

        path = os.path.join(parent, f"{Path(filename).stem}.im")
        with h5py.File(path, "w") as img:
            data = np.array(images[i * slices : i * slices + slices], dtype=np.float32)
            # NOTE: transposing to align with the low resolution axes
            data = np.transpose(data, axes=(1, 2, 0))
            data = normalize(data)
            img["data"] = data
            if args.v:
                print(f"... {path}")

        path = os.path.join(parent, f"{Path(filename).stem}.seg")
        with h5py.File(path, "w") as img:
            mask = np.array(masks[i * slices : i * slices + slices], dtype=bool)
            # NOTE: transposing to align with the low resolution axes
            mask = np.transpose(mask, axes=(1, 2, 0))
            img["data"] = mask
            if args.v:
                print(f"... {path}")


if __name__ == "__main__":
    main()
