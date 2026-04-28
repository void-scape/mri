import os
import glob
import argparse
import h5py
import numpy as np
from pathlib import Path
from normalize import normalize


"""
Very specific parser for the IWOAI dataset.

Generates MRI data 'im' files and binary 'seg' files. Removes both meniscus masks
from the segmentation. Normalizes the MRI data in accordance with the high resolution 
data, see `preprocess/high-res.py`.
"""


def parse_args():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("train_path", help="path to the IWOAI train dataset")
    parser.add_argument("valid_path", help="path to the IWOAI valid dataset")
    parser.add_argument("data_path", help="path to save volume data")
    parser.add_argument(
        "-r", help="ratio of data used for validation", default=0.2, type=float
    )
    parser.add_argument("-v", help="print generated filenames", action="store_true")
    return parser.parse_args()


def parse_dir(args, src, train, valid):
    imgs = sorted(glob.glob(os.path.join(src, "*.im")))
    for i, filename in enumerate(imgs):
        if i < round(len(imgs) * (1.0 - args.r)):
            parent = train
        else:
            parent = valid

        with h5py.File(filename, "r") as img:
            data = img["data"][:]
            data = normalize(data)
            path = os.path.join(parent, f"{Path(filename).stem}.im")
            with h5py.File(path, "w") as img:
                img["data"] = data
                if args.v:
                    print(f"... {path}")

        filename = filename.replace(".im", ".seg")
        with h5py.File(filename, "r") as mask:
            data = mask["data"][:]
            # collapse the mask and remove the meniscus
            data = np.sum(data[:, :, :, : data.shape[-1] - 2], axis=3)
            data = data > 0
            path = os.path.join(parent, f"{Path(filename).stem}.seg")
            with h5py.File(path, "w") as mask:
                mask["data"] = data
                if args.v:
                    print(f"... {path}")


def main():
    args = parse_args()

    train = os.path.join(args.data_path, "train")
    valid = os.path.join(args.data_path, "valid")

    os.makedirs(train, exist_ok=True)
    os.makedirs(valid, exist_ok=True)

    parse_dir(args, args.train_path, train, valid)
    parse_dir(args, args.valid_path, train, valid)


if __name__ == "__main__":
    main()
