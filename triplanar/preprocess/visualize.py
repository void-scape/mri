import argparse
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import zoom
import h5py


"""
Plots the slices of a 3D volume along an arbitrary axis.

Clamps the data to the range `[0..1]`.
"""


def parse_args():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("files", nargs="+")
    parser.add_argument("-a", "--axis", default=0, type=int)
    parser.add_argument("-s", "--size", default=2, type=float)
    parser.add_argument(
        "-c", "--coverage", default=1.0, type=float, help="percentage of slices to plot"
    )
    parser.add_argument(
        "-r", "--resize", action="store_true", help="resize image to a square"
    )
    return parser.parse_args()


def plot_img(img, axis, size, coverage, resize):
    assert len(img.shape) == 3, f"Expected 3D input, got {len(img.shape)} dimensions"

    slices = img.shape[axis]
    coverage = max(1, round(slices * coverage))
    subplots = max(1, int(np.ceil(np.sqrt(coverage))))
    plt.subplots(subplots, subplots, figsize=(subplots * size, subplots * size))
    step = slices / coverage
    for i, slice in enumerate(np.arange(0, slices, step).astype(int)):
        plt.subplot(subplots, subplots, i + 1)
        s = np.take(img, slice, axis=axis)
        if resize:
            shape = s.shape
            size = max(shape)
            s = zoom(s, (size / shape[0], size / shape[1]))
        plt.imshow(s, cmap="gray", vmin=0, vmax=1)
        plt.title("Slice {}".format(slice), fontsize=10)
        plt.axis("off")
    for i in range(coverage, subplots * subplots):
        plt.subplot(subplots, subplots, i + 1)
        plt.axis("off")
    plt.tight_layout()


def main():
    args = parse_args()
    for file in args.files:
        with h5py.File(file, "r") as hf:
            plot_img(
                np.array(hf["data"]), args.axis, args.size, args.coverage, args.resize
            )
    plt.show()


if __name__ == "__main__":
    main()
