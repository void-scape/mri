import os
import glob
import torch
import h5py
import numpy as np
import preprocess.normalize as pp
from torch.utils.data import Dataset


def load_imgs(path, device, normalize=False):
    with h5py.File(path, "r") as file:
        img = file["data"][:]
    assert img.dtype == np.float32

    if normalize:
        img = pp.normalize(img)

    # NOTE: I am transposing on the cpu so that the gpu does not have
    # to make multiple allocations and potentially fragment
    # the cpu has plenty of time for this work ;-)

    img_sagittal = np.transpose(img, axes=(0, 1, 2))
    img_coronal = np.transpose(img, axes=(2, 0, 1))
    img_axial = np.transpose(img, axes=(1, 2, 0))

    imgs = [
        torch.tensor(img_sagittal, dtype=torch.float32, device=device).unsqueeze(1),
        torch.tensor(img_coronal, dtype=torch.float32, device=device).unsqueeze(1),
        torch.tensor(img_axial, dtype=torch.float32, device=device).unsqueeze(1),
    ]

    return imgs


def load_mask(path, device):
    with h5py.File(path.replace(".im", ".seg"), "r") as file:
        seg = file["data"][:]
    assert len(seg.shape) == 3

    mask = torch.tensor(seg, dtype=torch.float32, device=device)
    return mask


class HDF5Dataset(Dataset):
    def __init__(self, data_dir, device):
        self.files = sorted(glob.glob(os.path.join(data_dir, "*.im")))
        self.device = device

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        # TODO: These names do not line up with the expected, medical axes,
        # they only reflect the axes of the input
        """
        Expects the volumes generated from `preprocess/low-res.py` and
        `preprocess/high-res.py`.

        Prepares the image and mask volumes in the following form:

            imgs   dtype=torch.float32
                                     x       y    z
              sagittal: torch.Size([384, 1, 384, 160])
                                     z       x    y
              coronal:  torch.Size([160, 1, 384, 384])
                                     y       z    x
              axial:    torch.Size([384, 1, 160, 384])

            mask  dtype=torch.bool
                          x    y    z
              toch.Size([384, 384, 160])
        """
        path = self.files[idx]
        imgs = load_imgs(path, self.device, normalize=False)
        mask = load_mask(path, self.device)
        return imgs, mask
