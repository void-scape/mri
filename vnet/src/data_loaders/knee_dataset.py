# knee_dataset.py
import os
import glob
import h5py
import numpy as np
import torch
from torch.utils.data import Dataset


def load_volume(path):
    """Load a knee MRI volume from an HDF5 .im file."""
    with h5py.File(path, "r") as f:
        vol = np.array(f["data"])  # e.g. (384, 384, 160) or similar
    return vol.astype(np.float32)


def load_segmentation(path):
    """Load cartilage segmentation labels from an HDF5 .seg file."""
    with h5py.File(path, "r") as f:
        seg = np.array(f["data"])
    return seg.astype(np.int16)


class KneeMRIDataset(Dataset):
    """
    Generic 3D knee MRI + segmentation dataset.

    Expects:
      root/
        train/ or valid/ or test/
          <split>_XXX_VYY.im
          <split>_XXX_VYY.seg  (for train/valid)
    """

    def __init__(self, root, split, normalize=True):
        assert split in ("train", "valid", "test")
        self.root = root
        self.split = split
        self.normalize = normalize

        split_dir = os.path.join(root, split)
        pattern = os.path.join(split_dir, f"{split}_*_V*.im")
        im_paths = sorted(glob.glob(pattern))

        self.samples = []
        for im_path in im_paths:
            base = os.path.splitext(os.path.basename(im_path))[0]  # e.g. valid_001_V00
            seg_path = os.path.join(split_dir, base + ".seg")
            if os.path.exists(seg_path):
                # train/valid
                self.samples.append((im_path, seg_path))
            else:
                # allow missing seg for test
                if split == "test":
                    self.samples.append((im_path, None))

        if len(self.samples) == 0:
            raise RuntimeError(f"No volumes found in {split_dir} with pattern {pattern}")

    def __len__(self):
        return len(self.samples)

    def _preprocess_image(self, vol):
        # vol: numpy array (D, H, W) or (H, W, D); whatever is in the HDF5
        vol = vol.astype(np.float32)
        if self.normalize:
            m = vol.mean()
            s = vol.std()
            if s > 0:
                vol = (vol - m) / s
        # Add channel dimension -> (1, D, H, W)
        vol = np.expand_dims(vol, 0)
        return vol

    def _preprocess_seg(self, seg):
        """
        Output binary labels:
        0 = background / no cartilage
        1 = cartilage

        Assumes one-hot seg with 6 channels total, where:
        channels 0..3 = cartilage to keep
        channels 4..5 = meniscus to discard
        """
        seg = seg.astype(np.int64)

        if seg.ndim == 4 and seg.shape[-1] == 6:
            # keep only first 4 channels, discard last 2
            cartilage = seg[..., :4].sum(axis=-1) > 0
            seg = cartilage.astype(np.int64)

        elif seg.ndim == 3:
            # if already label map:
            # keep labels 1..4 as cartilage, everything else as background
            seg = ((seg >= 1) & (seg <= 4)).astype(np.int64)

        else:
            raise RuntimeError(f"Unexpected seg shape: {seg.shape}")

        return seg

    def __getitem__(self, idx):
        im_path, seg_path = self.samples[idx]

        vol = load_volume(im_path)
        vol = self._preprocess_image(vol)
        vol = torch.from_numpy(vol)  # (1, D, H, W)

        if seg_path is None:
            # For pure inference / test without labels
            return vol, None

        seg = load_segmentation(seg_path)
        seg = self._preprocess_seg(seg)
        seg = torch.from_numpy(seg)  # (D, H, W)

        return vol, seg


if __name__ == "__main__":
    ds = KneeMRIDataset(root="data", split="valid")
    img, seg = ds[0]
    print("Image shape:", img.shape)
    print("Seg shape:", seg.shape, "unique labels:", torch.unique(seg))
