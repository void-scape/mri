import h5py
import numpy as np
import torch
from torch.utils.data import Dataset


LABEL_MAP_7T_TO_4 = {
    0: 0,
    1: 4,  # Patella -> patellar cartilage
    2: 2,  # Medial Tibia -> medial tibial cartilage
    3: 1,  # Central medial femoral condyle -> femoral cartilage
    4: 3,  # Lateral Tibia -> lateral tibial cartilage
    5: 1,  # Central lateral femoral condyle -> femoral cartilage
    6: 1,  # Trochlea -> femoral cartilage
    7: 1,  # Posterior medial femoral condyle -> femoral cartilage
    8: 1,  # Posterior lateral femoral condyle -> femoral cartilage
    9: 0,  # Patella Bone -> ignore/background
}


def remap_segmentation_7t_to_binary(seg: np.ndarray) -> np.ndarray:
    # 0 = background, 9 = patella bone -> background
    # 1..8 = cartilage -> 1
    out = np.zeros_like(seg, dtype=np.int64)
    out[(seg >= 1) & (seg <= 8)] = 1
    return out


def load_7t_volumes(h5_path, slices_per_volume=80):
    with h5py.File(h5_path, "r") as f:
        slice_keys = sorted([k for k in f.keys() if k.startswith("Slice")])

        volumes = []
        segmentations = []

        for start_idx in range(0, len(slice_keys), slices_per_volume):
            end_idx = min(start_idx + slices_per_volume, len(slice_keys))

            vol_slices = []
            seg_slices = []

            for i in range(start_idx, end_idx):
                key = slice_keys[i]
                img = np.array(f[key]["normalizedImage"])
                mask = np.array(f[key]["exportedSegMask"])
                vol_slices.append(img)
                seg_slices.append(mask)

            vol = np.stack(vol_slices, axis=0)
            seg = np.stack(seg_slices, axis=0)

            volumes.append(vol)
            segmentations.append(seg)

    return volumes, segmentations


class Knee7TDataset(Dataset):
    def __init__(
        self,
        h5_path,
        image_ids,
        slices_per_volume=80,
        patch_size=(80, 192, 192),
        num_patches=2000,
        normalize=True,
        augment=False,
        full_volume=False,
    ):
        self.h5_path = h5_path
        self.image_ids = image_ids
        self.slices_per_volume = slices_per_volume
        self.patch_size = patch_size
        self.num_patches = num_patches
        self.normalize = normalize
        self.augment = augment
        self.full_volume = full_volume

        all_volumes, all_segmentations = load_7t_volumes(h5_path, slices_per_volume)

        self.volumes = [all_volumes[i] for i in image_ids]
        self.segmentations = [all_segmentations[i] for i in image_ids]

        if self.normalize:
            for i in range(len(self.volumes)):
                vol = self.volumes[i]
                m = vol.mean()
                s = vol.std()
                if s > 0:
                    self.volumes[i] = (vol - m) / s

        self.num_volumes = len(self.volumes)
        self.D, self.H, self.W = self.volumes[0].shape

    def __len__(self):
        return self.num_volumes if self.full_volume else self.num_patches

    def __getitem__(self, idx):
        if self.full_volume:
            vol = self.volumes[idx]
            seg = self.segmentations[idx]
            seg = remap_segmentation_7t_to_binary(seg)

            vol = np.expand_dims(vol, 0)
            vol = np.ascontiguousarray(vol, dtype=np.float32)
            seg = np.ascontiguousarray(seg, dtype=np.int64)

            return torch.from_numpy(vol), torch.from_numpy(seg)

        vol_idx = np.random.randint(0, self.num_volumes)
        vol = self.volumes[vol_idx]
        seg = self.segmentations[vol_idx]

        d_patch, h_patch, w_patch = self.patch_size

        d_start = np.random.randint(0, self.D - d_patch + 1)
        h_start = np.random.randint(0, self.H - h_patch + 1)
        w_start = np.random.randint(0, self.W - w_patch + 1)

        vol_patch = vol[
            d_start:d_start + d_patch,
            h_start:h_start + h_patch,
            w_start:w_start + w_patch
        ]
        seg_patch = seg[
            d_start:d_start + d_patch,
            h_start:h_start + h_patch,
            w_start:w_start + w_patch
        ]

        seg_patch = remap_segmentation_7t_to_binary(seg_patch)

        if self.augment:
            if np.random.rand() > 0.5:
                vol_patch = np.flip(vol_patch, axis=1)
                seg_patch = np.flip(seg_patch, axis=1)
            if np.random.rand() > 0.5:
                vol_patch = np.flip(vol_patch, axis=2)
                seg_patch = np.flip(seg_patch, axis=2)

        vol_patch = np.expand_dims(vol_patch, 0)
        vol_patch = np.ascontiguousarray(vol_patch, dtype=np.float32)
        seg_patch = np.ascontiguousarray(seg_patch, dtype=np.int64)

        return torch.from_numpy(vol_patch), torch.from_numpy(seg_patch)