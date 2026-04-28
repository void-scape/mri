#!/usr/bin/env python3
import os
import sys
import argparse
from pathlib import Path
from typing import Optional, Tuple, List

import h5py
import numpy as np
import torch

_THIS = Path(__file__).resolve()
_REPO_ROOT = _THIS.parent.parent if _THIS.parent.name == 'inference' else _THIS.parent
if str(_THIS.parent) not in sys.path:
    sys.path.insert(0, str(_THIS.parent))
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import vnet
from data_loaders.dataset_7t import load_7t_volumes, remap_segmentation_7t_to_binary


def print_gpu_memory(tag: str = ""):
    if torch.cuda.is_available():
        alloc = torch.cuda.max_memory_allocated() / (1024 ** 3)
        reserv = torch.cuda.max_memory_reserved() / (1024 ** 3)
        print(f"\n[{tag}] Peak allocated: {alloc:.3f} GB")
        print(f"[{tag}] Peak reserved:  {reserv:.3f} GB\n")


def reset_gpu_peak_memory():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()


def strip_module_prefix(state_dict):
    if any(k.startswith("module.") for k in state_dict.keys()):
        return {k.replace("module.", "", 1): v for k, v in state_dict.items()}
    return state_dict


def preprocess_volume(vol: np.ndarray, normalize: bool = True) -> torch.Tensor:
    vol = np.asarray(vol, dtype=np.float32)
    if normalize:
        m = float(vol.mean())
        s = float(vol.std())
        if s > 0:
            vol = (vol - m) / s
    vol = np.expand_dims(vol, axis=0)
    vol = np.expand_dims(vol, axis=0)
    return torch.from_numpy(vol)


def binary_dice(pred: np.ndarray, target: np.ndarray, eps: float = 1e-5) -> float:
    pred_c = np.asarray(pred) == 1
    target_c = np.asarray(target) == 1
    intersection = np.logical_and(pred_c, target_c).sum()
    denom = pred_c.sum() + target_c.sum()
    return float((2.0 * intersection + eps) / (denom + eps))


def load_single_7t_volume(path: str) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    with h5py.File(path, "r") as f:
        if "image" in f:
            vol = np.array(f["image"]).astype(np.float32)
        elif "data" in f:
            vol = np.array(f["data"]).astype(np.float32)
        else:
            key = next(iter(f.keys()))
            vol = np.array(f[key]).astype(np.float32)

        seg = None
        if "seg" in f:
            seg = np.array(f["seg"]).astype(np.int64)
        elif "data" in f and str(path).endswith(("_mask.h5", "_mask.hdf5", ".seg")):
            seg = np.array(f["data"]).astype(np.int64)
    return vol, seg


def load_seg_h5(path: str) -> np.ndarray:
    with h5py.File(path, "r") as f:
        if "seg" in f:
            seg = np.array(f["seg"])
        elif "data" in f:
            seg = np.array(f["data"])
        else:
            key = next(iter(f.keys()))
            seg = np.array(f[key])
    return seg.astype(np.int64)


def save_prediction(out_path: str, pred_3d: np.ndarray) -> None:
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with h5py.File(out_path, "w") as f:
        f.create_dataset("data", data=pred_3d, compression="gzip")
    print(f"Saved predicted segmentation to {out_path}, shape {pred_3d.shape}")


def infer_one_volume(model, vol: np.ndarray, device: torch.device, normalize: bool, profile_memory: bool) -> np.ndarray:
    D, H, W = vol.shape
    x = preprocess_volume(vol, normalize=normalize).to(device)

    if profile_memory and device.type == "cuda":
        reset_gpu_peak_memory()

    with torch.no_grad():
        output = model(x)  # (1, D*H*W, C)
        pred_flat = torch.argmax(output, dim=1)

    if profile_memory and device.type == "cuda":
        torch.cuda.synchronize()
        print_gpu_memory("HIGH-RES INFER")

    pred_3d = pred_flat.view(D, H, W).cpu().numpy().astype(np.uint8)
    return pred_3d


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, help="path to binary high-res VNet checkpoint")
    ap.add_argument("--im", default=None, help="single 7T HDF5 file containing 'image' and optionally 'seg'")
    ap.add_argument("--data", default=None, help="legacy 7T HDF5 dataset path with SliceXXXX groups")
    ap.add_argument("--out", required=True, help="output path or prefix for predicted segmentation(s)")
    ap.add_argument("--seg", default=None, help="optional ground-truth mask HDF5 file for Dice calculation")
    ap.add_argument("--dice", action="store_true", help="compute binary Dice when GT is available")
    ap.add_argument("--no-normalize", action="store_true")
    ap.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    ap.add_argument("--slices-per-volume", type=int, default=80, help="Number of slices per legacy dataset volume")
    ap.add_argument("--profile-memory", action="store_true", help="report peak GPU memory for one inference pass")
    args = ap.parse_args()

    if not args.im and not args.data:
        ap.error("one of --im or --data is required")

    device = torch.device(args.device if (args.device == "cpu" or torch.cuda.is_available()) else "cpu")

    model = vnet.VNet(elu=False, nll=True, num_classes=2).to(device)
    model.eval()

    ckpt = torch.load(args.ckpt, map_location=device)
    state_dict = ckpt["state_dict"] if isinstance(ckpt, dict) and "state_dict" in ckpt else ckpt
    state_dict = strip_module_prefix(state_dict)
    model.load_state_dict(state_dict, strict=True)

    normalize = not args.no_normalize

    # Single combined/per-volume file path.
    if args.im:
        vol, embedded_seg = load_single_7t_volume(args.im)
        pred_3d = infer_one_volume(model, vol, device, normalize, args.profile_memory)
        save_prediction(args.out, pred_3d)

        if args.dice:
            gt_raw = None
            if args.seg:
                gt_raw = load_seg_h5(args.seg)
            elif embedded_seg is not None:
                gt_raw = embedded_seg
            if gt_raw is None:
                print("--dice requested, but no GT available (pass --seg or use an HDF5 with embedded 'seg').")
            else:
                gt = remap_segmentation_7t_to_binary(gt_raw)
                if pred_3d.shape != gt.shape:
                    raise RuntimeError(f"Prediction shape {pred_3d.shape} != GT shape {gt.shape}")
                dice = binary_dice(pred_3d, gt)
                print(f"Cartilage DSC: {dice:.4f}")
        return

    # Legacy dataset-wide path with SliceXXXX groups.
    volumes, segmentations = load_7t_volumes(args.data, args.slices_per_volume)
    out_base = Path(args.out)
    for i, vol in enumerate(volumes):
        pred_3d = infer_one_volume(model, vol, device, normalize, args.profile_memory)
        out_path = str(out_base.with_suffix("")) + f"_{i:02d}.h5"
        save_prediction(out_path, pred_3d)

        if args.dice:
            gt = remap_segmentation_7t_to_binary(segmentations[i])
            if pred_3d.shape != gt.shape:
                raise RuntimeError(f"Prediction shape {pred_3d.shape} != GT shape {gt.shape}")
            dice = binary_dice(pred_3d, gt)
            print(f"Volume {i:02d} Cartilage DSC: {dice:.4f}")

        if args.profile_memory:
            return


if __name__ == "__main__":
    main()
