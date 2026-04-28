#!/usr/bin/env python3
import os
import sys
import argparse
from pathlib import Path

import h5py
import numpy as np
import torch

# Allow running as a standalone script from inference/
_THIS = Path(__file__).resolve()
_REPO_ROOT = _THIS.parent.parent if _THIS.parent.name == 'inference' else _THIS.parent
if str(_THIS.parent) not in sys.path:
    sys.path.insert(0, str(_THIS.parent))
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import vnet 


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


def load_seg_h5(path: str) -> np.ndarray:
    with h5py.File(path, "r") as f:
        seg = np.array(f["data"])
    return seg.astype(np.int64)


def preprocess_lowres_seg_like_training(seg: np.ndarray) -> np.ndarray:
    """
    Match the low-res training target preprocessing.

    Output:
      0 = background / no cartilage
      1 = cartilage

    Low-res original format:
      channels 0..3 = cartilage
      channels 4..5 = meniscus -> background
    """
    seg = seg.astype(np.int64)

    if seg.ndim == 4 and seg.shape[-1] == 6:
        cartilage = seg[..., :4].sum(axis=-1) > 0
        seg = cartilage.astype(np.int64)
    elif seg.ndim == 3:
        seg = ((seg >= 1) & (seg <= 4)).astype(np.int64)
    else:
        raise RuntimeError(f"Unexpected seg shape: {seg.shape}")

    return seg


def binary_dice(pred: np.ndarray, target: np.ndarray, eps: float = 1e-5) -> float:
    pred_c = pred == 1
    target_c = target == 1
    intersection = np.logical_and(pred_c, target_c).sum()
    denom = pred_c.sum() + target_c.sum()
    return float((2.0 * intersection + eps) / (denom + eps))


def load_volume_h5(path: str) -> np.ndarray:
    with h5py.File(path, "r") as f:
        if "data" in f:
            vol = np.array(f["data"])
        else:
            # fallback to first dataset found
            key = next(iter(f.keys()))
            vol = np.array(f[key])
    return vol.astype(np.float32)


def preprocess_like_training(vol: np.ndarray, normalize: bool = True) -> torch.Tensor:
    if normalize:
        m = float(vol.mean())
        s = float(vol.std())
        if s > 0:
            vol = (vol - m) / s

    vol = np.expand_dims(vol, axis=0)  # (1, D, H, W)
    vol = np.expand_dims(vol, axis=0)  # (N=1, C=1, D, H, W)
    return torch.from_numpy(vol)


def strip_module_prefix(state_dict):
    if any(k.startswith("module.") for k in state_dict.keys()):
        return {k.replace("module.", "", 1): v for k, v in state_dict.items()}
    return state_dict


def save_prediction(out_path: str, pred_3d: np.ndarray) -> None:
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    if out_path.endswith(".npy"):
        np.save(out_path, pred_3d)
        print(f"Saved predicted labels to {out_path} with shape {pred_3d.shape}")
    elif out_path.endswith((".h5", ".hdf5", ".seg", ".im")):
        with h5py.File(out_path, "w") as f:
            f.create_dataset("data", data=pred_3d, compression="gzip")
        print(f"Saved predicted labels to {out_path} (HDF5 dataset 'data'), shape {pred_3d.shape}")
    else:
        fixed = out_path + ".hdf5"
        with h5py.File(fixed, "w") as f:
            f.create_dataset("data", data=pred_3d, compression="gzip")
        print(f"Unknown extension; saved to {fixed}, shape {pred_3d.shape}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, help="path to binary low-res VNet checkpoint")
    ap.add_argument("--im", required=True, help="path to input low-res HDF5 image file")
    ap.add_argument("--out", required=True, help="output path for predicted labels")
    ap.add_argument("--seg", default=None, help="optional ground-truth .seg/.hdf5 file for Dice calculation")
    ap.add_argument("--no-normalize", action="store_true")
    ap.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    ap.add_argument("--profile-memory", action="store_true", help="report peak GPU memory for one inference pass")
    args = ap.parse_args()

    device = torch.device(args.device if (args.device == "cpu" or torch.cuda.is_available()) else "cpu")

    model = vnet.VNet(elu=False, nll=True, num_classes=2).to(device)
    model.eval()
    print("Evaluation Started")

    ckpt = torch.load(args.ckpt, map_location=device)
    state_dict = ckpt["state_dict"] if isinstance(ckpt, dict) and "state_dict" in ckpt else ckpt
    state_dict = strip_module_prefix(state_dict)
    model.load_state_dict(state_dict, strict=True)
    print("Model Loaded")

    vol = load_volume_h5(args.im)
    x = preprocess_like_training(vol, normalize=(not args.no_normalize)).to(device)
    print("Preprocessing")

    if args.profile_memory and device.type == "cuda":
        reset_gpu_peak_memory()

    with torch.no_grad():
        out_flat = model(x)
        pred_flat = torch.argmax(out_flat, dim=1)

    if args.profile_memory and device.type == "cuda":
        torch.cuda.synchronize()
        print_gpu_memory("LOW-RES INFER")

    print("Forward Pass Completed")

    _, _, D, H, W = x.shape
    pred_3d = pred_flat.view(D, H, W).cpu().numpy().astype(np.uint8)

    if args.seg is not None:
        gt_raw = load_seg_h5(args.seg)
        gt = preprocess_lowres_seg_like_training(gt_raw)

        print("\n=== LOW-RES INFERENCE DICE ===")
        print("pred shape:", pred_3d.shape, "unique:", np.unique(pred_3d))
        print("gt raw shape:", gt_raw.shape, "unique:", np.unique(gt_raw)[:20])
        print("gt mapped shape:", gt.shape, "unique:", np.unique(gt))

        if pred_3d.shape != gt.shape:
            raise RuntimeError(f"Shape mismatch: pred {pred_3d.shape} vs gt {gt.shape}")

        dice = binary_dice(pred_3d, gt)
        print("pred cartilage voxels:", int((pred_3d == 1).sum()))
        print("gt cartilage voxels:", int((gt == 1).sum()))
        print(f"Binary cartilage Dice: {dice:.4f}")
        print("==============================\n")

    save_prediction(args.out, pred_3d)


if __name__ == "__main__":
    main()
