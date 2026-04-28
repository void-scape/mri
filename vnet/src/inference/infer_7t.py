#!/usr/bin/env python3
import os
import argparse
import h5py
import numpy as np
import torch

from ..models import vnet
from ..data_loaders.dataset_7t import load_7t_volumes
from ..data_loaders.dataset_7t import remap_segmentation_7t_to_binary

def print_gpu_memory(tag=""):
    if torch.cuda.is_available():
        alloc = torch.cuda.max_memory_allocated() / (1024 ** 3)
        reserv = torch.cuda.max_memory_reserved() / (1024 ** 3)
        print(f"\n[{tag}] Peak allocated: {alloc:.3f} GB")
        print(f"[{tag}] Peak reserved:  {reserv:.3f} GB\n")

def reset_gpu_peak_memory():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

def load_h5_data(path: str, dtype=None) -> np.ndarray:
    with h5py.File(path, "r") as f:
        arr = np.array(f["data"])

    if dtype is not None:
        arr = arr.astype(dtype)

    return arr

    return vol, seg

def binary_dice(pred: np.ndarray, target: np.ndarray, eps: float = 1e-5) -> float:
    pred_c = pred == 1
    target_c = target == 1

    intersection = np.logical_and(pred_c, target_c).sum()
    denom = pred_c.sum() + target_c.sum()

    return float((2.0 * intersection + eps) / (denom + eps))

def preprocess_volume(vol, normalize=True):
    # vol: (D, H, W)
    if normalize:
        m = float(vol.mean())
        s = float(vol.std())
        if s > 0:
            vol = (vol - m) / s

    # Add channel and batch dims: (1, 1, D, H, W)
    vol = np.expand_dims(vol, axis=0)   # (1, D, H, W)
    vol = np.expand_dims(vol, axis=0) 
    return torch.from_numpy(vol)

def strip_module_prefix(state_dict):
    if any(k.startswith("module.") for k in state_dict.keys()):
        return {k.replace("module.", "", 1): v for k, v in state_dict.items()}
    return state_dict

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, help="path to vnet_model_best.pth.tar")
    ap.add_argument("--im", required=True, help="single extracted 7T volume HDF5 file")
    ap.add_argument("--seg", default=None, help="optional extracted 7T segmentation .seg file")
    ap.add_argument("--dice", action="store_true", help="compute Dice if seg exists in the HDF5")
    ap.add_argument("--out", required=True, help="output path prefix for predicted segmentations (e.g., pred_7t)")
    ap.add_argument("--no-normalize", action="store_true")
    ap.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    ap.add_argument("--slices-per-volume", type=int, default=80, help="Number of slices per volume")
    ap.add_argument("--profile-memory", action="store_true",
                help="report peak GPU memory for one inference pass")
    args = ap.parse_args()

    device = torch.device(args.device if (args.device == "cpu" or torch.cuda.is_available()) else "cpu")

    # Build model exactly like training
    num_classes = 2
    model = vnet.VNet(elu=False, nll=True, num_classes=num_classes).to(device)
    model.eval()

    # Load checkpoint
    ckpt = torch.load(args.ckpt, map_location=device)
    state_dict = ckpt["state_dict"] if isinstance(ckpt, dict) and "state_dict" in ckpt else ckpt
    state_dict = strip_module_prefix(state_dict)
    model.load_state_dict(state_dict, strict=True)

    print("Model Loaded")

    vol = load_h5_data(args.im, dtype=np.float32)
    seg_raw = load_h5_data(args.seg, dtype=np.int64) if args.seg is not None else None

    D, H, W = vol.shape
    x = preprocess_volume(vol, normalize=(not args.no_normalize)).to(device)

    if args.profile_memory and device.type == "cuda":
        reset_gpu_peak_memory()

    with torch.no_grad():
        output = model(x)
        pred_flat = torch.argmax(output, dim=1)

    if args.profile_memory and device.type == "cuda":
        torch.cuda.synchronize()
        print_gpu_memory("HIGH-RES INFER")

    pred_3d = pred_flat.view(D, H, W).cpu().numpy().astype(np.uint8)

    if args.dice:
        if seg_raw is None:
            raise RuntimeError("Cannot compute Dice because this HDF5 has no 'seg' dataset.")

        gt = remap_segmentation_7t_to_binary(seg_raw)

        print("\n=== HIGH-RES INFERENCE DICE ===")
        print("pred shape:", pred_3d.shape, "unique:", np.unique(pred_3d))
        print("gt raw shape:", seg_raw.shape, "unique:", np.unique(seg_raw)[:20])
        print("gt mapped shape:", gt.shape, "unique:", np.unique(gt))

        if pred_3d.shape != gt.shape:
            raise RuntimeError(f"Shape mismatch: pred {pred_3d.shape} vs gt {gt.shape}")

        dice = binary_dice(pred_3d, gt)

        print("pred cartilage voxels:", int((pred_3d == 1).sum()))
        print("gt cartilage voxels:", int((gt == 1).sum()))
        print(f"Binary cartilage Dice: {dice:.4f}")
        print("===============================\n")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)

    with h5py.File(args.out, "w") as f:
        f.create_dataset("data", data=pred_3d, compression="gzip")

    print(f"Saved predicted segmentation to {args.out}, shape {pred_3d.shape}")

    if args.profile_memory:
        return

if __name__ == "__main__":
    main()