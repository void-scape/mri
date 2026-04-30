#!/usr/bin/env python3

import os
import argparse
import h5py
import numpy as np
import torch

from models import vnet


def print_gpu_memory(tag=""):
    if torch.cuda.is_available():
        alloc = torch.cuda.max_memory_allocated() / (1024**3)
        reserv = torch.cuda.max_memory_reserved() / (1024**3)
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
    Match KneeMRIDataset._preprocess_seg exactly.

    Output:
      0 = background / no cartilage
      1 = cartilage

    Low-res original format:
      channels 0..3 = cartilage
      channels 4..5 = meniscus
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
        vol = np.array(f["data"])
    return vol.astype(np.float32)


def preprocess_like_training(vol: np.ndarray, normalize: bool = True) -> torch.Tensor:
    if normalize:
        m = float(vol.mean())
        s = float(vol.std())
        if s > 0:
            vol = (vol - m) / s

    vol = np.expand_dims(vol, axis=0)
    vol = np.expand_dims(vol, axis=0)
    return torch.from_numpy(vol)


def strip_module_prefix(state_dict):
    if any(k.startswith("module.") for k in state_dict.keys()):
        return {k.replace("module.", "", 1): v for k, v in state_dict.items()}
    return state_dict


def parse_args():
    ap = argparse.ArgumentParser(description="Compute cartilage segmentation mask")
    ap.add_argument(
        "-c", "--checkpoint", required=True, help="path to model checkpoint"
    )
    ap.add_argument("-i", "--im", required=True, help="input path")
    ap.add_argument("-o", "--out", required=True, help="output path")
    ap.add_argument(
        "-s", "--seg", help="segmentation path, used to compute dice for output mask"
    )
    ap.add_argument(
        "--device",
        default="cuda",
        choices=["cuda", "cpu"],
        help="select preferred torch device",
    )
    return ap.parse_args()


def main():
    # ap.add_argument("--no-normalize", action="store_true")
    # ap.add_argument(
    #     "--profile-memory",
    #     action="store_true",
    #     help="report peak GPU memory for one inference pass",
    # )
    args = parse_args()

    device = torch.device(
        args.device if (args.device == "cpu" or torch.cuda.is_available()) else "cpu"
    )
    print(f"[vnet/infer.py] Chosen device {device}")

    num_classes = 2
    model = vnet.VNet(elu=False, nll=True, num_classes=num_classes).to(device)
    model.eval()

    print(f"[vnet/infer.py] Loading {args.checkpoint}")
    ckpt = torch.load(args.checkpoint, map_location=device)

    state_dict = (
        ckpt["state_dict"] if isinstance(ckpt, dict) and "state_dict" in ckpt else ckpt
    )
    state_dict = strip_module_prefix(state_dict)
    model.load_state_dict(state_dict, strict=True)

    vol = load_volume_h5(args.im)
    fix_highres = vol.shape == (512, 512, 80)
    # x = preprocess_like_training(vol, normalize=(not args.no_normalize)).to(device)
    # TODO: long term fix
    if fix_highres:
        vol = np.transpose(vol, axes=(2, 0, 1))
    x = preprocess_like_training(vol, normalize=True).to(device)
    # if args.profile_memory and device.type == "cuda":
    #     reset_gpu_peak_memory()

    print("[vnet/infer.py] Inference...")

    with torch.no_grad():
        out_flat = model(x)
        pred_flat = torch.argmax(out_flat, dim=1)

    # if args.profile_memory and device.type == "cuda":
    #     torch.cuda.synchronize()
    #     print_gpu_memory("LOW-RES INFER")

    _, _, D, H, W = x.shape
    pred_3d = pred_flat.view(D, H, W).cpu().numpy().astype(np.uint8)
    # TODO: long term fix
    if fix_highres:
        pred_3d = np.transpose(pred_3d, axes=(1, 2, 0))

    if args.seg is not None:
        gt_raw = load_seg_h5(args.seg)
        gt = preprocess_lowres_seg_like_training(gt_raw)
        dice = binary_dice(pred_3d, gt)
        print(f"[vnet/infer.py] DSC {dice:.4f}")

    out_path = args.out
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with h5py.File(out_path, "w") as f:
        f.create_dataset("data", data=pred_3d, compression="gzip")
    print(f"[vnet/infer.py] Saving segmentation mask to {args.out}")


if __name__ == "__main__":
    main()
