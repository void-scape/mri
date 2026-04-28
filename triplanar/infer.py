#!/usr/bin/env python3

import os

os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"

import argparse
import torch
import dataloader
import h5py
from triplanar import Triplanar
from evaluate import dice


def inference(
    im,
    seg,
    model,
    device,
    ckpt,
    amp=True,
):
    print(f"[triplanar/infer.py] Loading {ckpt}")
    ckpt = torch.load(ckpt, map_location=device)
    model.load_state_dict(ckpt["model"])

    model.eval()
    with torch.autocast(device.type if device.type != "mps" else "cpu", enabled=amp):
        print("[triplanar/infer.py] Inference...")
        volumes = dataloader.load_imgs(im, device, normalize=False)
        logits = model(*[v.squeeze(0) for v in volumes])

        if seg is not None:
            mask = dataloader.load_mask(seg, device)
            score = dice(logits=logits, targets=mask)
            print(f"[triplanar/infer.py] DSC {score.item():.4f}")

    logits = logits.squeeze(0)
    pred = torch.sigmoid(logits) > 0.5
    return pred


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
    args = parse_args()
    if torch.cuda.is_available() and args.device != "cpu":
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print(f"[triplanar/infer.py] Chosen device {device}")

    model = Triplanar(channels=1, features=1, chunks=16)
    model = model.to(device)
    pred = inference(args.im, args.seg, model, device, args.checkpoint)
    with h5py.File(args.out, "w") as mask:
        mask["data"] = pred.cpu()
    print(f"[triplanar/infer.py] Saving segmentation mask to {args.out}")


if __name__ == "__main__":
    main()
