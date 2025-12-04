from __future__ import annotations
import sys, pathlib
ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
import argparse
from pathlib import Path
from shared.infer import run_inference

def parse_args():
    ap = argparse.ArgumentParser(description="Cartilage segmentation CLI (V-Net stub)")
    ap.add_argument("--image", required=True, type=Path, help="Path to input volume (e.g., .mhd, .nii.gz)")
    ap.add_argument("--output-mask", required=True, type=Path, help="Where to write the output mask")
    return ap.parse_args()

def main():
    args = parse_args()
    if not args.image.exists():
        raise SystemExit(f"Input image not found: {args.image}")
    args.output_mask.parent.mkdir(parents=True, exist_ok=True)
    print(f"[CLI] Running V-Net on: {args.image}")
    out = run_inference(args.image, args.output_mask)
    print(f"[CLI] Done. Wrote mask: {out}")

if __name__ == "__main__":
    main()
