import subprocess
import sys, pathlib
ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
import argparse
from pathlib import Path

def parse_args():
    ap = argparse.ArgumentParser(description="Cartilage segmentation CLI")
    ap.add_argument("--image", required=True, type=Path, help="path to input volume")
    ap.add_argument("--output-mask", required=True, type=Path, help="output mask path")
    return ap.parse_args()

def run_inference(image_path, output_path):
    out_dir = Path("outputs")
    out_dir.mkdir(exist_ok=True)
    cmd = [
        sys.executable, "inference/infer_knee.py",
        "--ckpt", "checkpoints/vnet_model_best.pth.tar",
        "--im", str(image_path),
        "--out", output_path,
        "--device", "cpu",
    ]
    subprocess.run(cmd, check=True)

def main():
    args = parse_args()
    if not args.image.exists():
        raise SystemExit(f"Input not found: {args.image}")
    args.output_mask.parent.mkdir(parents=True, exist_ok=True)
    print(f"[CLI] Running inference on: {args.image}")
    run_inference(args.image, args.output_mask)
    print(f"[CLI] Done. Wrote mask: {args.output_mask}")

if __name__ == "__main__":
    main()
