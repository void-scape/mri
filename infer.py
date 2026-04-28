#!/usr/bin/env python3

import argparse
import subprocess
import sys
from pathlib import Path

"""
Dispatches to the appropriate `vnet/infer.py` and `triplanar/infer.py` scripts
based on the `--arch` argument.
"""


def resolve_script(args):
    scripts = {
        "vnet": "vnet/infer.py",
        "triplanar": "triplanar/infer.py",
    }
    script = scripts[args.arch]
    parent = Path(__file__).resolve().parent
    return (parent / script).resolve()


def parse_args():
    ap = argparse.ArgumentParser(description="Compute cartilage segmentation mask")
    ap.add_argument("-a", "--arch", required=True, choices=["vnet", "triplanar"])
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

    script = resolve_script(args)
    cmd = [
        sys.executable,
        str(script),
        "--checkpoint",
        args.checkpoint,
        "--im",
        args.im,
        "--out",
        args.out,
        "--device",
        args.device,
    ]
    if args.seg is not None:
        cmd.append("--seg")
        cmd.append(args.seg)

    print(" ".join(cmd))
    completed = subprocess.run(cmd)
    raise SystemExit(completed.returncode)


if __name__ == "__main__":
    main()
