#!/usr/bin/env python3
import argparse
import subprocess
import sys
from pathlib import Path


DEFAULT_SCRIPT_MAP = {
    ("3t", "vnet"): "infer_knee.py",
    ("7t", "vnet"): "infer_7t.py",
    # triplanar can be wired in later without changing the GUI contract
}


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description="Dispatch inference to the correct backend based on input kind and architecture."
    )
    ap.add_argument("--input-kind", required=True, choices=["3t", "7t"])
    ap.add_argument("--arch", required=True, choices=["vnet", "triplanar"])
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--im", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seg", default=None)
    ap.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    ap.add_argument("--no-normalize", action="store_true")
    ap.add_argument("--profile-memory", action="store_true")
    ap.add_argument("--script-3t-vnet", default=None, help="override script path for 3T V-Net")
    ap.add_argument("--script-7t-vnet", default=None, help="override script path for 7T V-Net")
    ap.add_argument("--script-3t-triplanar", default=None, help="override script path for 3T Tri-Planar")
    ap.add_argument("--script-7t-triplanar", default=None, help="override script path for 7T Tri-Planar")
    return ap


def resolve_script(args) -> Path:
    this_dir = Path(__file__).resolve().parent
    override_lookup = {
        ("3t", "vnet"): args.script_3t_vnet,
        ("7t", "vnet"): args.script_7t_vnet,
        ("3t", "triplanar"): args.script_3t_triplanar,
        ("7t", "triplanar"): args.script_7t_triplanar,
    }
    pair = (args.input_kind, args.arch)
    override = override_lookup[pair]
    if override:
        return Path(override).resolve()

    rel = DEFAULT_SCRIPT_MAP.get(pair)
    if rel is None:
        raise SystemExit(
            f"No default script configured for input-kind={args.input_kind!r}, arch={args.arch!r}. "
            f"Pass an explicit override flag or add a backend for this combination."
        )
    return (this_dir / rel).resolve()


def main() -> None:
    ap = build_parser()
    args = ap.parse_args()

    script = resolve_script(args)
    cmd = [
        sys.executable,
        str(script),
        "--ckpt", args.ckpt,
        "--im", args.im,
        "--out", args.out,
        "--device", args.device,
    ]
    if args.seg:
        cmd += ["--seg", args.seg]
    if args.no_normalize:
        cmd.append("--no-normalize")
    if args.profile_memory:
        cmd.append("--profile-memory")

    print("Dispatching:", " ".join(cmd))
    completed = subprocess.run(cmd)
    raise SystemExit(completed.returncode)


if __name__ == "__main__":
    main()
