from __future__ import annotations
from pathlib import Path
import time

def run_inference(image_path: Path, output_mask_path: Path, progress_callback=None) -> Path:
    """
    Placeholder V-Net inference.
    Replace this with your real model call. Keep the signature stable.

    progress_callback: callable(int 0..100) or None
    """
    # Simulate work + progress callbacks
    for i in range(20):
        time.sleep(0.1)
        if progress_callback:
            progress_callback(int((i + 1) / 20 * 100))

    # Write a tiny placeholder file so downstream paths exist
    output_mask_path = Path(output_mask_path)
    output_mask_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_mask_path, "wb") as f:
        f.write(b"PLACEHOLDER_MASK")
    return output_mask_path
