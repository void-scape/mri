from pathlib import Path

def open_case_dialog():
    """Return a (case_id, paths) structure or None. TODO: implement."""
    pass

def open_image_mask_dialog():
    """Return (image_path, mask_path) or None. TODO: implement."""
    pass

def load_pickle_volume(path: Path):
    """Load Zenodo volumetric pickle (numpy array). TODO: implement."""
    pass

def load_metaimage_pair(image_path: Path, seg_path: Path):
    """Load SKI10 .mhd/.raw image+mask pair. TODO: implement."""
    pass

def download_ski10(local_dir: Path):
    """Use huggingface_hub.snapshot_download to fetch SKI10. TODO: implement."""
    pass
