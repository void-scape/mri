Cartilage Viewer (V-Net) — Quick Start
Prereqs

Python 3.9+

macOS or Linux

1) Create & activate a virtual environment
python3 -m venv .venv
source .venv/bin/activate

2) Install UI dependency
pip install --upgrade pip wheel setuptools
pip install --no-compile pyside6==6.7.3

3) Launch the Viewer UI

From the repo root:

Option A (simple script run):

source .venv/bin/activate
python viewer/main.py


Option B (module run; also works without path tweaks):

source .venv/bin/activate
python -m viewer.main


Tip (HiDPI displays):
export QT_AUTO_SCREEN_SCALE_FACTOR=1 before launching.

What happens in the UI

Click Open Input Image… and select a 3D MRI volume (.mhd, .nii, .nii.gz, .nrrd, .mha).

Inference auto-starts in the background (V-Net stub now; real model later).
You’ll see a progress indicator in the status bar.

When finished, click Show Cartilage Mask to toggle the filled overlay.

Use the Sagittal/Coronal/Axial sliders to navigate slices.
(Render/zoom/edit features will expand over time—this is the minimal viewer.)

(Optional) CLI entry point

For batch or scripted runs (same inference engine as the viewer):

source .venv/bin/activate
python cli/main.py --image path/to/input.nii.gz --output-mask path/to/output_mask.nii.gz
# or
python -m cli.main --image ... --output-mask ...