# Running the MRI Viewer (Quick Instructions)

These steps assume you already pulled/cloned the full repo and you just want to run the GUI viewer.

## What you need
- Python 3.10+ installed
- The repo on your machine
- (Optional) A test image + mask/prediction file to load in the viewer

> Note: **Torch is NOT required just to run the viewer.**
> Torch is only needed if you click **Run Inference** in the GUI.

---

## Windows (PowerShell or Windows Terminal)

### 1) Open a terminal in the repo root
```
cd C:\path\to\repo
```

### 2) Create a virtual environment (first time only)
```powershell
python -m venv .venv
```

### 3) Install dependencies
This avoids activation-policy issues by using the venv’s Python directly:
```powershell
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

### 4) Run the viewer
```powershell
.\.venv\Scripts\python.exe viewer\main.py
```

---

## macOS / Linux (Terminal)

### 1) Open a terminal in the repo root
```bash
cd /path/to/repo
```

### 2) Create a virtual environment (first time only)
```bash
python3 -m venv .venv
```

### 3) Install dependencies (using the venv Python directly)
```bash
./.venv/bin/python -m pip install --upgrade pip
./.venv/bin/python -m pip install -r requirements.txt
```

### 4) Run the viewer
```bash
./.venv/bin/python viewer/main.py
```

---

## Using the Viewer
1. **Open Image…** and select a `.im` / `.hdf5` / `.h5` file (HDF5 with dataset key `data`).
2. (Optional) **Open Mask…** and select:
   - `.npy` ground-truth mask, or
   - predicted mask saved as `.hdf5/.h5` (dataset key `data`)
3. (Optional) **Open Ground Truth…** to compute Dice (if your GUI build includes it).
4. Use mouse wheel, click-to-move crosshair, label isolation, opacity, etc.

---

## Optional: Enable “Run Inference” button (Torch)
If you plan to click **Run Inference**, install PyTorch in the same venv.

CPU-only (works anywhere):
```bash
# Windows
.\.venv\Scripts\python.exe -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# macOS/Linux
./.venv/bin/python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```

GPU builds vary by OS/driver—use PyTorch’s official installer selector if needed:
https://pytorch.org/get-started/locally/
