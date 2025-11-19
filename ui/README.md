# MRI UI for Clinicians — Skeleton (No Logic)

Everything is layout-only right now; TODO: Wire Logic Later.

## Run
```bash
python3 -m venv .venv
# macOS/Linux:
source .venv/bin/activate
# Windows:
# .venv\Scripts\activate

pip install --upgrade pip wheel setuptools
# Use --no-compile to avoid byte-compiling template files inside the wheel
pip install --no-compile pyside6==6.7.3
python3 ui/clinician/clinician_ui.py

# For Qt plugin errors on MacOS
export LANG=en_US.UTF-8
export LC_ALL=en_US.UTF-8
export PYTHONIOENCODING=UTF-8
pip uninstall -y PySide6 PySide6-Addons PySide6-Essentials shiboken6
pip install --upgrade pip wheel setuptools
# Use --no-compile to avoid byte-compiling template files inside the wheel
pip install --no-compile pyside6==6.7.3