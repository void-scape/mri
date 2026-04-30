## TODO
- Be specific about training VRAM requirements in the README
- Add `--no-normalize` and `--profile-memory` to `infer.py` and `triplanar/infer.py`
- Add support in infer scripts for `npy` output
- Unify normalizing input for inference scripts
- Bring back cuda device selection in `viewer.py` after showcase
- Triplanar normalization transposes high res to match low res
  - We need a long term fix so that the vnet inference can use that normalized data

## Added
- `triplanar/`
- `infer.py` to delegate to `vnet/infer.py` and `triplanar/infer.py`
- Architecture GUI selection for `infer.py`
- Setting a new image automatically clears the ground truth/mask
- Hardcoded `self.spacing_xyz` defined by the dimensions of the datasets
  - We should have a better solution for this
  - I deleted the zoom stuff in `_set_pix`, I didn't see any way to use this in the viewer?
    - Nvm, I need to go back and fix this after showcase!

## Changed
- Moved `viewer/main.py` to `viewer.py`
- Moved `viewer/README.md` information to `README.md`
- Moved `RUN_VIEWER_INSTRUCTIONS.md` information to `README.md`
- `DSC vs Ground Truth` to `Statistics`
- Reordered viewers to `Sagittal`, `Coronal`, and `Axial`

## Removed
- `viewer/`
- `RUN_VIEWER_INSTRUCTIONS.md`
- `stretch-goal-ui/`
- `cli/`, replaced by `infer.py`
- `shared/`, replaced by `infer.py`
- `inference/`, replaced by `infer.py`
- `checkpoints/`
  - When we have four possible models to run, it no longer makes sense to automatically choose a checkpoint, it is just going to have to be selected by the user.
- `RUN_VIEWER_INSTRUCTIONS.md`
- Dead code from `viewer.py`
- Checking for dependencies in `viewer.py`, just assume requirements.txt is installed
- Auto contrast from `viewer.py`
  - Images seem to automatically scale.
- Clear ground truth/mask from `viewer.py`
