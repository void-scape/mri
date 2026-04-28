# MRI
System to automatically identify and segregate different types of tissue on MRI scans.

### Contributors
- Nic Ball
- Sean Denny
- James Stringham

### Sponsor
- Neal Bangerter

# Quick Start

Install dependencies:

```console
$ python3 -m venv .venv
$ source .venv/bin/activate
$ pip install -r requirements.txt
```

## Training

### Triplanar

The 3T and 7T datasets need to be normalized for the triplanar pipeline before training. Normalization can take quite some time.

```console
$ python3 triplanar/preprocess/low-res.py path/to/train path/to/valid data/3t -v
$ python3 triplanar/preprocess/high-res.py path/to/hdf5 data/7t -v
```

```console
$ python3 triplanar/train.py -d 3t -c triplanar/checkpoints/3t -i data/3t --amp --chunks=8
$ python3 triplanar/train.py -d 7t -c triplanar/checkpoints/7t -i data/7t --amp --chunks=2
```

> [!WARNING]
> `chunks` will determine VRAM usage. If you are running out of memory, reduce `chunks`.

## Infer

Use `infer.py` to automatically invoke `triplanar/infer.py` or `vnet/infer.py` according to the `-a`/`--arch` flag.

```console
$ infer.py -d 3t -a triplanar -c best_model.pth -i path/to/im -o seg.npy
```

Passing a segmentation mask will compute the DSC on the inferred segmentation:

```console
$ infer.py -d 3t -a triplanar -c best_model.pth -i path/to/im -o seg.npy -s path/to/seg
```

## Viewer

```console
$ python3 viewer.py
```

- Navigate orthogonal viewer with the mouse wheel or the primary mouse button.
- `Open Image` populates orthogonal viewer with an `im` input file.
- `Open Mask` overlays segmentation mask.
- `Open Ground Truth` computes the DSC between the selected mask and the active mask.
- `Set Checkpoint` chooses the model.pth weights for inference.
- `Run Inference` computes segmentation mask and overlays when finished. Outputs segmentation mask into `outputs/`.
