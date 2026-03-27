#!/usr/bin/env python3
import os

os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"

import argparse
import h5py
import numpy as np
import torch

import torch
import torch.nn as nn
import torch.nn.functional as F


def passthrough(x, **kwargs):
    return x

def ELUCons(elu, nchan):
    if elu:
        return nn.ELU(inplace=True)
    else:
        return nn.PReLU(nchan)

# normalization between sub-volumes is necessary
# for good performance
class ContBatchNorm3d(nn.BatchNorm3d):
    """
    3D BatchNorm for 5D tensors (N, C, D, H, W).
    This just wraps the standard BatchNorm3d.
    """
    def __init__(self, num_features):
        super(ContBatchNorm3d, self).__init__(num_features)


class LUConv(nn.Module):
    def __init__(self, nchan, elu):
        super(LUConv, self).__init__()
        self.relu1 = ELUCons(elu, nchan)
        self.conv1 = nn.Conv3d(nchan, nchan, kernel_size=5, padding=2)
        self.bn1 = ContBatchNorm3d(nchan)

    def forward(self, x):
        out = self.relu1(self.bn1(self.conv1(x)))
        return out


def _make_nConv(nchan, depth, elu):
    layers = []
    for _ in range(depth):
        layers.append(LUConv(nchan, elu))
    return nn.Sequential(*layers)


class InputTransition(nn.Module):
    def __init__(self, outChans, elu):
        super(InputTransition, self).__init__()
        self.conv1 = nn.Conv3d(1, 16, kernel_size=5, padding=2)
        self.bn1 = ContBatchNorm3d(16)
        self.relu1 = ELUCons(elu, 16)

    def forward(self, x):
        out = self.bn1(self.conv1(x))
        # Repeat 1-channel input into 16 channels along C dimension (dim=1)
        x16 = x.repeat(1, 16, 1, 1, 1)  # (N, 1, D, H, W) -> (N, 16, D, H, W)
        out = self.relu1(out + x16)
        return out


class DownTransition(nn.Module):
    def __init__(self, inChans, nConvs, elu, dropout=False):
        super(DownTransition, self).__init__()
        outChans = 2*inChans
        self.down_conv = nn.Conv3d(inChans, outChans, kernel_size=2, stride=2)
        self.bn1 = ContBatchNorm3d(outChans)
        self.do1 = passthrough
        self.relu1 = ELUCons(elu, outChans)
        self.relu2 = ELUCons(elu, outChans)
        if dropout:
            self.do1 = nn.Dropout3d()
        self.ops = _make_nConv(outChans, nConvs, elu)

    def forward(self, x):
        down = self.relu1(self.bn1(self.down_conv(x)))
        out = self.do1(down)
        out = self.ops(out)
        out = self.relu2(torch.add(out, down))
        return out


class UpTransition(nn.Module):
    def __init__(self, inChans, outChans, nConvs, elu, dropout=False):
        super(UpTransition, self).__init__()
        self.up_conv = nn.ConvTranspose3d(inChans, outChans // 2, kernel_size=2, stride=2)
        self.bn1 = ContBatchNorm3d(outChans // 2)
        self.do1 = passthrough
        self.do2 = nn.Dropout3d()
        self.relu1 = ELUCons(elu, outChans // 2)
        self.relu2 = ELUCons(elu, outChans)
        if dropout:
            self.do1 = nn.Dropout3d()
        self.ops = _make_nConv(outChans, nConvs, elu)

    def forward(self, x, skipx):
        out = self.do1(x)
        skipxdo = self.do2(skipx)
        out = self.relu1(self.bn1(self.up_conv(out)))
        xcat = torch.cat((out, skipxdo), 1)
        out = self.ops(xcat)
        out = self.relu2(torch.add(out, xcat))
        return out


class OutputTransition(nn.Module):
    def __init__(self, inChans, elu, nll, num_classes=2):
        super(OutputTransition, self).__init__()
        self.num_classes = num_classes
        self.conv1 = nn.Conv3d(inChans, num_classes, kernel_size=5, padding=2)
        self.bn1 = ContBatchNorm3d(num_classes)
        self.conv2 = nn.Conv3d(num_classes, num_classes, kernel_size=1)
        self.relu1 = ELUCons(elu, num_classes)
        if nll:
            # log_softmax over channel dimension
            self.softmax = lambda x: F.log_softmax(x, dim=1)
        else:
            self.softmax = lambda x: F.softmax(x, dim=1)

    def forward(self, x):
        # convolve down to num_classes channels
        out = self.relu1(self.bn1(self.conv1(x)))
        out = self.conv2(out)

        # make channels the last axis: (N, C, D, H, W) -> (N, D, H, W, C)
        out = out.permute(0, 2, 3, 4, 1).contiguous()
        # flatten voxels: (N, D, H, W, C) -> (Nvoxels, C)
        out = out.view(-1, self.num_classes)
        out = self.softmax(out)
        return out


class VNet(nn.Module):
    # the number of convolutions in each layer corresponds
    # to what is in the actual prototxt, not the intent
    def __init__(self, elu=True, nll=False, num_classes=2):
        super(VNet, self).__init__()
        self.in_tr = InputTransition(16, elu)
        self.down_tr32 = DownTransition(16, 1, elu)
        self.down_tr64 = DownTransition(32, 2, elu)
        self.down_tr128 = DownTransition(64, 3, elu, dropout=True)
        self.down_tr256 = DownTransition(128, 2, elu, dropout=True)
        self.up_tr256 = UpTransition(256, 256, 2, elu, dropout=True)
        self.up_tr128 = UpTransition(256, 128, 2, elu, dropout=True)
        self.up_tr64 = UpTransition(128, 64, 1, elu)
        self.up_tr32 = UpTransition(64, 32, 1, elu)
        self.out_tr = OutputTransition(32, elu, nll, num_classes=num_classes)

    # The network topology as described in the diagram
    # in the VNet paper
    # def __init__(self):
    #     super(VNet, self).__init__()
    #     self.in_tr =  InputTransition(16)
    #     # the number of convolutions in each layer corresponds
    #     # to what is in the actual prototxt, not the intent
    #     self.down_tr32 = DownTransition(16, 2)
    #     self.down_tr64 = DownTransition(32, 3)
    #     self.down_tr128 = DownTransition(64, 3)
    #     self.down_tr256 = DownTransition(128, 3)
    #     self.up_tr256 = UpTransition(256, 3)
    #     self.up_tr128 = UpTransition(128, 3)
    #     self.up_tr64 = UpTransition(64, 2)
    #     self.up_tr32 = UpTransition(32, 1)
    #     self.out_tr = OutputTransition(16)
    def forward(self, x):
        out16 = self.in_tr(x)
        out32 = self.down_tr32(out16)
        out64 = self.down_tr64(out32)
        out128 = self.down_tr128(out64)
        out256 = self.down_tr256(out128)
        out = self.up_tr256(out256, out128)
        out = self.up_tr128(out, out64)
        out = self.up_tr64(out, out32)
        out = self.up_tr32(out, out16)
        out = self.out_tr(out)
        return out

def load_volume_h5(path: str) -> np.ndarray:
    with h5py.File(path, "r") as f:
        vol = np.array(f["data"])
    return vol.astype(np.float32)

def preprocess_like_training(vol: np.ndarray, normalize: bool = True) -> torch.Tensor:
    # training did: mean/std normalize + add channel dim
    if normalize:
        m = float(vol.mean())
        s = float(vol.std())
        if s > 0:
            vol = (vol - m) / s

    # Expect vol shape is (D,H,W) like your training logs imply (384,384,160)
    vol = np.expand_dims(vol, axis=0)   # (1,D,H,W)
    vol = np.expand_dims(vol, axis=0)   # (N=1,C=1,D,H,W)
    return torch.from_numpy(vol)

def strip_module_prefix(state_dict):
    # If saved from DataParallel, keys look like "module.xxx"
    if any(k.startswith("module.") for k in state_dict.keys()):
        return {k.replace("module.", "", 1): v for k, v in state_dict.items()}
    return state_dict

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, help="path to vnet_model_best.pth.tar")
    ap.add_argument("--im", required=True, help="path to input .im (HDF5) file")
    ap.add_argument("--out", required=True, help="output path for predicted labels (e.g., pred.npy)")
    ap.add_argument("--no-normalize", action="store_true")
    ap.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    args = ap.parse_args()

    device = torch.device(args.device if (args.device == "cpu" or torch.cuda.is_available()) else "cpu")

    # 1) Build model exactly like training
    num_classes = 7
    model = VNet(elu=False, nll=True, num_classes=num_classes).to(device)
    model.eval()

    # 2) Load checkpoint
    ckpt = torch.load(args.ckpt, map_location=device)

    # Your save_checkpoint stored: {'epoch', 'state_dict', 'best_prec1'}
    state_dict = ckpt["state_dict"] if isinstance(ckpt, dict) and "state_dict" in ckpt else ckpt
    state_dict = strip_module_prefix(state_dict)
    model.load_state_dict(state_dict, strict=True)

    # 3) Load + preprocess input volume
    print("[LOG] Preprocessing...")
    vol = load_volume_h5(args.im)
    x = preprocess_like_training(vol, normalize=(not args.no_normalize)).to(device)

    # 4) Forward pass
    print("[LOG] Inferencing...")
    with torch.no_grad():
        # Your model returns log-probs flattened: (Nvoxels, C)
        out_flat = model(x)

        pred_flat = torch.argmax(out_flat, dim=1)  # (D*H*W,)

    # 5) Reshape back to 3D
    # x shape: (1,1,D,H,W)
    _, _, D, H, W = x.shape
    pred_3d = pred_flat.view(D, H, W).cpu().numpy().astype(np.uint8)

    # 6) Save output
    out_path = args.out
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)

    if out_path.endswith(".npy"):
        np.save(out_path, pred_3d)
        print(f"[LOG] Saved predicted labels to {out_path} with shape {pred_3d.shape}")
    elif out_path.endswith(".h5") or out_path.endswith(".seg"):
        # HDF5 style (like your training segs): dataset key "data"
        with h5py.File(out_path, "w") as f:
            f.create_dataset("data", data=pred_3d, compression="gzip")
        print(f"[LOG] Saved predicted labels to {out_path} (HDF5 dataset 'data'), shape {pred_3d.shape}")
    else:
        # default: save npy if extension unknown
        np.save(out_path + ".npy", pred_3d)
        print(f"[ERROR] Unknown extension; saved to {out_path}.npy, shape {pred_3d.shape}")

if __name__ == "__main__":
    main()
