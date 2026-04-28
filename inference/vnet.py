import torch
import torch.nn as nn
import torch.nn.functional as F



def passthrough(x, **kwargs):
    return x



def ELUCons(elu, nchan):
    return nn.ELU(inplace=True) if elu else nn.PReLU(nchan)


class ContBatchNorm3d(nn.BatchNorm3d):
    """3D BatchNorm for 5D tensors (N, C, D, H, W)."""

    def __init__(self, num_features):
        super().__init__(num_features)


class LUConv(nn.Module):
    def __init__(self, nchan, elu):
        super().__init__()
        self.relu1 = ELUCons(elu, nchan)
        self.conv1 = nn.Conv3d(nchan, nchan, kernel_size=5, padding=2)
        self.bn1 = ContBatchNorm3d(nchan)

    def forward(self, x):
        return self.relu1(self.bn1(self.conv1(x)))



def _make_nConv(nchan, depth, elu):
    return nn.Sequential(*[LUConv(nchan, elu) for _ in range(depth)])


class InputTransition(nn.Module):
    def __init__(self, outChans, elu):
        super().__init__()
        self.conv1 = nn.Conv3d(1, 16, kernel_size=5, padding=2)
        self.bn1 = ContBatchNorm3d(16)
        self.relu1 = ELUCons(elu, 16)

    def forward(self, x):
        out = self.bn1(self.conv1(x))
        x16 = x.repeat(1, 16, 1, 1, 1)
        return self.relu1(out + x16)


class DownTransition(nn.Module):
    def __init__(self, inChans, nConvs, elu, dropout=False):
        super().__init__()
        outChans = 2 * inChans
        self.down_conv = nn.Conv3d(inChans, outChans, kernel_size=2, stride=2)
        self.bn1 = ContBatchNorm3d(outChans)
        self.do1 = nn.Dropout3d() if dropout else passthrough
        self.relu1 = ELUCons(elu, outChans)
        self.relu2 = ELUCons(elu, outChans)
        self.ops = _make_nConv(outChans, nConvs, elu)

    def forward(self, x):
        down = self.relu1(self.bn1(self.down_conv(x)))
        out = self.do1(down)
        out = self.ops(out)
        return self.relu2(out + down)


class UpTransition(nn.Module):
    def __init__(self, inChans, outChans, nConvs, elu, dropout=False):
        super().__init__()
        self.up_conv = nn.ConvTranspose3d(inChans, outChans // 2, kernel_size=2, stride=2)
        self.bn1 = ContBatchNorm3d(outChans // 2)
        self.do1 = nn.Dropout3d() if dropout else passthrough
        self.do2 = nn.Dropout3d()
        self.relu1 = ELUCons(elu, outChans // 2)
        self.relu2 = ELUCons(elu, outChans)
        self.ops = _make_nConv(outChans, nConvs, elu)

    def forward(self, x, skipx):
        out = self.do1(x)
        skipxdo = self.do2(skipx)
        out = self.relu1(self.bn1(self.up_conv(out)))
        xcat = torch.cat((out, skipxdo), dim=1)
        out = self.ops(xcat)
        return self.relu2(out + xcat)


class OutputTransition(nn.Module):
    def __init__(self, inChans, elu, nll, num_classes=2):
        super().__init__()
        self.num_classes = num_classes
        self.conv1 = nn.Conv3d(inChans, num_classes, kernel_size=5, padding=2)
        self.bn1 = ContBatchNorm3d(num_classes)
        self.conv2 = nn.Conv3d(num_classes, num_classes, kernel_size=1)
        self.relu1 = ELUCons(elu, num_classes)
        self.use_nll = nll

    def forward(self, x):
        out = self.relu1(self.bn1(self.conv1(x)))
        out = self.conv2(out)
        # (N, C, D, H, W) -> (N * D * H * W, C)
        out = out.permute(0, 2, 3, 4, 1).contiguous().view(-1, self.num_classes)
        if self.use_nll:
            return F.log_softmax(out, dim=1)
        return F.softmax(out, dim=1)


class VNet(nn.Module):
    def __init__(self, elu=True, nll=False, num_classes=2):
        super().__init__()
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
        return self.out_tr(out)
