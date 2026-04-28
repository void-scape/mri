import torch
import torch.nn as nn
import unet as unet
from torch.utils.checkpoint import checkpoint


class Triplanar(nn.Module):
    def __init__(self, channels, features, chunks):
        super().__init__()
        self.chunks = chunks
        self.sagittal = unet.UNet(channels, features)
        self.coronal = unet.UNet(channels, features)
        self.axial = unet.UNet(channels, features)

    def process_chunks(self, x, model):
        outputs = []
        for i in range(0, x.size(0), self.chunks):
            chunk = x[i : i + self.chunks]
            out = checkpoint(model, chunk, use_reentrant=False)
            outputs.append(out)
        return torch.cat(outputs, dim=0)

    def forward(self, sagittal, coronal, axial):
        sagittal = self.process_chunks(sagittal, self.sagittal)
        coronal = self.process_chunks(coronal, self.coronal)
        axial = self.process_chunks(axial, self.axial)

        # axes as defined in the dataloader
        x, c, y, z = 0, 1, 2, 3
        sagittal = sagittal.permute(c, x, y, z)
        z, c, x, y = 0, 1, 2, 3
        coronal = coronal.permute(c, x, y, z)
        y, c, z, x = 0, 1, 2, 3
        axial = axial.permute(c, x, y, z)

        sagittal.add_(coronal).add_(axial)
        del coronal, axial
        return sagittal.contiguous()


def dice(logits, targets, epsilon=1e-6):
    logits = logits.squeeze(0)

    assert len(logits.shape) == 3
    assert logits.shape == targets.shape

    pred = torch.sigmoid(logits)
    pred = pred.flatten()
    targ = targets.flatten()
    inter = (pred * targ).sum()
    pred_sum = pred.sum()
    targ_sum = targ.sum()
    assert inter.sum() > 0.0
    scores = (2 * inter + epsilon) / (pred_sum + targ_sum + epsilon)
    return scores.mean()


class Loss(nn.Module):
    def __init__(self, epsilon=1e-6):
        super().__init__()
        self.epsilon = epsilon

    def forward(self, logits, targets):
        return 1.0 - dice(logits=logits, targets=targets, epsilon=self.epsilon)
