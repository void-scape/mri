import torch
import wandb
import numpy as np
from tqdm import tqdm


def log_images(volumes, mask, logits):
    logits = logits.squeeze(0)
    assert logits.shape == mask.shape

    pred = torch.sigmoid(logits) > 0.5

    target_x = 225
    coronal = volumes[0]
    coronal_slice = coronal[target_x][0]

    def to_uint8(t):
        return (np.clip(t.cpu().numpy() * 255, a_min=0, a_max=255)).astype(np.uint8)

    images = [wandb.Image(to_uint8(coronal_slice), caption="coronal input")]

    z, x, y = 0, 1, 2
    mask = mask.permute(x, y, z)
    pred = pred.permute(x, y, z)

    images.append(wandb.Image(to_uint8(mask[target_x]), caption="mask"))
    images.append(wandb.Image(to_uint8(pred[target_x]), caption="pred"))

    return {"val/images": images}


def dice(logits, targets, epsilon=1e-6):
    logits = logits.squeeze(0)
    assert logits.shape == targets.shape

    pred = torch.sigmoid(logits) > 0.5
    pred = pred.flatten()
    targ = targets.flatten()
    inter = (pred * targ).sum()
    pred_sum = pred.sum()
    targ_sum = targ.sum()
    assert inter.sum() > 0.0
    scores = (2 * inter + epsilon) / (pred_sum + targ_sum + epsilon)
    return scores.mean()


@torch.inference_mode()
def evaluate(model, dataloader, device, epoch, amp):
    model.eval()
    num_val_batches = len(dataloader)
    assert num_val_batches > 0
    dice_score = 0.0
    min_score = 1.0
    max_score = 0.0
    images = None
    all_scores = []

    with torch.autocast(device.type if device.type != "mps" else "cpu", enabled=amp):
        for volumes, mask in tqdm(
            dataloader,
            total=num_val_batches,
            desc="Validation round",
            unit="batch",
            leave=False,
        ):
            # NOTE: squeezing away the batch dimension appended by the dataloader
            # (b, ...) -> (...)
            volumes = [v.squeeze(0) for v in volumes]
            logits = model(*volumes)
            mask = mask.squeeze(0)
            score = dice(logits=logits, targets=mask)
            all_scores.append(score.item())
            dice_score += score
            if score < min_score:
                min_score = score
            if score > max_score:
                max_score = score
            if images is None:
                images = log_images(volumes, mask, logits)
            del volumes, logits, mask

    all_scores = np.array(all_scores, dtype=np.float64)
    std_dice = np.std(all_scores, dtype=np.float64)

    model.train()
    mean_dice = dice_score / max(num_val_batches, 1)
    print("Validation dice stats")
    print(f"  min: {min_score}")
    print(f"  max: {max_score}")
    print(f"  avg: {mean_dice}")
    print(f"  std: {std_dice}")
    return (mean_dice, images)
