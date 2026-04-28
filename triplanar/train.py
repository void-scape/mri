import os

os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"

import torch
import torch.optim as optim
import wandb
import dataloader
import argparse
from triplanar import Triplanar, Loss
from torch.utils.data import DataLoader
from evaluate import evaluate
from tqdm import tqdm


def train_triplanar_model(
    args,
    train,
    valid,
    model,
    device,
    checkpoints,
    epochs=5,
    learning_rate=1e-3,
    weight_decay=0.0,
):
    """
    Implementation based on:
    - https://www.codegenes.net/blog/logisitic-regression-pytorch/
    - https://github.com/milesial/Pytorch-UNet
    """

    model = model.to(device)
    optimizer = optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    criterion = Loss()
    scaler = torch.GradScaler(enabled=args.amp and device.type == "cuda")

    stop_early = False
    best_mean_dice = 0.0
    patience = 20
    counter = 0
    num_train_batches = len(train)
    start_epoch = 0
    epoch_step = 0

    print(f"[triplanar/train.py] Training for {args.dataset} dataset")

    if args.resume is not None:
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])
        start_epoch = ckpt["epoch"] + 1
        best_mean_dice = ckpt["best_mean_dice"]
        print(f"[triplanar/train.py] Resuming {args.resume}")

    if args.seed is not None:
        ckpt = torch.load(args.seed, map_location=device)
        model.load_state_dict(ckpt["model"])
        print(f"[triplanar/train.py] Seeding {args.seed}")

    if args.wandb:
        experiment = wandb.init(project="Knee Cartilage Triplanar")
        experiment.config.update(
            dict(
                epochs=epochs,
                weight_decay=weight_decay,
                learning_rate=learning_rate,
                device=device.type,
                amp=args.amp,
            )
        )
        experiment.watch(model, criterion=criterion, log="all", log_freq=100)

    for epoch in range(start_epoch, epochs):
        match args.dataset:
            case "7t":
                eval_steps = {num_train_batches - 1}
            case "3t":
                eval_steps = {
                    max(
                        0,
                        min(int(num_train_batches * p / 5) - 1, num_train_batches - 1),
                    )
                    for p in range(1, 6)
                }

        model.train()
        with tqdm(
            total=num_train_batches, desc=f"Epoch {epoch}/{epochs}", unit="img"
        ) as pbar:
            epoch_step = 0
            for batch_index, (volumes, mask) in enumerate(train):
                pbar.update(volumes[0].shape[0])

                optimizer.zero_grad(set_to_none=True)
                with torch.autocast(
                    device_type=device.type if device.type != "mps" else "cpu",
                    enabled=args.amp,
                ):
                    # data loader automatically appends an annoying batch
                    logits = model(*[v.squeeze(0) for v in volumes])
                    loss = criterion(logits, mask.squeeze(0))

                scaler.scale(loss).backward()
                del volumes, mask, logits

                # NOTE: I don't experience exploding gradients and this has slowed
                # down training in the past. I believe this sort of thing is best
                # practice however...
                # scaler.unscale_(optimizer)
                # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

                scaler.step(optimizer)
                scaler.update()

                pbar.set_postfix(**{"loss (batch)": loss.item()})
                if args.wandb:
                    experiment.log({"train loss": loss.item(), "epoch": epoch})

                del loss

                if batch_index in eval_steps:
                    mean_dice, log_images = evaluate(
                        model, valid, device, epoch, amp=args.amp
                    )

                    if args.wandb:
                        experiment.log(
                            {
                                "learning rate": optimizer.param_groups[0]["lr"],
                                "validation Dice": mean_dice,
                                "epoch": epoch,
                                **log_images,
                            }
                        )

                    model_path = f"{checkpoints}/model_e{epoch}_s{epoch_step}.pth"
                    epoch_step += 1
                    torch.save(
                        {
                            "model": model.state_dict(),
                            "optimizer": optimizer.state_dict(),
                            "scheduler": scheduler.state_dict(),
                            "epoch": epoch,
                            "best_mean_dice": best_mean_dice,
                        },
                        model_path,
                    )
                    print(
                        f"[triplanar/train.py] Saved model to {model_path}, dsc: {mean_dice:.2f}"
                    )

                    if mean_dice > best_mean_dice:
                        model_path = f"{checkpoints}/best_model.pth"
                        torch.save(
                            {
                                "model": model.state_dict(),
                                "optimizer": optimizer.state_dict(),
                                "scheduler": scheduler.state_dict(),
                                "epoch": epoch,
                                "best_mean_dice": best_mean_dice,
                            },
                            model_path,
                        )
                        print(
                            f"[triplanar/train.py] Saved best model to {model_path}, dsc: {mean_dice:.2f}"
                        )
                        best_mean_dice = mean_dice
                        counter = 0
                    else:
                        counter += 1

                    if counter >= patience:
                        print("[triplanar/train.py] Patience exhausted")
                        stop_early = True
                        break

        if stop_early:
            break

        scheduler.step()

    return best_mean_dice


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("-d", "--dataset", required=True, choices=["3t", "7t"])
    ap.add_argument(
        "-c", "--checkpoints", required=True, help="path for saving model checkpoints"
    )
    ap.add_argument(
        "-i",
        "--input",
        required=True,
        help="input data path, should contain valid/ and train/ directories",
    )
    ap.add_argument(
        "--device",
        default="cuda",
        choices=["cuda", "cpu"],
        help="select preferred torch device",
    )
    ap.add_argument(
        "-w", "--wandb", action="store_true", help="log training with wandb"
    )
    ap.add_argument("--resume", help="resume training from a checkpoint")
    ap.add_argument("--seed", help="seed model weights from a checkpoint")
    ap.add_argument("--amp", action="store_true")
    ap.add_argument("--chunks", default=1, type=int, help="maximum unet batch size")
    return ap.parse_args()


def main():
    args = parse_args()
    if torch.cuda.is_available() and args.device != "cpu":
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print(f"[triplanar/train.py] Chosen device {device}")

    train = DataLoader(
        dataloader.HDF5Dataset(
            data_dir=f"{args.input}/train",
            device=device,
        ),
        batch_size=1,
        shuffle=True,
    )
    valid = DataLoader(
        dataloader.HDF5Dataset(
            data_dir=f"{args.input}/valid",
            device=device,
        ),
        batch_size=1,
        shuffle=True,
    )
    model = Triplanar(channels=1, features=1, chunks=args.chunks)

    existing = (
        [
            d
            for d in os.listdir(args.checkpoints)
            if os.path.isdir(f"{args.checkpoints}/{d}")
        ]
        if os.path.exists(args.checkpoints)
        else []
    )
    i = len(existing)
    checkpoints = f"{args.checkpoints}/{i}"
    os.makedirs(checkpoints, exist_ok=True)
    print(f"[triplanar/train.py] Saving checkpoints to {checkpoints}")

    best_dice = train_triplanar_model(
        args=args,
        train=train,
        valid=valid,
        model=model,
        device=device,
        epochs=150,
        learning_rate=0.0005,
        weight_decay=0.00001,
        checkpoints=checkpoints,
    )
    print(f"[triplanar/train.py] Best dice achieved: {best_dice}")


if __name__ == "__main__":
    main()
