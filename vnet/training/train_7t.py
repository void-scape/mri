#!/usr/bin/env python3

import time
import argparse
import os
import shutil

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader

import setproctitle

from ..utils.local import *
from ..data_loaders.dataset_7t import Knee7TDataset
from ..models import vnet

def strip_module_prefix(state_dict):
    if any(k.startswith("module.") for k in state_dict.keys()):
        return {k.replace("module.", "", 1): v for k, v in state_dict.items()}
    return state_dict

def print_gpu_memory(tag=""):
    if torch.cuda.is_available():
        alloc = torch.cuda.max_memory_allocated() / (1024 ** 3)
        reserv = torch.cuda.max_memory_reserved() / (1024 ** 3)
        print(f"\n[{tag}] Peak allocated: {alloc:.3f} GB")
        print(f"[{tag}] Peak reserved:  {reserv:.3f} GB\n")

def reset_gpu_peak_memory():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv3d') != -1:
        nn.init.kaiming_normal_(m.weight)
        if m.bias is not None:
            m.bias.data.zero_()


def datestr():
    now = time.gmtime()
    return '{}{:02}{:02}_{:02}{:02}'.format(
        now.tm_year, now.tm_mon, now.tm_mday, now.tm_hour, now.tm_min
    )


def save_checkpoint(state, is_best, path, prefix, filename='checkpoint.pth.tar'):
    prefix_save = os.path.join(path, prefix)
    name = prefix_save + '_' + filename
    torch.save(state, name)
    if is_best:
        shutil.copyfile(name, prefix_save + '_model_best.pth.tar')


def multiclass_dice(pred, target, num_classes, ignore_index=0, eps=1e-5):
    dice_scores = []

    for c in range(num_classes):
        if c == ignore_index:
            continue

        pred_c = (pred == c).float()
        target_c = (target == c).float()

        if target_c.sum() == 0 and pred_c.sum() == 0:
            continue

        intersection = (pred_c * target_c).sum()
        denom = pred_c.sum() + target_c.sum()
        dice_c = (2.0 * intersection + eps) / (denom + eps)
        dice_scores.append(dice_c)

    if len(dice_scores) == 0:
        return pred.new_tensor(0.0)

    return torch.stack(dice_scores).mean()


def soft_dice_loss_flat(log_probs, target, num_classes, ignore_index=0, eps=1e-5):
    probs = log_probs.exp()
    dice_terms = []

    for c in range(num_classes):
        if c == ignore_index:
            continue

        p_c = probs[:, c]
        t_c = (target == c).float()

        if t_c.sum() == 0 and p_c.sum() == 0:
            continue

        intersection = (p_c * t_c).sum()
        denom = p_c.sum() + t_c.sum()
        dice_c = (2.0 * intersection + eps) / (denom + eps)
        dice_terms.append(dice_c)

    if len(dice_terms) == 0:
        return log_probs.new_tensor(0.0)

    mean_dice = torch.stack(dice_terms).mean()
    return 1.0 - mean_dice


def load_partial_checkpoint(model, checkpoint_path):
    print(f"=> loading pretrained weights from '{checkpoint_path}'")
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    pretrained_state = checkpoint['state_dict']
    pretrained_state = strip_module_prefix(pretrained_state)
    model_state = model.state_dict()

    compatible = {
        k: v for k, v in pretrained_state.items()
        if k in model_state and model_state[k].shape == v.shape
    }

    print(f"=> loaded {len(compatible)} matching tensors")
    model_state.update(compatible)
    model.load_state_dict(model_state, strict=False)


def train_epoch(args, epoch, model, trainLoader, optimizer, trainF, weights, scaler,
                ce_weight=0.3, dice_weight=0.7):
    model.train()
    nProcessed = 0
    nTrain = len(trainLoader.dataset)

    for batch_idx, (data, target) in enumerate(trainLoader):
        if batch_idx == 0:
            print("data shape:", data.shape)
            print("target shape:", target.shape)

        if args.cuda:
            data = data.cuda(non_blocking=True)
            target = target.cuda(non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        with torch.cuda.amp.autocast(enabled=args.cuda):
            output = model(data)
            target_flat = target.view(-1)

            ce_loss = F.nll_loss(output, target_flat, weight=weights)
            dice_loss = soft_dice_loss_flat(
                output,
                target_flat,
                num_classes=output.size(1),
                ignore_index=0
            )
            loss = ce_weight * ce_loss + dice_weight * dice_loss

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        nProcessed += len(data)

        pred = output.detach().max(1)[1]
        incorrect = pred.ne(target_flat).sum().item()
        err = 100.0 * incorrect / target_flat.numel()

        num_classes = output.size(1)
        dice = multiclass_dice(
            pred.detach(),
            target_flat.detach(),
            num_classes,
            ignore_index=0
        ).item()

        partialEpoch = epoch + batch_idx / len(trainLoader) - 1
        print(
            'Train Epoch: {:.2f} [{}/{} ({:.0f}%)]\t'
            'Loss: {:.4f}\tError: {:.3f}%\tDice: {:.4f}'.format(
                partialEpoch,
                nProcessed,
                nTrain,
                100.0 * batch_idx / len(trainLoader),
                float(loss.item()),
                float(err),
                float(dice),
            )
        )

        trainF.write('{},{},{},{}\n'.format(
            partialEpoch, float(loss.item()), float(err), float(dice)
        ))
        trainF.flush()


def validate_epoch(args, epoch, model, testLoader, testF, weights,
                   ce_weight=0.3, dice_weight=0.7):
    model.eval()
    test_loss = 0.0
    incorrect = 0
    numel = 0
    dice_accum = 0.0
    dice_batches = 0

    class_dice_sum = None
    class_counts = None
    num_classes = None

    with torch.no_grad():
        for data, target in testLoader:
            if args.cuda:
                data = data.cuda(non_blocking=True)
                target = target.cuda(non_blocking=True)

            target_flat = target.view(-1)
            numel += target_flat.numel()

            with torch.cuda.amp.autocast(enabled=args.cuda):
                output = model(data)
                ce_loss = F.nll_loss(output, target_flat, weight=weights)
                dice_loss = soft_dice_loss_flat(
                    output,
                    target_flat,
                    num_classes=output.size(1),
                    ignore_index=0
                )
                loss = ce_weight * ce_loss + dice_weight * dice_loss

            test_loss += float(loss.item())

            pred = output.detach().max(1)[1]
            incorrect += pred.ne(target_flat).sum().item()

            if num_classes is None:
                num_classes = output.size(1)
                class_dice_sum = torch.zeros(num_classes, dtype=torch.float64)
                class_counts = torch.zeros(num_classes, dtype=torch.float64)

            for c in range(1, num_classes):
                pred_c = (pred == c)
                target_c = (target_flat == c)

                if target_c.sum() == 0:
                    continue

                intersection = (pred_c & target_c).sum().float()
                denom = pred_c.sum().float() + target_c.sum().float()
                dice_c = (2.0 * intersection + 1e-5) / (denom + 1e-5)

                class_dice_sum[c] += dice_c.item()
                class_counts[c] += 1.0

            dice = multiclass_dice(
                pred.detach(),
                target_flat.detach(),
                num_classes,
                ignore_index=0
            ).item()
            dice_accum += dice
            dice_batches += 1

    test_loss /= len(testLoader)
    err = 100.0 * incorrect / numel
    mean_dice = dice_accum / max(dice_batches, 1)

    if num_classes is not None:
        print("\nPer-class Dice:")
        for c in range(1, num_classes):
            if class_counts[c] > 0:
                avg_dice = class_dice_sum[c] / class_counts[c]
                print(f"Class {c}: {avg_dice:.4f}")

    print(
        '\nTest set: Average loss: {:.4f}, Error: {}/{} ({:.3f}%), '
        'Mean Dice: {:.4f}\n'.format(
            test_loss, incorrect, numel, float(err), float(mean_dice)
        )
    )

    testF.write('{},{},{},{}\n'.format(epoch, test_loss, float(err), float(mean_dice)))
    testF.flush()
    return err, mean_dice


def maybe_freeze_for_warmup(model, freeze=True):
    for name, param in model.named_parameters():
        # keep final output-related layers trainable
        if any(x in name.lower() for x in ["out", "final"]):
            param.requires_grad = True
        else:
            param.requires_grad = not freeze


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batchSz', type=int, default=1)
    parser.add_argument('--ngpu', type=int, default=1)
    parser.add_argument('--nEpochs', type=int, default=150)
    parser.add_argument('--start-epoch', default=0, type=int, metavar='N')
    parser.add_argument('--resume', default='', type=str, metavar='PATH')
    parser.add_argument('--pretrained', default='', type=str,
                        help='path to pretrained low-res checkpoint')
    parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true')
    parser.add_argument('--data', default='7T_Data/Neal_7T_Cartilages_20200504.hdf5', type=str)
    parser.add_argument('--weight-decay', '--wd', default=1e-8, type=float, metavar='W')
    parser.add_argument('--no-cuda', action='store_true')
    parser.add_argument('--save')
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--opt', type=str, default='adam',
                        choices=('sgd', 'adam', 'rmsprop'))
    parser.add_argument('--warmup-epochs', type=int, default=5)
    parser.add_argument('--profile-memory', action='store_true',
                    help='run one training batch and report peak GPU memory, then exit')
    args = parser.parse_args()

    best_dice = -1.0
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    args.save = args.save or 'work/vnet.7t_transfer.{}'.format(datestr())
    weight_decay = args.weight_decay

    setproctitle.setproctitle(args.save)

    torch.manual_seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed(args.seed)

    kwargs = {'num_workers': 0, 'pin_memory': True} if args.cuda else {}
    scaler = torch.cuda.amp.GradScaler(enabled=args.cuda)

    print("build vnet")
    num_classes = 2
    nll = True
    model = vnet.VNet(elu=False, nll=nll, num_classes=num_classes)

    batch_size = args.ngpu * args.batchSz
    gpu_ids = list(range(args.ngpu))
    if args.ngpu > 1:
        model = nn.DataParallel(model, device_ids=gpu_ids)
    if args.cuda:
        model = model.cuda()

    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume, map_location='cpu')
            args.start_epoch = checkpoint['epoch']
            best_dice = checkpoint.get('best_dice', -1.0)
            state_dict = strip_module_prefix(checkpoint['state_dict'])
            model.load_state_dict(state_dict)
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))
    else:
        model.apply(weights_init)
        if args.pretrained:
            load_partial_checkpoint(model, args.pretrained)

    print('  + Number of params: {}'.format(
        sum([p.data.nelement() for p in model.parameters()])
    ))

    if not args.resume:
        if os.path.exists(args.save):
            shutil.rmtree(args.save)
        os.makedirs(args.save, exist_ok=True)
    else:
        os.makedirs(args.save, exist_ok=True)

    h5_path = args.data
    slices_per_volume = 80

    train_ids = [0,1,2,3,4,5,6,7,8,9]
    valid_ids = [10,11,12,13]

    print("loading 7T training set")
    trainSet = Knee7TDataset(
        h5_path,
        image_ids=train_ids,
        slices_per_volume=slices_per_volume,
        normalize=True,
        augment=False,
        full_volume=True,
    )

    trainLoader = DataLoader(
        trainSet,
        batch_size=batch_size,
        shuffle=True,
        **kwargs
    )

    print("loading 7T validation set")
    testSet = Knee7TDataset(
        h5_path,
        image_ids=valid_ids,
        slices_per_volume=slices_per_volume,
        normalize=True,
        augment=False,
        full_volume=True,
    )
    testLoader = DataLoader(
        testSet,
        batch_size=batch_size,
        shuffle=False,
        **kwargs
    )

    # background downweighted, cartilage classes emphasized equally
    class_weights = torch.tensor([0.05, 2.0], dtype=torch.float32)
    if args.cuda:
        class_weights = class_weights.cuda()

    if args.opt == 'sgd':
        optimizer = optim.SGD(
            model.parameters(),
            lr=1e-2,
            momentum=0.99,
            weight_decay=weight_decay
        )
    elif args.opt == 'adam':
        optimizer = optim.Adam(
            model.parameters(),
            lr=5e-5,
            weight_decay=weight_decay
        )
    elif args.opt == 'rmsprop':
        optimizer = optim.RMSprop(
            model.parameters(),
            lr=1e-4,
            weight_decay=weight_decay
        )

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='max',
        factor=0.5,
        patience=15,
        threshold=1e-4,    
        cooldown=5,
        min_lr=1e-6,
        verbose=True
    )

    csv_mode = 'a' if args.resume else 'w'
    trainF = open(os.path.join(args.save, 'train.csv'), csv_mode)
    testF = open(os.path.join(args.save, 'test.csv'), csv_mode)

    if args.profile_memory:
        print("=== PROFILING HIGH-RES TRAINING (1 BATCH) ===")

        reset_gpu_peak_memory()

        data, target = next(iter(trainLoader))

        if args.cuda:
            data = data.cuda(non_blocking=True)
            target = target.cuda(non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        with torch.cuda.amp.autocast(enabled=args.cuda):
            output = model(data)
            target_flat = target.view(-1)

            ce_loss = F.nll_loss(output, target_flat, weight=class_weights)
            dice_loss = soft_dice_loss_flat(
                output,
                target_flat,
                num_classes=output.size(1),
                ignore_index=0
            )
            loss = 0.3 * ce_loss + 0.7 * dice_loss

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        if args.cuda:
            torch.cuda.synchronize()
            print_gpu_memory("HIGH-RES TRAIN")

        trainF.close()
        testF.close()
        return

    # optional warmup: freeze most layers briefly when using pretrained weights
    if args.pretrained and not args.resume and args.warmup_epochs > 0:
        maybe_freeze_for_warmup(model, freeze=True)
    else:
        maybe_freeze_for_warmup(model, freeze=False)

    for epoch in range(args.start_epoch + 1, args.start_epoch + args.nEpochs + 1):
        if args.pretrained and not args.resume and epoch == args.warmup_epochs + 1:
            print("Unfreezing full model after warmup.")
            maybe_freeze_for_warmup(model, freeze=False)
            # rebuild optimizer so newly unfrozen params are trainable
            if args.opt == 'sgd':
                optimizer = optim.SGD(
                    filter(lambda p: p.requires_grad, model.parameters()),
                    lr=1e-2,
                    momentum=0.99,
                    weight_decay=weight_decay
                )
            elif args.opt == 'adam':
                optimizer = optim.Adam(
                    filter(lambda p: p.requires_grad, model.parameters()),
                    lr=5e-5,
                    weight_decay=weight_decay
                )
            elif args.opt == 'rmsprop':
                optimizer = optim.RMSprop(
                    filter(lambda p: p.requires_grad, model.parameters()),
                    lr=1e-4,
                    weight_decay=weight_decay
                )

            scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode='max',
                factor=0.5,
                patience=15,
                threshold=1e-4,
                cooldown=5,
                min_lr=1e-6,
                verbose=True
            )

        train_epoch(args, epoch, model, trainLoader, optimizer, trainF, class_weights, scaler,
                    ce_weight=0.3, dice_weight=0.7)
        err, val_dice = validate_epoch(args, epoch, model, testLoader, testF, class_weights,
                                       ce_weight=0.3, dice_weight=0.7)

        scheduler.step(val_dice)
        current_lr = optimizer.param_groups[0]['lr']
        print(f"Epoch {epoch}: val_dice={val_dice:.4f}, val_err={err:.4f}, lr={current_lr:.2e}")

        is_best = val_dice > best_dice
        if is_best:
            best_dice = val_dice

        save_checkpoint(
            {
                'epoch': epoch,
                'state_dict': model.state_dict(),
                'best_dice': best_dice
            },
            is_best,
            args.save,
            "vnet"
        )

    trainF.close()
    testF.close()


if __name__ == '__main__':
    main()