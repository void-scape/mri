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
from ..data_loaders.knee_dataset import KneeMRIDataset
from ..models import vnet

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


def noop(x):
    return x

def multiclass_dice(pred, target, num_classes, ignore_index=0, eps=1e-5):
    """
    Compute mean Dice over all classes except ignore_index.

    pred: LongTensor of shape [N]     (predicted class indices)
    target: LongTensor of shape [N]   (ground-truth class indices)
    num_classes: total number of classes (e.g., 7)
    ignore_index: class to ignore in Dice (usually background=0)
    """
    dice_scores = []

    for c in range(num_classes):
        if c == ignore_index:
            continue

        pred_c = (pred == c).float()
        target_c = (target == c).float()

        # If this class is absent in both pred and target, skip it
        if target_c.sum() == 0 and pred_c.sum() == 0:
            continue

        intersection = (pred_c * target_c).sum()
        denom = pred_c.sum() + target_c.sum()
        dice_c = (2.0 * intersection + eps) / (denom + eps)
        dice_scores.append(dice_c.item())

    if len(dice_scores) == 0:
        # no foreground classes present at all
        return 0.0

    return float(sum(dice_scores) / len(dice_scores))

def soft_dice_loss_flat(log_probs, target, num_classes, ignore_index=0, eps=1e-5):
    """
    log_probs: [N, C] from log_softmax output
    target:    [N] class indices
    """
    probs = log_probs.exp()  # convert log-probs to probs

    dice_terms = []
    for c in range(num_classes):
        if c == ignore_index:
            continue

        p_c = probs[:, c]
        t_c = (target == c).float()

        # skip absent class in both pred and target
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


def train_nll(args, epoch, model, trainLoader, optimizer, trainF, weights, scaler):
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
            loss = 0.2 * ce_loss + 0.8 * dice_loss

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        nProcessed += len(data)

        pred = output.detach().max(1)[1]
        incorrect = pred.ne(target_flat).sum().item()
        err = 100.0 * incorrect / target_flat.numel()

        num_classes = output.size(1)
        dice = multiclass_dice(
            pred.detach().cpu(),
            target_flat.detach().cpu(),
            num_classes,
            ignore_index=0
        )

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


def test_nll(args, epoch, model, testLoader, optimizer, testF, weights):
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
                loss = 0.5 * ce_loss + 0.5 * dice_loss

            test_loss += float(loss.item())

            pred = output.detach().max(1)[1]

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

            incorrect += pred.ne(target_flat).sum().item()

            dice = multiclass_dice(
                pred.detach().cpu(),
                target_flat.detach().cpu(),
                num_classes,
                ignore_index=0
            )
            dice_accum += dice
            dice_batches += 1

    if num_classes is not None:
        print("\nPer-class Dice:")
        for c in range(1, num_classes):
            if class_counts[c] > 0:
                avg_dice = class_dice_sum[c] / class_counts[c]
                print(f"Class {c}: {avg_dice:.4f}")

    test_loss /= len(testLoader)
    err = 100.0 * incorrect / numel
    mean_dice = dice_accum / max(dice_batches, 1)

    print(
        '\nTest set: Average loss: {:.4f}, Error: {}/{} ({:.3f}%), '
        'Mean Dice: {:.4f}\n'.format(
            test_loss, incorrect, numel, float(err), float(mean_dice)
        )
    )

    testF.write('{},{},{},{}\n'.format(epoch, test_loss, float(err), float(mean_dice)))
    testF.flush()
    return err, mean_dice


def adjust_opt(optAlg, optimizer, epoch):
    if optAlg == 'sgd':
        if epoch < 150:
            lr = 1e-1
        elif epoch == 150:
            lr = 1e-2
        elif epoch == 225:
            lr = 1e-3
        else:
            return

        for param_group in optimizer.param_groups:
            param_group['lr'] = lr


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batchSz', type=int, default=10)
    parser.add_argument('--ngpu', type=int, default=1)
    parser.add_argument('--nEpochs', type=int, default=300)
    parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                        help='manual epoch number (useful on restarts)')
    parser.add_argument('--resume', default='', type=str, metavar='PATH',
                        help='path to latest checkpoint (default: none)')
    parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                        help='evaluate model on validation set')
    parser.add_argument('--data', default='data', type=str,
                        help='path to data directory (default: data)')
    # inference path disabled for now (we’re not doing LUNA16-style inference)
    parser.add_argument('--weight-decay', '--wd', default=1e-8, type=float,
                        metavar='W', help='weight decay (default: 1e-8)')
    parser.add_argument('--no-cuda', action='store_true')
    parser.add_argument('--save')
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--opt', type=str, default='adam',
                        choices=('sgd', 'adam', 'rmsprop'))
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--profile-memory', action='store_true',
                    help='run one training batch and report peak GPU memory, then exit')
    args = parser.parse_args()
    print("RUNNING FILE:", __file__)
    print("VALIDATION CALL IS ENABLED")

    best_dice = -1.0
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    args.save = args.save or 'work/vnet.knee.{}'.format(datestr())
    weight_decay = args.weight_decay

    setproctitle.setproctitle(args.save)

    torch.manual_seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed(args.seed)

    scaler = torch.cuda.amp.GradScaler(enabled=args.cuda)
    # DataLoader kwargs
    kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}

    # Build VNet for 2 classes (0 = bg, 1 = cartilage)
    print("build vnet")
    num_classes = 2
    nll = True  # we only support NLL now
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
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            best_dice = checkpoint.get('best_dice', -1.0)
            model.load_state_dict(checkpoint['state_dict'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))
    else:
        model.apply(weights_init)

    print('  + Number of params: {}'.format(
        sum([p.data.nelement() for p in model.parameters()]))
    )

    # Fresh save directory
    if not args.resume:
        if os.path.exists(args.save):
            shutil.rmtree(args.save)
        os.makedirs(args.save, exist_ok=True)
    else:
        os.makedirs(args.save, exist_ok=True)

    # Datasets / loaders
    print("loading training set")
    trainSet = KneeMRIDataset(root=args.data, split='train')
    trainLoader = DataLoader(
        trainSet,
        batch_size=batch_size,
        shuffle=True,
        **kwargs
    )

    print("loading validation set")
    testSet = KneeMRIDataset(root=args.data, split='valid')
    testLoader = DataLoader(
        testSet,
        batch_size=batch_size,
        shuffle=False,
        **kwargs
    )

    # 7-class weights (can tune later)
    class_weights = torch.tensor([0.1,2.0], dtype=torch.float32)
    if args.cuda:
        class_weights = class_weights.cuda()

    # Optimizer
    if args.opt == 'sgd':
        optimizer = optim.SGD(
            model.parameters(),
            lr=1e-1,
            momentum=0.99,
            weight_decay=weight_decay
        )
    elif args.opt == 'adam':
        optimizer = optim.Adam(
            model.parameters(),
            lr=args.lr,
            weight_decay=weight_decay
        )
    elif args.opt == 'rmsprop':
        optimizer = optim.RMSprop(
            model.parameters(),
            weight_decay=weight_decay
        )

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='max',
        factor=0.5,
        patience=3,
        threshold=1e-4,
        min_lr=1e-6
    )

    trainF = open(os.path.join(args.save, 'train.csv'), 'w')
    testF = open(os.path.join(args.save, 'test.csv'), 'w')

    if args.profile_memory:
        print("=== PROFILING LOW-RES TRAINING (1 BATCH) ===")

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
            loss = 0.2 * ce_loss + 0.8 * dice_loss

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        if args.cuda:
            torch.cuda.synchronize()
            print_gpu_memory("LOW-RES TRAIN")

        trainF.close()
        testF.close()
        return

    if args.evaluate:
        print("Running evaluation only...")

        # run validation once using the loaded checkpoint
        err, val_dice = test_nll(
            args,
            args.start_epoch,
            model,
            testLoader,
            None,
            testF,
            class_weights
        )

        print(f"FINAL VALIDATION: val_dice={val_dice:.4f}, val_err={err:.4f}")

        trainF.close()
        testF.close()
        return

    # Always use NLL training/testing
    train_fn = train_nll
    test_fn = test_nll

    for epoch in range(1, args.nEpochs + 1):
        adjust_opt(args.opt, optimizer, epoch)
        train_fn(args, epoch, model, trainLoader, optimizer, trainF, class_weights, scaler)
        err, val_dice = test_fn(args, epoch, model, testLoader, optimizer, testF, class_weights)
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
