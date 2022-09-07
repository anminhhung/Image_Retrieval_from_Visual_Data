from model.models import DenseCrossEntropy, Swish_module, ArcFaceLossAdaptiveMargin,Effnet_Landmark
from config import EfficientnetB5_Config as cfg
from utils.util import global_average_precision_score, GradualWarmupSchedulerV2
from data_loader.dataset import LandmarkDataset, get_df, get_transforms
from apex.parallel import DistributedDataParallel
from apex import amp
import apex
from torch.backends import cudnn
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.optim import lr_scheduler
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader, Dataset
import torch.nn.functional as F
import torch.nn as nn
import torch
from sklearn.metrics import cohen_kappa_score, confusion_matrix
from tqdm import tqdm as tqdm
import pandas as pd
import numpy as np
import albumentations
import argparse
import random
import pickle
import time
import os
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"


def parse_args():

    parser = argparse.ArgumentParser()
    parser.add_argument('--kernel-type', type=str, required=True)
    parser.add_argument('--data-dir', type=str, default='/raid/GLD2')
    parser.add_argument('--train-step', type=int, required=True)
    parser.add_argument('--image-size', type=int, required=True)
    parser.add_argument("--local_rank", type=int)
    parser.add_argument('--enet-type', type=str, required=True)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--num-workers', type=int, default=32)
    parser.add_argument('--init-lr', type=float, default=1e-4)
    parser.add_argument('--n-epochs', type=int, default=15)
    parser.add_argument('--start-from-epoch', type=int, default=1)
    parser.add_argument('--stop-at-epoch', type=int, default=999)
    parser.add_argument('--use-amp', action='store_false')
    parser.add_argument('--DEBUG', action='store_true')
    parser.add_argument('--model-dir', type=str, default='./weights')
    parser.add_argument('--log-dir', type=str, default='./logs')
    parser.add_argument('--CUDA_VISIBLE_DEVICES', type=str,
                        default='0,1,2,3,4,5,6,7')
    parser.add_argument('--fold', type=int, default=0)
    parser.add_argument('--load-from', type=str, default='')
    args, _ = parser.parse_known_args()
    return args


def set_seed(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


def train_epoch(model, loader, optimizer, criterion):

    model.train()
    train_loss = []
    bar = tqdm(loader)
    for (data, target) in bar:

        data, target = data.cuda(), target.cuda()
        optimizer.zero_grad()

        if not args.use_amp:
            logits_m = model(data)
            loss = criterion(logits_m, target)
            loss.backward()
            optimizer.step()
        else:
            logits_m = model(data)
            loss = criterion(logits_m, target)
            with amp.scale_loss(loss, optimizer) as scaled_loss:
                scaled_loss.backward()
            optimizer.step()

        torch.cuda.synchronize()

        loss_np = loss.detach().cpu().numpy()
        train_loss.append(loss_np)
        smooth_loss = sum(train_loss[-100:]) / min(len(train_loss), 100)
        bar.set_description('loss: %.5f, smth: %.5f' % (loss_np, smooth_loss))

    return train_loss


def val_epoch(model, valid_loader, criterion, get_output=False):

    model.eval()
    val_loss = []
    PRODS_M = []
    PREDS_M = []
    TARGETS = []

    with torch.no_grad():
        for (data, target) in tqdm(valid_loader):
            data, target = data.cuda(), target.cuda()

            logits_m = model(data)

            lmax_m = logits_m.max(1)
            probs_m = lmax_m.values
            preds_m = lmax_m.indices

            PRODS_M.append(probs_m.detach().cpu())
            PREDS_M.append(preds_m.detach().cpu())
            TARGETS.append(target.detach().cpu())

            loss = criterion(logits_m, target)
            val_loss.append(loss.detach().cpu().numpy())

        val_loss = np.mean(val_loss)
        PRODS_M = torch.cat(PRODS_M).numpy()
        PREDS_M = torch.cat(PREDS_M).numpy()
        TARGETS = torch.cat(TARGETS)

    if get_output:
        return LOGITS_M
    else:
        acc_m = (PREDS_M == TARGETS.numpy()).mean() * 100.
        y_true = {idx: target if target >=
                  0 else None for idx, target in enumerate(TARGETS)}
        y_pred_m = {idx: (pred_cls, conf) for idx, (pred_cls,
                                                    conf) in enumerate(zip(PREDS_M, PRODS_M))}
        gap_m = global_average_precision_score(y_true, y_pred_m)
        return val_loss, acc_m, gap_m


def main():

    # get dataframe
    df, out_dim = get_df(args.kernel_type, args.data_dir, args.train_step)

    # get adaptive margin
    tmp = np.sqrt(
        1 / np.sqrt(df['landmark_id'].value_counts().sort_index().values))
    margins = (tmp - tmp.min()) / (tmp.max() - tmp.min()) * 0.45 + 0.05

    # get augmentations (Resize and Normalize)
    transforms_train, transforms_val = get_transforms(args.image_size)

    dataset_train = LandmarkDataset(
        df, 'train', 'train', transform=transforms_train)

    model = ModelClass(args.enet_type, out_dim=out_dim)
    model = model.cuda()
    model = apex.parallel.convert_syncbn_model(model)

    # loss func
    def criterion(logits_m, target):
        arc = ArcFaceLossAdaptiveMargin(margins=margins, s=80)
        loss_m = arc(logits_m, target, out_dim=out_dim)
        return loss_m

    # optimizer
    optimizer = optim.Adam(model.parameters(), lr=args.init_lr)
    if args.use_amp:
        model, optimizer = amp.initialize(model, optimizer, opt_level="O1")

    # load pretrained
    if len(args.load_from) > 0:
        checkpoint = torch.load(
            args.load_from,  map_location='cuda:{}'.format(args.local_rank))
        state_dict = checkpoint['model_state_dict']
        state_dict = {k[7:] if k.startswith(
            'module.') else k: state_dict[k] for k in state_dict.keys()}
        if args.train_step == 1:
            del state_dict['metric_classify.weight']
            model.load_state_dict(state_dict, strict=False)
        else:
            model.load_state_dict(state_dict, strict=True)
#             if 'optimizer_state_dict' in checkpoint:
#                 optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        del checkpoint, state_dict
        torch.cuda.empty_cache()
        import gc
        gc.collect()

    model = DistributedDataParallel(model, delay_allreduce=True)

    # lr scheduler
    scheduler_cosine = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, args.n_epochs-1)
    scheduler_warmup = GradualWarmupSchedulerV2(
        optimizer, multiplier=10, total_epoch=1, after_scheduler=scheduler_cosine)

    # train & valid loop
    gap_m_max = 0.
    model_file = os.path.join(
        args.model_dir, f'{args.kernel_type}_fold{args.fold}.pth')
    for epoch in range(args.start_from_epoch, args.n_epochs+1):

        print(time.ctime(), 'Epoch:', epoch)
        scheduler_warmup.step(epoch - 1)

        train_sampler = torch.utils.data.distributed.DistributedSampler(
            dataset_train)
        train_sampler.set_epoch(epoch)

        train_loader = torch.utils.data.DataLoader(dataset_train, batch_size=args.batch_size, num_workers=args.num_workers,
                                                   shuffle=train_sampler is None, sampler=train_sampler, drop_last=True)

        train_loss = train_epoch(model, train_loader, optimizer, criterion)
        # val_loss, acc_m, gap_m = val_epoch(model, valid_loader, criterion)

        if args.local_rank == 0:
            content = time.ctime() + ' ' + \
                f'Fold {args.fold}, Epoch {epoch}, lr: {optimizer.param_groups[0]["lr"]:.7f}, train loss: {np.mean(train_loss):.5f}.'
            print(content)
           

            print('Saving model ...')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }, model_file)
           

        if epoch == args.stop_at_epoch:
            print(time.ctime(), 'Training Finished!')
            break

    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }, os.path.join(args.model_dir, f'{args.kernel_type}_fold{args.fold}_final.pth'))


if __name__ == '__main__':

    args = parse_args()
    os.makedirs(args.model_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)
    os.environ['CUDA_VISIBLE_DEVICES'] = args.CUDA_VISIBLE_DEVICES
    ModelClass = Effnet_Landmark

    set_seed(0)

    if args.CUDA_VISIBLE_DEVICES != '-1':
        torch.backends.cudnn.benchmark = True
        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group(
            backend='nccl', init_method='env://')
        cudnn.benchmark = True

    main()
