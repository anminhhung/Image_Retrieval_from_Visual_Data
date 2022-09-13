import apex
import torch.optim as optim
import torch.nn.functional as F
import torch
import numpy as np
import argparse
import random
import time
import os
import glob
import re

from torch.backends import cudnn
from tqdm import tqdm as tqdm
from apex.parallel import DistributedDataParallel
from apex import amp

from configs.config import init_config
from model.DOLG import ArcFaceLossAdaptiveMargin,DOLG
from utils.util import global_average_precision_score, GradualWarmupSchedulerV2
from data_loader.dataset import LandmarkDataset, get_df, get_transforms
from pathlib import Path

os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"

def increment_path(path, exist_ok=True, sep=''):
    path = Path(path)  # os-agnostic
    if (path.exists() and exist_ok) or (not path.exists()):
        return str(path)
    else:
        dirs = glob.glob(f"{path}{sep}*")  # similar paths
        matches = [re.search(rf"%s{sep}(\d+)" % path.stem, d) for d in dirs]
        i = [int(m.groups()[0]) for m in matches if m]  # indices
        n = max(i) + 1 if i else 2  # increment number
        return f"{path}{sep}{n}"  # update path

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_name', type=str, required=True)
    parser.add_argument('--trainCSVPath', type=str, required=True)
    parser.add_argument('--valCSVPath', type=str, required=True)
    parser.add_argument('--checkpoint', nargs='?', const=True, default=False, help='resume most recent training')
    parser.add_argument('--exist_ok', action='store_true', help='existing project/name ok, do not increment')

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
        if not cfg['train']['use_amp']:
            _, logits_m = model(data)
            loss = criterion(logits_m, target)
            loss.backward()
            optimizer.step()
        else:
            _, logits_m = model(data)
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
    LOGITS_M = []

    with torch.no_grad():
        for (data, target) in tqdm(valid_loader):
            data, target = data.cuda(), target.cuda()
            _, logits_m = model(data)

            lmax_m = logits_m.max(1)
            probs_m = lmax_m.values
            preds_m = lmax_m.indices

            PRODS_M.append(probs_m.detach().cpu())
            PREDS_M.append(preds_m.detach().cpu())
            TARGETS.append(target.detach().cpu())
            LOGITS_M.append(logits_m)

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

def train(cfg):
    # get augmentations (Resize and Normalize)
    transforms_train, transforms_val = get_transforms(cfg['train']['image_size'])
    # get dataframe
    df_train, out_dim = get_df(trainCSVPath)
    df_val, _ = get_df(valCSVPath)
    print(f"out_dim = {out_dim}")
    
    dataset_train = LandmarkDataset(df_train, 'train', 'train', transform=transforms_train)
    dataset_valid = LandmarkDataset(df_val, 'train', 'train', transform=transforms_val)
    valid_loader = torch.utils.data.DataLoader(dataset_valid, batch_size=cfg['val']['batch_size'], num_workers=cfg['val']['num_workers'])

    # get adaptive margin
    tmp = np.sqrt(
        1 / np.sqrt(df_train['landmark_id'].value_counts().sort_index().values))
    margins = (tmp - tmp.min()) / (tmp.max() - tmp.min()) * 0.45 + 0.05

    # DOLG model
    model = DOLG(cfg)
    model = model.cuda()
    model = apex.parallel.convert_syncbn_model(model)

    # loss func
    def criterion(logits_m, target):
        arc = ArcFaceLossAdaptiveMargin(margins=margins, s=cfg['train']['arcface_s'])
        loss_m = arc(logits_m, target, out_dim=out_dim)
        return loss_m

    # optimizer
    optimizer = optim.Adam(model.parameters(), lr=cfg['train']['init_lr'])
    if cfg['train']['use_amp']:
        model, optimizer = amp.initialize(model, optimizer, opt_level="O1")
    
    # load pretrained
    if args.checkpoint:
        print('-------Load Checkpoint-------')
        checkpoint = torch.load(args.checkpoint,  map_location='cuda:{}'.format(cfg['train']['local_rank']))
        state_dict = checkpoint['model_state_dict']
        state_dict = {k[7:] if k.startswith(
            'module.') else k: state_dict[k] for k in state_dict.keys()}
        model.load_state_dict(state_dict, strict=True)
        del checkpoint, state_dict
        torch.cuda.empty_cache()
        import gc
        gc.collect()

    model = DistributedDataParallel(model, delay_allreduce=True)

    # lr scheduler
    scheduler_cosine = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, cfg['train']['n_epochs']-1)
    scheduler_warmup = GradualWarmupSchedulerV2(
        optimizer, multiplier=10, total_epoch=1, after_scheduler=scheduler_cosine)

    # train & valid loop
    gap_m_max = 0
    model_path = increment_path(Path(cfg['train']['model_dir']), exist_ok=args.exist_ok)
    os.makedirs(model_path, exist_ok=True)
    for epoch in range(cfg['train']['start_from_epoch'], cfg['train']['n_epochs']+1):
        print(time.ctime(), 'Epoch:', epoch)
        scheduler_warmup.step(epoch - 1)

        train_sampler = torch.utils.data.distributed.DistributedSampler(
            dataset_train)
        train_sampler.set_epoch(epoch)      
        train_loader = torch.utils.data.DataLoader(dataset_train, batch_size=cfg['train']['batch_size'], num_workers=cfg['train']['num_workers'],
                                                   shuffle=train_sampler is None, sampler=train_sampler, drop_last=True)
        
        train_loss = train_epoch(model, train_loader, optimizer, criterion)
        val_loss, acc_m, gap_m = val_epoch(model, valid_loader, criterion)
          
        if cfg['train']['save_per_epoch']:
            content = time.ctime() + ' ' + \
                f'Epoch {epoch}, lr: {optimizer.param_groups[0]["lr"]:.7f}, \
                train loss: {np.mean(train_loss):.5f}, valid loss: {(val_loss):.5f}, acc_m: {(acc_m):.6f}, gap_m: {(gap_m):.6f}.'
            print(content)

            save_dir = os.path.join(model_path, 
                            "dolg_{}_{}.pth".format(cfg['train']['model_name'], epoch))
            print('gap_m_max ({:.6f} --> {:.6f}). Saving model to {}'.format(gap_m_max, gap_m, save_dir))
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }, save_dir)
            gap_m_max = gap_m
      
    if cfg['train']['save_per_epoch'] == False:
        save_dir = increment_path(model_path, exist_ok=args.exist_ok)  # increment run
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
        }, os.path.join(save_dir, 'dolg_{}.pth'.format(cfg['train']['model_name'])))

if __name__ == '__main__':
    args = parse_args()
    if args.config_name == None:
        assert "Wrong config_file.....!"

    cfg = init_config(args.config_name)
    os.environ['CUDA_VISIBLE_DEVICES'] = cfg['train']['CUDA_VISIBLE_DEVICES']
    trainCSVPath = args.trainCSVPath
    valCSVPath = args.valCSVPath
  
    set_seed(0)
    if cfg['train']['CUDA_VISIBLE_DEVICES'] != '-1':
        torch.backends.cudnn.benchmark = True
        torch.cuda.set_device(cfg['train']['local_rank'])
        torch.distributed.init_process_group(
            backend='nccl', init_method='env://')
        cudnn.benchmark = True

    train(cfg)