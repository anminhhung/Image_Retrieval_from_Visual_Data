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
from model.hybrid_swin_transformer import ArcFaceLossAdaptiveMargin, SwinTransformer
from utils.util import global_average_precision_score, GradualWarmupSchedulerV2
from data_loader.dataset import LandmarkDataset, get_df, get_transforms
from pathlib import Path
from data_loader.make_dataloader import make_dataloader
import wandb

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

    return np.mean(train_loss)

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

def train(cfg, args):
    # get dataframe
    df_train, out_dim = get_df(args.trainCSVPath)
    df_val, _ = get_df(args.valCSVPath)
    
    train_loader = make_dataloader(cfg['train'], args.trainCSVPath)
    valid_loader = make_dataloader(cfg['val'], args.valCSVPath)

    # get adaptive margin
    tmp = np.sqrt(
        1 / np.sqrt(df_train['landmark_id'].value_counts().sort_index().values))
    margins = (tmp - tmp.min()) / (tmp.max() - tmp.min()) * 0.45 + 0.05

    # swin model
    model = SwinTransformer(cfg).cuda()
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
        
    best_fitness = 0.0
    # load pretrained
    if args.checkpoint:
        print('-------Load Checkpoint-------')
        cfg['train']['model_dir'] = '/'.join(args.checkpoint.split('/')[:-1])
        args.exist_ok = True
        checkpoint = torch.load(args.checkpoint,  map_location='cuda:{}'.format(cfg['train']['local_rank']))
        cfg['train']['start_from_epoch'] = checkpoint['epoch'] + 1
        best_fitness = checkpoint['best_fitness']
        state_dict = checkpoint['model_state_dict']
        state_dict = {k[7:] if k.startswith(
            'module.') else k: state_dict[k] for k in state_dict.keys()}
        model.load_state_dict(state_dict, strict=True)
        del checkpoint, state_dict
        torch.cuda.empty_cache()
        import gc
        gc.collect()
        print('-------DONE-------')

    model = DistributedDataParallel(model, delay_allreduce=True)

    # lr scheduler
    scheduler_cosine = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, cfg['train']['n_epochs']-1)
    scheduler_warmup = GradualWarmupSchedulerV2(
        optimizer, multiplier=10, total_epoch=1, after_scheduler=scheduler_cosine)

    # Directories
    model_path = increment_path(Path(cfg['train']['model_dir']), exist_ok=args.exist_ok)
    os.makedirs(model_path, exist_ok=True)
    last = os.path.join(model_path, 'last.pth')
    best = os.path.join(model_path, 'best.pth')

    # train & valid loop
    gap_m_max = 0
    for epoch in range(cfg['train']['start_from_epoch'], cfg['train']['n_epochs']+1):
        print(time.ctime(), 'Epoch:', epoch)
        scheduler_warmup.step(epoch - 1)

        train_loss = train_epoch(model, train_loader, optimizer, criterion)
        val_loss, acc_m, gap_m = val_epoch(model, valid_loader, criterion)
        content = time.ctime() + ' ' + \
            f'Epoch {epoch}, lr: {optimizer.param_groups[0]["lr"]:.7f}, train loss: {train_loss:.5f}, valid loss: {(val_loss):.5f}, acc_m: {(acc_m):.6f}, gap_m: {(gap_m):.6f}.'
        print(content)
        if "wandb" in cfg:
            wandb.log({'loss_train':train_loss,
                        'loss_val':val_loss,
                        'accuracy': acc_m,
                        'gap': gap_m})
            wandb.watch(model)
            
        if gap_m > best_fitness:
            best_fitness = gap_m

        ckpt = {'epoch': epoch,
                'best_fitness': best_fitness,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict()}
        torch.save(ckpt, last)
        if best_fitness == gap_m:
            print(f"Save best epoch: {epoch}")
            torch.save(ckpt, best)
        if epoch % cfg['train']['save_per_epoch'] == 0:
            save_dir = os.path.join(model_path, 
                            "{}_{}.pth".format(cfg['train']['model_name'], epoch))
            print('gap_m_max ({:.6f} --> {:.6f}). Saving model to {}'.format(gap_m_max, gap_m, save_dir))
            torch.save(ckpt, save_dir)
            gap_m_max = gap_m

if __name__ == '__main__':
    args = parse_args()
    if args.config_name == None:
        assert "Wrong config_file.....!"
    
    cfg = init_config(args.config_name)
    os.environ['CUDA_VISIBLE_DEVICES'] = cfg['train']['CUDA_VISIBLE_DEVICES']
    set_seed(0)

    if cfg['train']['CUDA_VISIBLE_DEVICES'] != '-1':
        torch.backends.cudnn.benchmark = True
        torch.cuda.set_device(cfg['train']['local_rank'])
        torch.distributed.init_process_group(
            backend='nccl', init_method='env://')
        cudnn.benchmark = True
    if "wandb" in cfg:
        wandb.init(project=cfg['wandb']['project'])
    
    train(cfg, args)