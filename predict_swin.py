import os
import cv2
import glob
import math
import pickle
import argparse
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import albumentations
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm.notebook import tqdm as tqdm

import torch
from torch.utils.data import DataLoader, Dataset
import torch.nn as nn
import torch.nn.functional as F
from model.hybrid_swin_transformer import ArcFaceLossAdaptiveMargin, SwinTransformer
from configs.config import init_config
import geffnet
import faiss
from collections import Counter
import h5py
from data_loader.dataset import LandmarkDataset, get_df, get_transforms
from apex.parallel import DistributedDataParallel
from apex import amp
import apex

from torch.nn.parameter import Parameter
from timm.models.vision_transformer_hybrid import HybridEmbed   
import timm
from utils.reranking import *
# If tqdm error => pip install tqdm --upgrade
#MODEL

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_name', type=str, required=True)
    parser.add_argument('--reranking_method', type=str, default="new")
    parser.add_argument('--data_dir', type=str, required=True)
    parser.add_argument('--weight_path', type=str, required=True)
    parser.add_argument('--trainCSVPath', type=str, required=True)
    parser.add_argument('--testCSVPath', type=str, required=True)
    parser.add_argument('--trainH5Path', type=str, required=True)
    parser.add_argument('--indexH5Path', type=str, required=True)

    args, _ = parser.parse_known_args()
    return args

def load_model(model, model_file):
    state_dict = torch.load(model_file)
    if "model_state_dict" in state_dict.keys():
        state_dict = state_dict["model_state_dict"]
    state_dict = {k[7:] if k.startswith('module.') else k: state_dict[k] for k in state_dict.keys()}
#     del state_dict['metric_classify.weight']
    model.load_state_dict(state_dict, strict=True)
    print(f"loaded {model_file}")
    model.eval()    
    return model


def load_scattered_h5data(file_path):
    
    ids, feats = [], []

    with h5py.File(file_path, 'r') as f:
            ids.append(f['ids'][()].astype(str))
            feats.append(f['feats'][()])    

    ids = np.concatenate(ids, axis=0)
    feats = np.concatenate(feats, axis=0)

    order = np.argsort(ids)
    ids = ids[order]
    feats = feats[order]

    return ids, feats

def prepare_ids_and_feats(path, weights=None, normalize=True):
    if weights is None:
        weights = [1.0]
        
    ids, feats = load_scattered_h5data(path)
    feats = l2norm_numpy(feats) * weights

    return ids, feats.astype(np.float32)

def l2norm_numpy(x):
    return x / np.linalg.norm(x, ord=2, axis=1, keepdims=True)

def main(use_reranking_method = "new"):
  # swin model
  model = SwinTransformer(cfg)
  model = model.cuda()
  model = load_model(model, weight_path)

  # create test loader
  df_test = pd.read_csv(testCSVPath)
  df_test['filepath'] = df_test['id'].apply(lambda x: os.path.join(data_dir,'train', '_'.join(x.split("_")[:-1]), f'{x}.jpg'))
  dataset_test = LandmarkDataset(df_test, 'test', 'test',transforms)
  test_loader = torch.utils.data.DataLoader(dataset_test, batch_size=batch_size, num_workers=num_workers, shuffle = True)

  # Check if train h5 file path is exists
  if os.path.exists(trainH5Path):
    ids_train, feats_train = prepare_ids_and_feats(trainH5Path)

  else:
    # Predict feature of train dataset and save to h5 file
    saveTrainFeatueToH5File(trainCSVPath, trainH5Path, model,transforms, data_dir, batch_size, num_workers)
    ids_train, feats_train = prepare_ids_and_feats(trainH5Path)
  if use_reranking_method == "top1_3_2019":

    # Predict feature of test dataset
    feats_test = []
    ids_test = df_test['id']
    for image in tqdm(test_loader):
      feat,_= model(image.cuda())
      feat = feat.detach().cpu()
      feats_test.append(feat)

    feats_test = torch.cat(feats_test)
    feats_test = feats_test.cuda()
    feats_test = F.normalize(feats_test).cpu().detach().numpy().astype(np.float32)

    # Find k nearest neighbor from index dataset with given test image
    k = 8
    print("--- Build index for index dataset to search given test image ---")
    ids_index, feats_index = prepare_ids_and_feats(indexH5Path)
    cpu_index = faiss.IndexFlatL2(feats_index.shape[1])
    gpu_index = faiss.index_cpu_to_all_gpus(cpu_index)
    gpu_index.add(feats_index)
    dists, topk_idx = gpu_index.search(x=feats_test, k=k)
    print("-"*60)
    subm = pd.DataFrame(df_test['id'], columns=['id'])
    images = []
    for idx in topk_idx:
      images.append(' '.join(ids_index[idx]))

    subm['images'] = images
    reranking = Reranking1_3_2019(trainCSVPath)
    subm = reranking(ids_index, feats_index,
                    ids_test, feats_test,
                    ids_train, feats_train,
                    subm, topk=TOP_K)
    print(subm)

  elif use_reranking_method == "top3_2020":
    df_train = pd.read_csv(trainCSVPath)
    landmark_id2idx = {landmark_id:idx for idx, landmark_id in enumerate(sorted(df_train['landmark_id'].unique()))}
    idx2landmark_id = {idx:landmark_id for idx, landmark_id in enumerate(sorted(df_train['landmark_id'].unique()))}
    pred_mask = pd.Series(df_train['landmark_id'].unique()).map(landmark_id2idx).values
    reranking = Reranking_3_2020( batch_size, out_dim, CLS_TOP_K, TOP_K, data_dir, num_workers)
    PRODS, PREDS, PRODS_M,PREDS_M = reranking(feats_train, test_loader, model, pred_mask)
    print('-'*30, "PRODS",'-'*30)
    print(PRODS)
    print('-'*30, "PREDS",'-'*30)
    print(PREDS)
    print('-'*30, "PRODS_M",'-'*30)
    print(PRODS_M)
    print('-'*30, "PREDS_M",'-'*30)
    print(PREDS_M)
  elif use_reranking_method=='top1_shopee':
    reranking = Reranking1_Shoppe()
    # Remember add feats_train and feats_test
    threshes = [0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8] # Adjust thresholds
    result_emb, match_index_lst = reranking.iterative_neighborhood_blending(feats_train, threshes,k_neighbors=51)
    print(match_index_lst)
  else:
    print("No Found Reranking Method")

         
if __name__ == '__main__': 
    args = parse_args()
    
    if args.config_name == None:
        assert "Wrong config_file.....!"
    
    cfg = init_config(args.config_name)

    device = torch.device('cuda')
    batch_size = cfg['inference']['batch_size']
    num_workers = cfg['inference']['num_workers']
    out_dim = cfg['inference']['out_dim'] 
    image_size = cfg['inference']['image_size']
    TOP_K = cfg['inference']['TOP_K']
    CLS_TOP_K = cfg['inference']['CLS_TOP_K']

    data_dir = args.data_dir
    weight_path = args.weight_path
    trainCSVPath = args.trainCSVPath
    testCSVPath = args.testCSVPath

    trainH5Path = args.trainH5Path
    indexH5Path = args.indexH5Path

    transforms = albumentations.Compose([
        albumentations.Resize(image_size, image_size),
        albumentations.Normalize()
    ])
    main(use_reranking_method=args.reranking_method)
