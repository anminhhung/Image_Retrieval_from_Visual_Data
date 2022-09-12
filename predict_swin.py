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
# If tqdm error => pip install tqdm --upgrade
#MODEL

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_name', type=str, required=True)

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
    
    if normalize:
        feats = l2norm_numpy(feats)

    return ids, feats.astype(np.float32)

def l2norm_numpy(x):
    return x / np.linalg.norm(x, ord=2, axis=1, keepdims=True)

def predict_landmark_id(ids_query, feats_query, ids_train, feats_train, landmark_dict, voting_k=3):
    print("\n")
    print('build index...')
    cpu_index = faiss.IndexFlatL2(feats_train.shape[1])
    cpu_index.add(feats_train)
    sims, topk_idx = cpu_index.search(x=feats_query, k=voting_k)
    print('query search done.')

    df = pd.DataFrame(ids_query, columns=['id'])
    images = []
    for idx in topk_idx:
      images.append(' '.join(ids_train[idx]))

    df['images'] = images
    rows = []
    for imidx, (_, r) in tqdm(enumerate(df.iterrows()), total=len(df)):
        image_ids = [name for name in r.images.split(' ')]
        counter = Counter()
        for i, image_id in enumerate(image_ids[:voting_k]):
            landmark_id = landmark_dict[image_id]

            counter[landmark_id] += sims[imidx, i]

        landmark_id, score = counter.most_common(1)[0]
        rows.append({
            'id': r['id'],
            'landmarks': f'{landmark_id} {score:.9f}',
        })

    pred = pd.DataFrame(rows).set_index('id')
    pred['landmark_id'], pred['score'] = list(
        zip(*pred['landmarks'].apply(lambda x: str(x).split(' '))))
    pred['score'] = pred['score'].astype(np.float32) / voting_k

    return pred


def reranking_new(ids_index, feats_index,
                         ids_test, feats_test,
                         ids_train, feats_train,
                         subm, 
                         topk=100, voting_k=3, thresh=0.4):
    train19_csv = pd.read_csv(trainCSVPath)
    landmark_dict = train19_csv.set_index(
        'id').sort_index().to_dict()['landmark_id']

    pred_index = predict_landmark_id(
        ids_index, feats_index, ids_train, feats_train, landmark_dict, voting_k=voting_k)
    pred_test = predict_landmark_id(
        ids_test, feats_test, ids_train, feats_train, landmark_dict, voting_k=voting_k)

    assert np.all(subm['id'] == pred_test.index)
    subm['index_id_list'] = subm['images'].apply(lambda x: x.split(' ')[:topk])

    # to make higher score be inserted ealier position in insert-step
    pred_index = pred_index.sort_values('score', ascending=False)

    images = []
    for test_id, pred, ids in tqdm(zip(subm['id'], pred_test['landmark_id'], subm['index_id_list']),
                                        total=len(subm)):
        retrieved_pred = pred_index.loc[ids, ['landmark_id', 'score']]

        # Sort-step
        is_positive: pd.Series = (pred == retrieved_pred['landmark_id'])
        # use mergesort to keep relative order of original list.
        sorted_retrieved_ids: list = (~is_positive).sort_values(
            kind='mergesort').index.to_list()

        # Insert-step
        whole_positives = pred_index[pred_index['landmark_id'] == pred]
        whole_positives = whole_positives[whole_positives['score'] > thresh]
        # keep the order by referring to the original index
        # in order to insert the samples in descending order of score
        diff = sorted(set(whole_positives.index) - set(ids),
                      key=whole_positives.index.get_loc)
        pos_cnt = is_positive.sum()
        reranked_ids = np.insert(sorted_retrieved_ids, pos_cnt, diff)[:topk]

        images.append(' '.join(reranked_ids))

    subm['images'] = images

    return subm
def reranking_top3(feats_train, test_loader, model, pred_mask,device="cuda"):
    if True:
      if isinstance(feats_train, np.ndarray):
        feats_train = torch.Tensor(feats_train).cuda()
      with torch.no_grad():
        PRODS = []
        PREDS = []
        PRODS_M = []
        PREDS_M = []   
        test_bar = tqdm(test_loader)   
        for img in test_bar:
          img = img.cuda()
          
          probs_m = torch.zeros([batch_size, out_dim],device=device)
          feat_b5,logits_m      = model(img)
          probs_m += logits_m
          
          feat = F.normalize(feat_b5)

          
          probs_m[:, pred_mask] += 1.0
          probs_m -= 1.0              

          (values, indices) = torch.topk(probs_m, CLS_TOP_K, dim=1)
          probs_m = values
          preds_m = indices              
          PRODS_M.append(probs_m.detach().cpu())
          PREDS_M.append(preds_m.detach().cpu())            
          
          distance = feat.mm(feats_train.T)
          (values, indices) = torch.topk(distance, TOP_K, dim=1)
          probs = values
          preds = indices    
          PRODS.append(probs.detach().cpu())
          PREDS.append(preds.detach().cpu())

        PRODS = torch.cat(PRODS).numpy()
        PREDS = torch.cat(PREDS).numpy()
        PRODS_M = torch.cat(PRODS_M).numpy()
        PREDS_M = torch.cat(PREDS_M).numpy()  
        
        return PRODS, PREDS, PRODS_M,PREDS_M

def saveTrainFeatueToH5File(trainCSVPath, trainH5Path, model):
  df_train = pd.read_csv(trainCSVPath)
  df_train['filepath'] = df_train['id'].apply(lambda x: os.path.join(data_dir,'train', '_'.join(x.split("_")[:-1]), f'{x}.jpg'))

  dataset_train = LandmarkDataset(df_train, 'test', 'test',transforms)
  train_loader = torch.utils.data.DataLoader(dataset_train, batch_size=batch_size, num_workers=num_workers, shuffle = True)
  ids_train = df_train['id']
  max_len = len(max(ids_train, key = lambda x: len(x)))

  feats_train = []
  ids_train = df_train['id']
  for image in tqdm(train_loader):
    feat, _ = model(image.cuda())
    feat = feat.detach().cpu()
    feats_train.append(feat)

  feats_train = torch.cat(feats_train)
  feats_train = feats_train.cuda()
  feats_train = F.normalize(feats_train).cpu().detach().numpy().astype(np.float32)
  with h5py.File(trainH5Path, 'w') as f:
    f.create_dataset('ids', data=np.array(ids_train, dtype=f'S{max_len}'))
    f.create_dataset('feats', data=feats_train)

  torch.cuda.empty_cache()
  import gc
  gc.collect()

def main(use_reranking_method = "new"):
  # swin model
  model = SwinTransformer(cfg, mode = "test")
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
    saveTrainFeatueToH5File(trainCSVPath, trainH5Path, model)
    ids_train, feats_train = prepare_ids_and_feats(trainH5Path)
  if use_reranking_method == "new":

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
    ids_index, feats_index = prepare_ids_and_feats(index_h5_dir)
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

    subm = reranking_new(ids_index, feats_index,
                                ids_test, feats_test,
                                ids_train, feats_train,
                                subm, topk=TOP_K)
    print(subm)

  elif use_reranking_method == "top3":
    df_train = pd.read_csv(trainCSVPath)
    landmark_id2idx = {landmark_id:idx for idx, landmark_id in enumerate(sorted(df_train['landmark_id'].unique()))}
    idx2landmark_id = {idx:landmark_id for idx, landmark_id in enumerate(sorted(df_train['landmark_id'].unique()))}
    pred_mask = pd.Series(df_train['landmark_id'].unique()).map(landmark_id2idx).values
    PRODS, PREDS, PRODS_M,PREDS_M = reranking_top3(feats_train, test_loader, model, pred_mask)
    print('-'*30, "PRODS",'-'*30)
    print(PRODS)
    print('-'*30, "PREDS",'-'*30)
    print(PREDS)
    print('-'*30, "PRODS_M",'-'*30)
    print(PRODS_M)
    print('-'*30, "PREDS_M",'-'*30)
    print(PREDS_M)


if __name__ == '__main__': 
    args = parse_args()
    
    if args.config_name == None:
        assert "Wrong config_file.....!"
    
    cfg = init_config(args.config_name)

    device = torch.device('cuda')
    batch_size = cfg['inference']['batch_size']
    num_workers = cfg['inference']['num_workers']
    out_dim = cfg['inference']['out_dim'] 
    normalize = cfg['inference']['normalize']
    image_size = cfg['inference']['image_size']
    TOP_K = cfg['inference']['TOP_K']
    CLS_TOP_K = cfg['inference']['CLS_TOP_K']
    
    
    data_dir = cfg['train']['data_dir']
    weight_path = cfg['inference']['weight_path']
    trainCSVPath = os.path.join(data_dir,cfg['train']['train_list_file_path'] )
    testCSVPath = os.path.join(data_dir,cfg['inference']['test_list_file_path'] )

    trainH5Path = os.path.join(data_dir,'train', "train.h5")
    index_h5_dir = os.path.join(data_dir, cfg['inference']['index_h5_file_path'])

    transforms = albumentations.Compose([
        albumentations.Resize(image_size, image_size),
        albumentations.Normalize()
    ])
    main(use_reranking_method="new")
