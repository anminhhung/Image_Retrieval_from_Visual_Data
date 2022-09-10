
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
from data_loader.dataset import LandmarkDataset
from configs.config import init_config
import geffnet

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
  
def get(query_loader, test_loader, model_b5, pred_mask,device="cuda"):
    if True:
      with torch.no_grad():
        feats = []
        query_bar = tqdm(query_loader)
        for img in query_bar: # 672, 768, 512
          img = img.cuda()
          feat_b5,_ = model_b5(img)
          feats.append(feat_b5.detach().cpu())
        feats = torch.cat(feats)
        feats = feats.cuda()
        feat = F.normalize(feat_b5)        

        PRODS = []
        PREDS = []
        PRODS_M = []
        PREDS_M = []   
        test_bar = tqdm(test_loader)   
        print(batch_size)  
        for img in test_bar:
          img = img.cuda()
          
          probs_m = torch.zeros([batch_size, out_dim],device=device)
          feat_b5,logits_m      = model_b5(img)
          probs_m += logits_m
          
          feat = F.normalize(feat_b5)

          
          probs_m[:, pred_mask] += 1.0
          probs_m -= 1.0              

          (values, indices) = torch.topk(probs_m, CLS_TOP_K, dim=1)
          probs_m = values
          preds_m = indices              
          PRODS_M.append(probs_m.detach().cpu())
          PREDS_M.append(preds_m.detach().cpu())            
          
          distance = feat.mm(feats.t())
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
 


def main():
    df = pd.read_csv(os.path.join(data_dir, 'train_list.txt'))
    df['filepath'] = df['id'].apply(lambda x: os.path.join(data_dir, '_'.join(x.split("_")[:-1]), f'{x}.jpg'))
    df_sub = pd.read_csv(os.path.join(data_dir, 'test_list.txt'))

    df_test = df_sub[['id']].copy()
    df_test['filepath'] = df_test['id'].apply(lambda x: os.path.join(data_dir, '_'.join(x.split("_")[:-1]), f'{x}.jpg'))

    #use_metric = True


    if df.shape[0] > 100001: # commit
        df = df[df.index % 10 == 0].iloc[500:1000].reset_index(drop=True)
        df_test = df_test.head(101).copy()

    dataset_query = LandmarkDataset(df, 'test', 'test',transforms)
    query_loader = torch.utils.data.DataLoader(dataset_query, batch_size=batch_size, num_workers=num_workers)

    dataset_test = LandmarkDataset(df_test, 'test', 'test',transforms)
    test_loader = torch.utils.data.DataLoader(dataset_test, batch_size=batch_size, num_workers=num_workers)

    # model_b5 = Effnet_Landmark("tf_efficientnet_b5_ns", out_dim=17)
    # model_b5 = enet_arcface_FINAL(args.backbone, out_dim=out_dim).to(device)
    model_b5 = SwinTransformer(cfg, mode = "test")
    model_b5 = model_b5.cuda()
    model_b5 = load_model(model_b5, weight_path)

    landmark_id2idx = {landmark_id:idx for idx, landmark_id in enumerate(sorted(df['landmark_id'].unique()))}
    idx2landmark_id = {idx:landmark_id for idx, landmark_id in enumerate(sorted(df['landmark_id'].unique()))}
    pred_mask = pd.Series(df['landmark_id'].unique()).map(landmark_id2idx).values

    PRODS, PREDS, PRODS_M,PREDS_M = get(query_loader, test_loader,model_b5, pred_mask)

    # map both to landmark_id
    gallery_landmark = df['landmark_id'].values
    PREDS = gallery_landmark[PREDS]
    PREDS_M = np.vectorize(idx2landmark_id.get)(PREDS_M)

    PRODS_F = []
    PREDS_F = []
    for i in tqdm(range(PREDS.shape[0])):
        tmp = {}
        classify_dict = {PREDS_M[i,j] : PRODS_M[i,j] for j in range(CLS_TOP_K)}
        for k in range(TOP_K):
            lid = PREDS[i, k]
            tmp[lid] = tmp.get(lid, 0.) + float(PRODS[i, k]) ** 9 * classify_dict.get(lid,1e-8)**10
        pred, conf = max(tmp.items(), key=lambda x: x[1])
        PREDS_F.append(pred)
        PRODS_F.append(conf)
        
     
    df_test['pred_id'] = PREDS_F
    df_test['pred_conf'] = PRODS_F

    df_sub['landmarks'] = df_test.apply(lambda row: f'{row["pred_id"]} {row["pred_conf"]}', axis=1)

    print(df_sub.head())
    
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
    
    data_dir = cfg['train']['data_dir']+"/train"
    weight_path = cfg['inference']['weight_path']
    transforms = albumentations.Compose([
        albumentations.Resize(image_size, image_size),
        albumentations.Normalize()
    ])
    main()
