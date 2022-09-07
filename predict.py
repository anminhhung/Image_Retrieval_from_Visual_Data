
import os
import cv2
import glob
import math
import pickle
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
from model.models import Effnet_Landmark
import geffnet
import argparse

# If tqdm error => pip install tqdm --upgrade

class LandmarkDataset(Dataset):
    def __init__(self, csv, split, mode, transforms=None):

        self.csv = csv.reset_index()
        self.split = split
        self.mode = mode
        self.transform = transforms

    def __len__(self):
        return self.csv.shape[0]

    def __getitem__(self, index):
        row = self.csv.iloc[index]
        image = cv2.imread(row.filepath)
        image = image[:, :, ::-1]
      
        res = self.transform(image=image)
        image = res['image'].astype(np.float32)
        image = image.transpose(2, 0, 1)        
               
        
        if self.mode == 'test':
            return torch.tensor(image)
#MODEL
class ArcMarginProduct_subcenter(nn.Module):
    def __init__(self, in_features, out_features, k=3):
        super().__init__()
        self.weight = nn.Parameter(torch.FloatTensor(out_features*k, in_features))
        self.reset_parameters()
        self.k = k
        self.out_features = out_features
        
    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        
    def forward(self, features):
        cosine_all = F.linear(F.normalize(features), F.normalize(self.weight))
        cosine_all = cosine_all.view(-1, self.out_features, self.k)
        cosine, _ = torch.max(cosine_all, dim=2)
        return cosine 
        
sigmoid = torch.nn.Sigmoid()
class Swish(torch.autograd.Function):
    @staticmethod
    def forward(ctx, i):
        result = i * sigmoid(i)
        ctx.save_for_backward(i)
        return result
    @staticmethod
    def backward(ctx, grad_output):
        i = ctx.saved_variables[0]
        sigmoid_i = sigmoid(i)
        return grad_output * (sigmoid_i * (1 + i * (1 - sigmoid_i)))

class Swish_module(nn.Module):
    def forward(self, x):
        return Swish.apply(x)

    
class enet_arcface_FINAL(nn.Module):

    def __init__(self, enet_type, out_dim):
        super(enet_arcface_FINAL, self).__init__()
        self.enet = geffnet.create_model(enet_type.replace('-', '_'), pretrained=None)
        self.feat = nn.Linear(self.enet.classifier.in_features, 512)
        self.swish = Swish_module()
        self.metric_classify = ArcMarginProduct_subcenter(512, out_dim)
        self.enet.classifier = nn.Identity()
 
    def forward(self, x):
        x = self.enet(x)
        x = self.swish(self.feat(x))
        return F.normalize(x), self.metric_classify(x)

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
        for img in tqdm(query_loader): # 672, 768, 512
          img = img.cuda()
          feat_b5,_ = model_b5(img)
          feat = torch.cat([feat_b5], dim=1)    
          feats.append(feat.detach().cpu())
        feats = torch.cat(feats)
        feats = feats.cuda()
        feat = F.normalize(feat)        

        PRODS = []
        PREDS = []
        PRODS_M = []
        PREDS_M = []        
        for img in tqdm(test_loader):
          img = img.cuda()
          
          probs_m = torch.zeros([16, 17],device=device)
          feat_b5,logits_m      = model_b5(img); probs_m += logits_m
          feat = torch.cat([feat_b5],dim=1)
          feat = F.normalize(feat)

          #probs_m = probs_m/9
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
 

def parse_args():

    parser = argparse.ArgumentParser()
    parser.add_argument('--backbone', type=str, default="tf_efficientnet_b5_ns")
    args, _ = parser.parse_known_args()
    return args

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
    model_b5 = enet_arcface_FINAL(args.backbone, out_dim=out_dim).to(device)
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
    device = torch.device('cuda')
    batch_size = 16
    num_workers = 2
    out_dim = 17 
    image_size=256
    TOP_K = 5
    CLS_TOP_K = 5

    
    data_dir = '/content/drive/MyDrive/AIC_HCM/DOLG/DOLG_Efficientnet_3rd_2020/data/train/'
    model_dir = './weights/'
    weight_path= '/content/drive/MyDrive/AIC_HCM/DOLG/DOLG_Efficientnet_3rd_2020/weights/b5ns_DDP_final_256_300w_f2_10ep_fold2.pth'
    transforms = albumentations.Compose([
        albumentations.Resize(image_size, image_size),
        albumentations.Normalize()
    ])

    
    main()

