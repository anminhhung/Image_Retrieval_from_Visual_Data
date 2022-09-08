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
import timm
from torch.nn.parameter import Parameter

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

class DenseCrossEntropy(nn.Module):
    def forward(self, x, target):
        x = x.float()
        target = target.float()
        logprobs = torch.nn.functional.log_softmax(x, dim=-1)

        loss = -logprobs * target
        loss = loss.sum(-1)
        return loss.mean()

def gem(x, p=3, eps=1e-6):
    return F.avg_pool2d(x.clamp(min=eps).pow(p), (x.size(-2), x.size(-1))).pow(1./p)

class GeM(nn.Module):
    def __init__(self, p=3, eps=1e-6, p_trainable=False):
        super(GeM,self).__init__()
        if p_trainable:
            self.p = Parameter(torch.ones(1)*p)
        else:
            self.p = p
        self.eps = eps

    def forward(self, x):
        ret = gem(x, p=self.p, eps=self.eps)   
        return ret
    def __repr__(self):
        return self.__class__.__name__ + '(' + 'p=' + '{:.4f}'.format(self.p.data.tolist()[0]) + ', ' + 'eps=' + str(self.eps) + ')'

class MultiAtrousModule(nn.Module):
    def __init__(self, in_chans, out_chans, dilations):
        super(MultiAtrousModule, self).__init__()
        
        self.d0 = nn.Conv2d(in_chans, 512, kernel_size=3, dilation=dilations[0],padding='same')
        self.d1 = nn.Conv2d(in_chans, 512, kernel_size=3, dilation=dilations[1],padding='same')
        self.d2 = nn.Conv2d(in_chans, 512, kernel_size=3, dilation=dilations[2],padding='same')
        self.conv1 = nn.Conv2d(512 * 3, out_chans, kernel_size=1)
        self.relu = nn.ReLU()
        
    def forward(self,x):
        
        x0 = self.d0(x)
        x1 = self.d1(x)
        x2 = self.d2(x)
        x = torch.cat([x0,x1,x2],dim=1)
        x = self.conv1(x)
        x = self.relu(x)
        return x

class SpatialAttention2d(nn.Module):
    def __init__(self, in_c):
        super(SpatialAttention2d, self).__init__()
        self.conv1 = nn.Conv2d(in_c, 1024, 1, 1)
        self.bn = nn.BatchNorm2d(1024)
        self.act1 = nn.ReLU()
        self.conv2 = nn.Conv2d(1024, 1, 1, 1)
        self.softplus = nn.Softplus(beta=1, threshold=20) # use default setting.

    def forward(self, x):
        '''
        x : spatial feature map. (b x c x w x h)
        att : softplus attention score 
        '''
        x = self.conv1(x)
        x = self.bn(x)
        
        feature_map_norm = F.normalize(x, p=2, dim=1)
         
        x = self.act1(x)
        x = self.conv2(x)
        att_score = self.softplus(x)
        att = att_score.expand_as(feature_map_norm)
        
        x = att * feature_map_norm
        return x, att_score   

class OrthogonalFusion(nn.Module):
    def __init__(self):
        super(OrthogonalFusion, self).__init__()

    def forward(self, fl, fg):

        bs, c, w, h = fl.shape
        
        fl_dot_fg = torch.bmm(fg[:,None,:],fl.reshape(bs,c,-1))
        fl_dot_fg = fl_dot_fg.reshape(bs,1,w,h)
        fg_norm = torch.norm(fg, dim=1)
        
        fl_proj = (fl_dot_fg / fg_norm[:,None,None,None]) * fg[:,:,None,None]
        fl_orth = fl - fl_proj
        
        f_fused = torch.cat([fl_orth,fg[:,:,None,None].repeat(1,1,w,h)],dim=1)
        return f_fused  

class DOLG(nn.Module):
    def __init__(self):
        super(DOLG, self).__init__()

        self.n_classes = 17
        self.backbone = timm.create_model('tf_efficientnet_b5_ns', 
                                          pretrained=True, 
                                          num_classes=0, 
                                          global_pool="", 
                                          features_only = True)

        
        # if ("efficientnet" in cfg.backbone) & (self.cfg.stride is not None):
        #     self.backbone.conv_stem.stride = self.cfg.stride
        backbone_out = self.backbone.feature_info[-1]['num_chs']
        backbone_out_1 = self.backbone.feature_info[-2]['num_chs']
        
        feature_dim_l_g = 1024
        fusion_out = 2 * feature_dim_l_g

        self.global_pool = GeM(p_trainable=True)

        self.fusion_pool = nn.AdaptiveAvgPool2d(1)
        self.embedding_size = 512

        self.neck = nn.Sequential(
                nn.Linear(fusion_out, 512, bias=True),
                nn.BatchNorm1d(512),
                torch.nn.PReLU()
            )

        self.head_in_units = 512
        self.head = ArcMarginProduct_subcenter(512, 17)
    
        self.mam = MultiAtrousModule(backbone_out_1, feature_dim_l_g, [6,12,18])
        self.conv_g = nn.Conv2d(backbone_out,feature_dim_l_g,kernel_size=1)
        self.bn_g = nn.BatchNorm2d(feature_dim_l_g, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
        self.act_g =  nn.SiLU(inplace=True)
        self.attention2d = SpatialAttention2d(feature_dim_l_g)
        self.fusion = OrthogonalFusion()

        self.swish = Swish_module()
    
    def forward(self, x):      
        x = self.backbone(x)
        
        x_l = x[-2]
        x_g = x[-1]
        
        x_l = self.mam(x_l)
        x_l, att_score = self.attention2d(x_l)
        
        x_g = self.conv_g(x_g)
        x_g = self.bn_g(x_g)
        x_g = self.act_g(x_g)
        
        x_g = self.global_pool(x_g)
        x_g = x_g[:,:,0,0]
        
        x_fused = self.fusion(x_l, x_g)
        x_fused = self.fusion_pool(x_fused)
        x_fused = x_fused[:,:,0,0]        
        
        x_emb = self.neck(x_fused)
       
        logits_m  = self.head(x_emb)

        return F.normalize(x_emb), logits_m

    def freeze_weights(self, freeze=[]):
        for name, child in self.named_children():
            if name in freeze:
                for param in child.parameters():
                    param.requires_grad = False


    def unfreeze_weights(self, freeze=[]):
        for name, child in self.named_children():
            if name in freeze:
                for param in child.parameters():
                    param.requires_grad = True

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
  
def get(query_loader, test_loader, model, pred_mask,device="cuda"):
    if True:
      with torch.no_grad():
        feats = []
        for img in tqdm(query_loader): # 672, 768, 512
          img = img.cuda()
          feat_b5,_ = model(img)
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
          feat_b5,logits_m      = model(img); probs_m += logits_m
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

    model_dolg = DOLG().to(device)
    model_dolg = load_model(model_dolg, weight_path)

    landmark_id2idx = {landmark_id:idx for idx, landmark_id in enumerate(sorted(df['landmark_id'].unique()))}
    idx2landmark_id = {idx:landmark_id for idx, landmark_id in enumerate(sorted(df['landmark_id'].unique()))}
    pred_mask = pd.Series(df['landmark_id'].unique()).map(landmark_id2idx).values

    PRODS, PREDS, PRODS_M,PREDS_M = get(query_loader, test_loader,model_dolg, pred_mask)

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
    
    data_dir = '/content/drive/MyDrive/AIC_2022/Google_Landmark_Retrieval/top1/data/train/'
    weight_path= '/content/drive/MyDrive/AIC_2022/Google_Landmark_Retrieval/top1/weights/b5ns_DDP_final_256_300w_f2_10ep_fold2.pth'
    transforms = albumentations.Compose([
        albumentations.Resize(image_size, image_size),
        albumentations.Normalize()
    ])

    
    main()

