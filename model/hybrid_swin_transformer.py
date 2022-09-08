import timm
from torch import nn
import torch
from torch.nn import functional as F
from torch import nn
import math
from torch.nn import functional as F
from torch.nn.parameter import Parameter
import numpy as np

from timm.models.vision_transformer_hybrid import HybridEmbed    

class Swish(torch.autograd.Function):
    @staticmethod
    def forward(ctx, i):
        result = i * torch.sigmoid(i)
        ctx.save_for_backward(i)
        return result

    @staticmethod
    def backward(ctx, grad_output):
        i = ctx.saved_variables[0]
        sigmoid_i = torch.sigmoid(i)
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

class ArcFaceLossAdaptiveMargin(nn.modules.Module):
    def __init__(self, margins, s=30.0):
        super().__init__()
        self.crit = DenseCrossEntropy()
        self.s = s
        self.margins = margins
            
    def forward(self, logits, labels, out_dim):
        ms = []
        ms = self.margins[labels.cpu().numpy()]
        cos_m = torch.from_numpy(np.cos(ms)).float().cuda()
        sin_m = torch.from_numpy(np.sin(ms)).float().cuda()
        th = torch.from_numpy(np.cos(math.pi - ms)).float().cuda()
        mm = torch.from_numpy(np.sin(math.pi - ms) * ms).float().cuda()
        labels = F.one_hot(labels, out_dim).float()
        logits = logits.float()
        cosine = logits
        sine = torch.sqrt(1.0 - torch.pow(cosine, 2))
        phi = cosine * cos_m.view(-1,1) - sine * sin_m.view(-1,1)
        phi = torch.where(cosine > th.view(-1,1), phi, cosine - mm.view(-1,1))
        output = (labels * phi) + ((1.0 - labels) * cosine)
        output *= self.s
        loss = self.crit(output, labels)
        return loss  

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

class SwinTransformer(nn.Module):
    def __init__(self):
        super(SwinTransformer, self).__init__()
        self.n_classes = 17
        backbone = "swin_base_patch4_window7_224"
        neck = "option-D"

        self.backbone = timm.create_model(backbone, 
                                          pretrained=True, 
                                          num_classes=0, 
                                        )

        embedder = timm.create_model("tf_efficientnet_b3_ns", 
                                      pretrained=True, 
                                      features_only=True, out_indices=[1])

        
        self.backbone.patch_embed = HybridEmbed(embedder,img_size=896, 
                                              patch_size=1, 
                                              feature_size=self.backbone.patch_embed.grid_size, 
                                              in_chans=3, 
                                              embed_dim=self.backbone.embed_dim)

        self.global_pool = GeM(p_trainable=True)

        if "xcit_small_24_p16" in backbone:
            backbone_out = 384
        elif "xcit_medium_24_p16" in backbone:
            backbone_out = 512
        elif "xcit_small_12_p16" in backbone:
            backbone_out = 384
        elif "xcit_medium_12_p16" in backbone:
            backbone_out = 512   
        elif "swin" in backbone:
            backbone_out = self.backbone.num_features
        elif "vit" in backbone:
            backbone_out = self.backbone.num_features
        elif "cait" in backbone:
            backbone_out = self.backbone.num_features
        else:
            backbone_out = 2048 

        self.embedding_size = 512

        # https://www.groundai.com/project/arcface-additive-angular-margin-loss-for-deep-face-recognition
        if neck == "option-D":
            self.neck = nn.Sequential(
                nn.Linear(backbone_out, 512, bias=True),
                nn.BatchNorm1d(512),
                torch.nn.PReLU()
            )
        elif neck == "option-F":
            self.neck = nn.Sequential(
                nn.Dropout(0.3),
                nn.Linear(backbone_out, 512, bias=True),
                nn.BatchNorm1d(512),
                torch.nn.PReLU()
            )
        elif neck == "option-X":
            self.neck = nn.Sequential(
                nn.Linear(backbone_out, 512, bias=False),
                nn.BatchNorm1d(512),
            )
            
        elif neck == "option-S":
            self.neck = nn.Sequential(
                nn.Linear(backbone_out, 512),
                Swish_module()
            )

        self.head_in_units = 512
        self.head = ArcMarginProduct_subcenter(512, 17)
        
    def forward(self, x):
        x = self.backbone(x)
        x_emb = self.neck(x)
        
        logits_m = self.head(x_emb)

        return logits_m 