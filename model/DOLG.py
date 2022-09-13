import timm
from torch import nn
import torch
import math
from torch.nn import functional as F
from torch.nn.parameter import Parameter
import numpy as np

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
    def __init__(self, cfg):
        super(DOLG, self).__init__()

        self.n_classes = cfg['model']['n_classes']
        self.backbone = timm.create_model(cfg['model']['backbone'], 
                                          pretrained=cfg['model']['pretrained'], 
                                          num_classes=0, 
                                          global_pool="", 
                                          features_only = True)

        
        if ("efficientnet" in cfg['model']['backbone']) & (cfg['model']['stride'] is not None):
            self.backbone.conv_stem.stride = cfg['model']['stride']

        backbone_out = self.backbone.feature_info[-1]['num_chs']
        backbone_out_1 = self.backbone.feature_info[-2]['num_chs']
        
        feature_dim_l_g = 1024
        fusion_out = 2 * feature_dim_l_g

        if cfg['model']['pool'] == "gem":
            self.global_pool = GeM(p_trainable=cfg['model']['gem_p_trainable'])
        elif cfg['model']['pool'] == "identity":
            self.global_pool = torch.nn.Identity()
        elif cfg['model']['pool'] == "avg":
            self.global_pool = nn.AdaptiveAvgPool2d(1)

        self.fusion_pool = nn.AdaptiveAvgPool2d(1)
        self.embedding_size = cfg['model']['embedding_size']

        self.neck = nn.Sequential(
                nn.Linear(fusion_out, cfg['model']['embedding_size'], bias=True),
                nn.BatchNorm1d(cfg['model']['embedding_size']),
                torch.nn.PReLU()
            )

        self.head_in_units = cfg['model']['embedding_size']
        self.head = ArcMarginProduct_subcenter(cfg['model']['embedding_size'], cfg['model']['n_classes'])
    
        self.mam = MultiAtrousModule(backbone_out_1, feature_dim_l_g, cfg['model']['dilations'])
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

        return x_emb, logits_m

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
