<h1>HCM AI CHALLENGE 2022 - Event Retrieval from Visual Data</h1>

---
## To do task 
- [x] DOLG
- [x] Hybrid swin transformer
- [x] Reranking top 3 2019
- [x] Reranking top 3 2020
- [x] ID uniform sampling
- [x] Softmax uniform sampling 
- [ ] Continent-aware
- [ ] Mish activation 
- [ ] Gradient Clipping 
---

## Requirements

```
pip install -r requirements.txt
```

---
## Setup 
**If apex folder is not exist**

```
!git clone https://github.com/NVIDIA/apex
%cd apex
!python setup.py install
```

**Train**

**Train DOLG**
```
python -u -m torch.distributed.launch --nproc_per_node=1 \
          train_DOLG.py \
          --config_name dolg_b5_step3 \
          --trainCSVPath ./data/train/train_list.txt \
          --valCSVPath ./data/train/test_train_list.txt \
          --use_wandb
```

**Load Trained Model and Continue Training**
```
python -u -m torch.distributed.launch --nproc_per_node=1 \
          train_DOLG.py \
          --config_name dolg_b5_step3 \
          --trainCSVPath ./data/train/train_list.txt \
          --valCSVPath ./data/train/test_train_list.txt \
          --checkpoint './run/saved/dolg_efficientnet_b5_ns_step3_2.pth' \
          --use_wandb
```
          
**Train Swin Transformer**
```
python -u -m torch.distributed.launch \
          --nproc_per_node=1 train_swin.py \
          --config_name swin_224_b5 \
          --trainCSVPath ./dataset/data/train/train_list.txt \
          --valCSVPath ./data/train/val_list.txt \
          --use_wandb
```

**Load Trained Model and Continue Training**
```
python -u -m torch.distributed.launch --nproc_per_node=1 \
          train_swin.py \
          --config_name swin_224_b5 \
          --trainCSVPath ./data/train/train_list.txt \
          --valCSVPath ./data/train/val_list.txt \
          --checkpoint './run/saved/swin_224_b3_efficientnet_b5_ns.pth' \
          --use_wandb
```
**Note**- Source: https://github.com/haqishen/Google-Landmark-Recognition-2020-3rd-Place-Solution 

**[Optional]**
```
!pip install wandb -qqq
import wandb
wandb.login()
```

**Note**- To use wandb to visuallize while training. You have to enter your API key to login. Check https://wandb.ai/authorize to get API key.

**Predict**

**Predict DOLG with reranking method ("top1_3_2019","top3_2020","top1_shopee")**

```
python predict_DOLG.py \
  --config_name dolg_b7_step3 \
  --reranking_method top1_3_2019 \
  --weight_path ./run/saved/dolg_efficientnet_b7_ns_step3_2.pth \
  --data_dir ./data \
  --trainCSVPath ./data/train/train_list.txt \
  --testCSVPath ./data/train/test_list.txt \
  --trainH5Path ./data/train/train.h5 \
  --indexH5Path ./data/train/index.h5 
```

**Predict Swin Transformer with reranking method ("top1_3_2019","top3_2020","top1_shopee")**

```
python predict_swin.py \
  --config_name swin_224_b5 \
  --reranking_method top1_3_2019 \
  --weight_path /content/drive/MyDrive/AIC_HCM/DOLG/DOLG_GIT/saved/dolg_swin_224_b3_efficientnet_b5_ns_1.pth \
  --data_dir /content/drive/MyDrive/AIC_HCM/DOLG/DOLG-pytorch/dataset/data \
  --trainCSVPath /content/drive/MyDrive/AIC_HCM/DOLG/DOLG-pytorch/dataset/data/train/train_list.txt \
  --testCSVPath /content/drive/MyDrive/AIC_HCM/DOLG/DOLG-pytorch/dataset/data/train/test_list.txt \
  --trainH5Path /content/drive/MyDrive/AIC_HCM/DOLG/DOLG-pytorch/dataset/data/train/train.h5 \
  --indexH5Path /content/drive/MyDrive/AIC_HCM/DOLG/DOLG-pytorch/dataset/data/train/index.h5 
```

---
## Folder Structure

```
  Main-folder/
  │
  ├── config/ 
  │   ├── config.py - configuration
  │
  ├── data/ - default directory for storing input data
  │
  ├── data_loader/ - anything about data loading goes here
  │   └── data_loaders.py
  |
  ├── model/ - this folder contains any net of your project.
  │   ├── model.py
  │   ├── metric.py
  │   └── loss.py
  │
  ├── saved/
  │   ├── models/ - trained models are saved here
  │   └── log/ - default logdir for tensorboard and logging 
  │   └── submission/ -  submission file are saved here
  │
  ├── scripts/ - main function 
  │   └── pipeline.py
  │   └── OCR.py
  │   └── segment.py
  |
  ├── test/ - test functions
  │   └── run.py
  │   └── ...
  |
  ├── tools/ - open source are saved here
  │   └── detectron2 dir
  │   └── ...
  │  
  └── utils/ - small utility functions
      ├── util.py
      └── ...
```
