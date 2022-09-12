<h1>HCM AI CHALLENGE 2022 - Event Retrieval from Visual Data</h1>

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
python -u -m torch.distributed.launch --nproc_per_node=1 train_DOLG.py --config_name dolg_b5_step3
```

**Train Swin Transformer**
```
python -u -m torch.distributed.launch --nproc_per_node=1 train_swin.py --config_name swin_224_b5
```

**Note**- If you want run with others kernel-type. Check github: https://github.com/haqishen/Google-Landmark-Recognition-2020-3rd-Place-Solution 


**Predict**

```
!python predict.py --backbone tf_efficientnet_b5_ns
```

**Predict Swin Transformer with reranking method ("new","top3")**
```
!python predict_swin_reranking.py --config_name swin_224_b5 --reranking_method new
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
