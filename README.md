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
python -u -m torch.distributed.launch --nproc_per_node=1 train_DOLG.py --kernel-type b5ns_DDP_final_256_300w_f2_10ep --train-step 0 --data-dir data --image-size 256 --batch-size 32 --n-epochs 10 --fold 2  --CUDA_VISIBLE_DEVICES 0
```

**Train Swin Transformer**
```
python -u -m torch.distributed.launch --nproc_per_node=1 train_swin.py --kernel-type b5ns_DDP_final_256_300w_f2_10ep --train-step 0 --data-dir data --image-size 224 --batch-size 8 --n-epochs 10 --fold 2  --CUDA_VISIBLE_DEVICES 0
```

**Note**- If you want run with others kernel-type. Check github: https://github.com/haqishen/Google-Landmark-Recognition-2020-3rd-Place-Solution 


**Predict**

```
!python predict.py --backbone tf_efficientnet_b5_ns
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
