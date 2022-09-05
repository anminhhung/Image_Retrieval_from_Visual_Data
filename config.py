import os
import pandas as pd
import cv2

class Config:
    DATA_DIR = 'data'
    MODEL_DIR = '/content/drive/MyDrive/AIC_HCM/DOLG/DOLG-pytorch/model/weights'
    LOG_DIR = '/content/drive/MyDrive/AIC_HCM/DOLG/DOLG-pytorch/model/logs'

    # DATA_DIR = '/content/drive/.shortcut-targets-by-id/16sHXxeyDT8ujcHkEcNcq0PrZlM8MvoJF/DOLG-pytorch/dataset/data'
    # MODEL_DIR = '/content/drive/.shortcut-targets-by-id/16sHXxeyDT8ujcHkEcNcq0PrZlM8MvoJF/DOLG-pytorch/model/weights'
    # LOG_DIR = '/content/drive/.shortcut-targets-by-id/16sHXxeyDT8ujcHkEcNcq0PrZlM8MvoJF/DOLG-pytorch/model/logs'

    
    # CSV_PATH = os.path.join(DATA_DIR, 'train/train_list.txt')
    CSV_PATH = "data/train/train_list.txt"
    

    train_batch_size = 32
    val_batch_size = 10
    num_workers = 2
    # image_size = 224
    image_size = 768 

    output_dim = 224
    hidden_dim = 1024
    input_dim = 3
    epochs = 10
    lr = 1e-4
    num_of_classes = 17
    pretrained = True
    model_name = 'resnet101'
    # backbone = "tf_efficientnet_b5_ns"
    seed = 42

    # start_from_epoch=1
    # stop_at_epoch=999
    # fold=0
    # train_step=0
    # CUDA_VISIBLE_DEVICES="0"
    # local_rank=0
    # kernel_type='b5ns_DDP_final_256_300w_f2_10ep'
    # use_amp = False
    # load_from=''
    # stride = 2 

class EfficientnetB5_Config:
    debug = True
    seed = 42

    stride = 2    
    # paths
    name = os.path.basename(__file__).split(".")[0]
    # data_dir = "/content/drive/MyDrive/AIC_HCM/DOLG/DOLG-pytorch/dataset/data/"
    data_dir = "data/"

    data_folder = data_dir + "train/"
    train_df = "/mount/glr2021/data/2021/train_gldv2x.csv"

    val_df = '/raid/landmark-recognition-2019/' + "recognition_solution_v2.1.csv"
    output_dir = f"/mount/glr2021/models/{os.path.basename(__file__).split('.')[0]}"
    val_data_folder = "/raid/landmark-recognition-2019/" + "test/"

    test = False
    test_data_folder = data_dir + "test/"

    eval_retrieval = True
    query_data_folder = "/raid/landmark-recognition-2019/" + "test/"
    index_data_folder = "/raid/landmark-recognition-2019/" + "index/"
    query_df = '/mount/glr2021/data/2019/query_v2.csv'
    index_df = '/mount/glr2021/data/2019/index_v2.csv'

    #logging
    neptune_project = "xxx"
    neptune_connection_mode = "debug"
    tags = "debug"


    # MODEL
    model = "ch_mdl_dolg_efficientnet"
    dilations = [6,12,18]
    backbone = "tf_efficientnet_b5_ns"
    # backbone = "resnet101"
    neck = "option-D"
    embedding_size = 512
    pool = "gem"
    gem_p_trainable = True
    # pretrained_weights = '/mount/glr2021/models/cfg_ch_dolg_2_2head_s2c/fold0/checkpoint_last_seed472505.pth'
    # pretrained_weights_strict = True
    pretrained=True
    # DATASET
    dataset = "ch_ds_1"
    normalization = 'imagenet'
    landmark_id2class_id = pd.read_csv('data/train/train_list.txt')
    num_workers = 0
    # data_sample = 100000
    loss = 'adaptive_arcface'
    arcface_s = 45
    arcface_m = 0.3
    in_channels = 3


    # OPTIMIZATION & SCHEDULE

    # fold = 0
    lr = 0.00005
    # optimizer = "adam"
    # weight_decay = 1e-4
    warmup = 1
    epochs = 7
    # stop_at = 16
    save_headless = False
    batch_size = 16
    mixed_precision = True
    pin_memory = False
    grad_accumulation = 1.

    #inference
    train = True
    val = True
    test = False
    save_val_data = True
    train_val = False
    save_only_last_ckpt = False
    eval_ddp =True
    img_size = (768,768)

    n_classes=17
    headless = False