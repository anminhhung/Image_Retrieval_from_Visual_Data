cfg_swin_224_b3 = {
    'model': {
        "backbone":"swin_base_patch4_window7_224",
        "n_classes":17,
        "neck": "option-D", # type neck layer
        "pretrained":True,
        "stride":None, 
        "pool":"gem",
        "gem_p_trainable":True,
        "embedder": "tf_efficientnet_b3_ns",
        "embedding_size":512,
        "dilations":[3,6,9],
        "image_size": (896,896), # size in HybridEmbed layer
        "freeze_backbone_head": False
      },
    'train': {
        'model_name': 'swin_224_b3_efficientnet_b3_ns',
        'train_step': 0,
        'image_size': 224, 
        'save_per_epoch': True,
        'batch_size': 32,
        'num_workers': 2,
        'init_lr': 0.00005, #1e-4
        'n_epochs': 2,
        'start_from_epoch': 1,
        'use_amp': False,
        'load_pretrain': 'Not_load',
        'train_list_file_path': 'train/train_list.txt',
        'model_dir': 'saved', # save model
        'CUDA_VISIBLE_DEVICES': '0', # set device
        'arcface_s': 45, # arcface loss
        'local_rank': 0
      },
    'inference': {
        'image_size': 224,
        'batch_size': 16,
        'num_workers': 2,
        'out_dim': 17,
        'TOP_K': 5,
        'CLS_TOP_K': 5,
      }
    }  