cfg_b5 = {
    'model': {
        'backbone': 'tf_efficientnet_b5_ns',
        'n_classes': 17, 
        'pretrained': True,
        'stride': 2,
        'pool': 'gem', # gem pool config
        'gem_p_trainable': True,
        'embedding_size': 512,
        'dilations': [6,12,18]
      },
    'train': {
        'model_name': 'efficientnet_b5_ns_step3',
        'data_dir': 'data',
        'train_step': 0,
        'image_size': 768, #256
        'save_per_epoch': True,
        'batch_size': 4,
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
        'image_size': 768,
        'batch_size': 4,
        'num_workers': 2,
        'out_dim': 17,
        'TOP_K': 5,
        'CLS_TOP_K': 5,
        'weight_path': './saved/dolg_efficientnet_b5_ns_step3_2.pth'
      }
    }  