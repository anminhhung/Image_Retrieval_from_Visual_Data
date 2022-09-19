cfg_b5 = {
    "model": {
        'backbone': 'tf_efficientnet_b5_ns',
        'n_classes': 127, 
        'pretrained': True,
        'stride': None,
        'pool': 'gem', # gem pool config
        'gem_p_trainable': True,
        'embedding_size': 512,
        'dilations': [6,12,18]
      },
    "train": {
        'model_name': 'efficientnet_b5_ns_step3',
        'train_step': 0,
        'image_size': 256, 
        'save_per_epoch': True,
        'batch_size': 32,
        'image_per_batch': 64,
        'num_instance': 16,
        'num_workers': 2,
        'init_lr': 0.00005, #1e-4
        'n_epochs': 100,
        'start_from_epoch': 1,
        'use_amp': False,
        'model_dir': './run/saved', # save model
        'CUDA_VISIBLE_DEVICES': '0', # set device
        'arcface_s': 45, # arcface loss
        'local_rank': 0,
        'sampler': 'id_uniform'
      },
    "val": {
        'batch_size': 2,
        'num_workers': 2,
        'image_size': 256,
        'num_instance': 16,
        'image_per_batch': 32,
        'sampler': 'id_uniform'
      },
    "inference": {
        'image_size': 256,
        'batch_size': 4,
        'num_workers': 2,
        'out_dim': 17,
        'TOP_K': 5,
        'CLS_TOP_K': 5,
      }

    }  