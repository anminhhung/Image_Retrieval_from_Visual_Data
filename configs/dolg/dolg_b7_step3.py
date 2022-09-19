cfg_b7_step3 = {
    "model":{
        "backbone":"tf_efficientnet_b7_ns",
        "n_classes":17,
        "pretrained":True,
        "stride":(1,1),
        "pool":"gem",
        "gem_p_trainable":True,
        "embedding_size":512,
        "dilations":[3,6,9]
    },
    "train":{
        "model_name":"efficientnet_b7_ns_step3",
        "train_step":0,
        "image_size":256,
        "save_per_epoch":True,
        "batch_size":4,
        'image_per_batch': 64,
        'num_instance': 16,
        "num_workers":2,
        "init_lr":0.00005, #0.0001
        "n_epochs":2,
        "start_from_epoch":1,
        "use_amp":False,
				"model_dir":"./run/saved",
        "CUDA_VISIBLE_DEVICES":"0",
        "arcface_s":45, #80
        "arcface_m":0.3,
        'local_rank': 0,
        'sampler': 'id_uniform'
    },
    "val": {
        'batch_size': 2,
        'num_workers': 5,
        'image_size': 256,
        'num_instance': 16,
        'image_per_batch': 32,
        'sampler': 'id_uniform'
      },
    'inference': {
      "image_size": 256,
      "batch_size": 2,
      "num_workers": 2,
      "out_dim": 17,
      "TOP_K": 5,
      "CLS_TOP_K": 5,
      }

}