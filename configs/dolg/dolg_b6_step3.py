cfg_b6 = {
		"model":{
				"backbone":"tf_efficientnet_b6_ns",
				"n_classes":17,
				"pretrained":True,
				"stride":None,
				"pool":"gem",
				"gem_p_trainable":True,
				"embedding_size":512,
				"dilations":[3,6,9]
      },
		"train":{
				"model_name":"efficientnet_b6_ns_step3",
				"train_step":0,
				"image_size":256, 
				"save_per_epoch":True,
				"batch_size":16,
				"num_workers":2,
        "weight_decay":1e-4,
				"init_lr":0.0001,
				"n_epochs":2,
				"start_from_epoch":1,
				"use_amp":False,
				"model_dir":"./run/saved",
				"CUDA_VISIBLE_DEVICES":"0",
				"arcface_s":45,
        "arcface_m":0.3,
        'local_rank': 0
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