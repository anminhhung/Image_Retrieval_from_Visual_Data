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
				"data_dir":"data",
				"train_step":0,
				"image_size":256,
				"save_per_epoch":True,
				"batch_size":32,
				"num_workers":2,
				"init_lr":0.0001,
				"n_epochs":2,
				"start_from_epoch":1,
				"use_amp":False,
				"load_pretrain":"Not_load",
				"train_list_file_path":"train/train_list.txt",
				"model_dir":"saved",
				"CUDA_VISIBLE_DEVICES":"0",
				"arcface_s":80,
        'local_rank': 0
      },
    'inference': {
      'image_size': 256,
      'batch_size': 16,
      'num_workers': 2,
      'out_dim': 17,
      'TOP_K': 5,
      'CLS_TOP_K': 5,
      'weight_path': './saved/dolg_efficientnet_b6_ns_step3_2.pth'
      }
		}
