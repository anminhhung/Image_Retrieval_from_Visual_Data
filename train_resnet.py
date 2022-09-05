from pytorch_lightning.utilities.seed import seed_everything
from pytorch_lightning import Trainer

from model.dolg import DolgNet
from model.ch_mdl_dolg_efficientnet import Net_Lightning
from config import Config
from data_loader.dataset import LmkRetrDataset
from config import EfficientnetB5_Config as cfg
# import pandas as pd


seed_everything(cfg.seed)

# model = DolgNet(
#     input_dim=Config.input_dim,
#     hidden_dim=Config.hidden_dim,
#     output_dim=Config.output_dim,
#     num_of_classes=Config.num_of_classes,
#     # backbone='tf_efficientnet_b5_ns'
#     backbone='tv_resnet101'
    
# )

dataset = LmkRetrDataset()

model = Net_Lightning(
    cfg=cfg,
    dataset = dataset
)

trainer = Trainer(gpus=0, max_epochs=cfg.epochs)

trainer.fit(model)