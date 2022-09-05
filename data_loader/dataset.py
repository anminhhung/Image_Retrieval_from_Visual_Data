import os
import jpeg4py as jpeg
import pandas as pd
from sklearn import preprocessing
import torch
from torch.utils.data import Dataset
from data_loader.transform import image_transform
from config import Config
# from config import EfficientnetB5_Config as cfg
from torchvision import transforms
import albumentations as A
import albumentations.pytorch
import cv2


import numpy as np


def img_path_from_id(id):
    split_id = id.split("_")
    sub_folder = "_".join(split_id[:-1])
    img_path = os.path.join(Config.DATA_DIR, 'train',
                            sub_folder, f'{id}.jpg')
    return img_path


class LmkRetrDataset(Dataset):
    def __init__(self):
        self.df = pd.read_csv(Config.CSV_PATH)
        self.landmark_id_encoder = preprocessing.LabelEncoder()
        self.df['landmark_id'] = self.landmark_id_encoder.fit_transform(
            self.df['landmark_id'])
        self.df['path'] = self.df['id'].apply(img_path_from_id)
        self.paths = self.df['path'].values
        self.ids = self.df['id'].values
        self.landmark_ids = self.df['landmark_id'].values
        self.transform = image_transform
        # self.mode = mode

    def __len__(self):
        return len(self.df)
      
    def __getitem__(self, idx):
        path, id, landmark_id = self.paths[idx], self.ids[idx], self.landmark_ids[idx]
        # img = cv2.imread(path)[:,:,::-1]
        # img = img.astype(np.float32)

        img = cv2.imread(path)
        image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # auto = A.Compose([
        #     A.Resize(768,768),
        #     A.Normalize(),
        #     A.pytorch.transforms.ToTensorV2()
        # ])
        
        MEAN = [0.485, 0.456, 0.406]
        STD = [0.229, 0.224, 0.225]

        transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((768, 768)),
            transforms.ToTensor(),
            transforms.Normalize(MEAN, STD),
        ])

        img = transform(img)

        return img, torch.tensor(landmark_id)