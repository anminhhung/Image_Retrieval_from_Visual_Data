import torch
import torch.distributed as dist
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from .dataset import LandmarkDataset, get_df, get_transforms
from .sampler import RandomIdentitySampler
from .sampler_ddp import RandomIdentitySampler_DDP, ImageUniformSampler_DDP, GLDSampler_DDP



class ShortDistributedSampler(DistributedSampler):
    def __init__(self, dataset, **kwargs):
        DistributedSampler.__init__(self, dataset, **kwargs)

    def __len__(self):
        # Control how many iterations
        # return 1000 iter * batch_size * 8
        return self.num_samples

# def train_collate_fn(batch):
#     imgs, pids, camids, viewids , _ = zip(*batch)
#     pids = torch.tensor(pids, dtype=torch.int64)
#     return torch.stack(imgs, dim=0), pids

# def val_collate_fn(batch):
#     imgs, pids, camids, viewids, img_paths = zip(*batch)
#     return torch.stack(imgs, dim=0), pids, camids, img_paths

def make_dataloader(cfg, dataset_train, dataset_valid):
    num_workers = cfg["train"]["num_workers"]

    valid_loader = torch.utils.data.DataLoader(dataset_valid, batch_size=cfg['val']['batch_size'], num_workers=cfg['val']['num_workers'])

    if cfg["dataloader"]["sampler"] == 'id_uniform':
        print('using id_uniform sampler')

        mini_batch_size = cfg["train"]["batch_size"],
        data_sampler = RandomIdentitySampler_DDP(dataset.train, cfg["train"]["batch_size"], cfg["dataloader"]["num_instance"])
        batch_sampler = torch.utils.data.sampler.BatchSampler(data_sampler, mini_batch_size, True)

        train_loader = torch.utils.data.DataLoader(
            dataset_train,
            num_workers=num_workers,
            batch_sampler=batch_sampler,
            # collate_fn=train_collate_fn,
            pin_memory=True,
        )
    elif cfg["dataloader"]["sampler"] == 'gld':
        print('using gld sampler')

        mini_batch_size = cfg["train"]["batch_size"]
        data_sampler = GLDSampler_DDP(dataset.train, cfg["train"]["batch_size"], cfg["dataloader"]["num_instance"])
        batch_sampler = torch.utils.data.sampler.BatchSampler(data_sampler, mini_batch_size, True)

        train_loader = torch.utils.data.DataLoader(
            dataset_train,
            num_workers=num_workers,
            batch_sampler=batch_sampler,
            # collate_fn=train_collate_fn,
            pin_memory=True,
        )
    elif cfg["dataloader"]["sampler"] == 'img_uniform':
        print('using img_uniform sampler')

        mini_batch_size = cfg["train"]["batch_size"]
        data_sampler = ImageUniformSampler_DDP(dataset.train, cfg["train"]["batch_size"], cfg["dataloader"]["num_instance"])
        batch_sampler = torch.utils.data.sampler.BatchSampler(data_sampler, mini_batch_size, True)

        train_loader = torch.utils.data.DataLoader(
            dataset_train,
            num_workers=num_workers,
            batch_sampler=batch_sampler,
            # collate_fn=train_collate_fn,
            pin_memory=True,
        )
    elif cfg["dataloader"]["sampler"] == 'softmax':
        print('using softmax sampler')

        local_rank = cfg["train"]["local_rank"]
        world_size = dist.get_world_size()
        # datasampler = DistributedSampler(dataset_train, num_replicas=world_size, rank=local_rank, seed=cfg.SOLVER.SEED)
        data_sampler = ShortDistributedSampler(dataset_train, num_replicas=world_size, rank=local_rank, seed=0)
        train_loader = DataLoader(
            dataset_train,
            num_workers=num_workers, 
            sampler=data_sampler,
            batch_size=cfg["train"]["batch_size"], 
            # collate_fn=train_collate_fn
        )
    else:
        print('unsupported sampler! expected id_uniform, img_uniform, softmax but got {}'.format(cfg["dataloader"]["sampler"]))

    return train_loader, valid_loader
