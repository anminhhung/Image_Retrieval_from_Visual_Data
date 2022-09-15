import torch
import torch.distributed as dist

from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from data_loader.sampler import RandomIdentitySampler, ImageUniformSampler
from data_loader.dataset import LandmarkDataset, get_df, get_transforms

class ShortDistributedSampler(DistributedSampler):
    def __init__(self, dataset, **kwargs):
        DistributedSampler.__init__(self, dataset, **kwargs)

    def __len__(self):
        # Control how many iterations
        # return 1000 iter * batch_size * 8
        return self.num_samples


def make_dataloader(cfg, args):
    df, out_dim = df, out_dim = get_df(args.trainCSVPath)
    transforms_train, transforms_val = get_transforms(cfg['train']['image_size'])

    dataset_train = LandmarkDataset(df, 'train', 'train', transform=transforms_train)
    sampler = "img_uniform"

    if sampler == 'id_uniform':
        print('using id_uniform sampler')
        mini_batch_size = cfg['train']['batch_size'] 
        data_sampler = RandomIdentitySampler(args.trainCSVPath, cfg['train']['image_per_batch'], cfg['train']['num_instance']) 
        batch_sampler = torch.utils.data.sampler.BatchSampler(data_sampler, mini_batch_size, True)
        
        train_loader = torch.utils.data.DataLoader(
            dataset_train,
            num_workers=cfg['train']['num_workers'],
            batch_sampler=batch_sampler,
            # collate_fn=train_collate_fn,
            pin_memory=True,
        )
    elif sampler == 'img_uniform':
        print('using img_uniform sampler')
        mini_batch_size = cfg['train']['batch_size'] 
        data_sampler = ImageUniformSampler(args.trainCSVPath, cfg['train']['image_per_batch'], cfg['train']['num_instance']) 
        batch_sampler = torch.utils.data.sampler.BatchSampler(data_sampler, mini_batch_size, True)

        train_sampler = torch.utils.data.distributed.DistributedSampler(dataset_train)

        train_loader = torch.utils.data.DataLoader(
            dataset_train,
            num_workers=cfg['train']['num_workers'],
            batch_sampler=batch_sampler,
            # collate_fn=train_collate_fn,
            pin_memory=True,
        )
    elif sampler == 'softmax':
        print('using softmax sampler')
        local_rank = cfg['train']['local_rank']
        world_size = dist.get_world_size()
        # datasampler = DistributedSampler(train_set, num_replicas=world_size, rank=local_rank, seed=cfg.SOLVER.SEED)
        data_sampler = ShortDistributedSampler(args.trainCSVPath, num_replicas=world_size, rank=local_rank, seed=0)
        train_loader = DataLoader(dataset_train, batch_size=cfg['train']['batch_size'] , sampler=data_sampler,
                                  num_workers=cfg['train']['num_workers']) #  #cfg.SOLVER.IMS_PER_BATCH=64
    else:
        assert 'unsupported sampler! expected id_uniform, img_uniform, softmax but got {}'.format(cfg.SAMPLER)

    return train_loader