import torch
import torchvision
from functools import partial
from multiprocessing.pool import ThreadPool

import numpy as np

from airbus.pipelines import test_pipeline
from airbus.pipelines import train_pipeline
from airbus.pipelines import validation_pipeline
from airbus.utils import get_images_in
from airbus.utils import get_mask_db
from airbus.utils import get_fold_split

class ImageDataset(torch.utils.data.Dataset):
    def __init__(self, samples, transform):
        self.samples = samples
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        return self.transform(self.samples[i])

def get_validation_generator(num_folds, fold_ids, batch_size, limit=None):
    mask_db = get_mask_db('data/train_ship_segmentations_v2.csv')
    all_image_ids, all_fold_ids = get_fold_split(mask_db, num_folds)
    image_ids = all_image_ids[np.isin(all_fold_ids, fold_ids)]
    image_paths = list(map(lambda id: f'data/train/{id}', image_ids))
    transform = partial(validation_pipeline, mask_db)
    dataset = ImageDataset(image_paths[:limit], transform)
    num_workers = 16 if torch.cuda.is_available() else 0
    return torch.utils.data.DataLoader(
        dataset,
        num_workers=num_workers,
        batch_size=batch_size,
        shuffle=False,
        drop_last=True,
        pin_memory=True
    )

def get_train_generator(num_folds, fold_ids, batch_size, limit=None):
    mask_db = get_mask_db('data/train_ship_segmentations_v2.csv')
    all_image_ids, all_fold_ids = get_fold_split(mask_db, num_folds)
    image_ids = all_image_ids[np.isin(all_fold_ids, fold_ids)]
    image_paths = list(map(lambda id: f'data/train/{id}', image_ids))
    transform = partial(train_pipeline, mask_db)
    dataset = ImageDataset(image_paths[:limit], transform)
    num_workers = 16 if torch.cuda.is_available() else 0
    return torch.utils.data.DataLoader(
        dataset,
        num_workers=num_workers,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        pin_memory=True
    )

def get_test_generator(batch_size, limit=None):
    image_paths = get_images_in('data/test')
    dataset = ImageDataset(image_paths[:limit], test_pipeline)
    num_workers = 16 if torch.cuda.is_available() else 0
    return torch.utils.data.DataLoader(
        dataset,
        num_workers=num_workers,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        pin_memory=True
    )
