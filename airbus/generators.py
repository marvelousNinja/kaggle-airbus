import math
from functools import partial
from multiprocessing.pool import ThreadPool

import numpy as np

from airbus.pipelines import train_pipeline
from airbus.pipelines import validation_pipeline
from airbus.utils import get_images_in
from airbus.utils import get_mask_db
from airbus.utils import get_area_stratified_split

class DataGenerator:
    def __init__(self, records, batch_size, transform, shuffle=True, drop_last=False):
        self.records = records
        self.batch_size = batch_size
        self.transform = transform
        self.shuffle = shuffle
        self.drop_last = drop_last

    def __iter__(self):
        if self.shuffle: np.random.shuffle(self.records)
        batch = []
        pool = ThreadPool()
        prefetch_size = 2000
        num_slices = len(self.records) // prefetch_size + 1

        for i in range(num_slices):
            start = i * prefetch_size
            end = start + prefetch_size
            for output in pool.imap(self.transform, self.records[start:end]):
                batch.append(output)
                if len(batch) >= self.batch_size:
                    split_outputs = list(zip(*batch))
                    yield list(map(np.stack, split_outputs))
                    batch = []

        if (not self.drop_last) and len(batch) > 0:
            split_outputs = list(zip(*batch))
            yield list(map(np.stack, split_outputs))

        pool.close()

    def __len__(self):
        num_batches = len(self.records) / self.batch_size
        if self.drop_last:
            return math.floor(num_batches)
        else:
            return math.ceil(num_batches)

def get_validation_generator(num_folds, fold_ids, batch_size, limit=None):
    mask_db = get_mask_db('data/train.csv')
    all_image_ids, all_fold_ids = get_area_stratified_split(mask_db, num_folds)
    image_ids = all_image_ids[np.isin(all_fold_ids, fold_ids)]
    image_paths = list(map(lambda id: f'data/train/images/{id}.png', image_ids))
    transform = partial(validation_pipeline, {}, mask_db)
    return DataGenerator(image_paths[:limit], batch_size, transform, shuffle=False, drop_last=True)

def get_train_generator(num_folds, fold_ids, batch_size, limit=None):
    mask_db = get_mask_db('data/train.csv')
    all_image_ids, all_fold_ids = get_area_stratified_split(mask_db, num_folds)
    image_ids = all_image_ids[np.isin(all_fold_ids, fold_ids)]
    image_paths = list(map(lambda id: f'data/train/images/{id}.png', image_ids))
    transform = partial(train_pipeline, {}, mask_db)
    return DataGenerator(image_paths[:limit], batch_size, transform, drop_last=True)

def get_test_generator(batch_size, limit=None):
    mask_db = get_mask_db('data/train.csv')
    image_paths = get_images_in('data/test/images')
    transform = partial(validation_pipeline, {}, mask_db)
    return DataGenerator(image_paths[:limit], batch_size, transform, shuffle=False)
