import math
from functools import partial

import numpy as np

from airbus.utils import get_images_in
from airbus.utils import get_mask_db
from airbus.utils import get_train_validation_holdout_split
from airbus.utils import pipeline

class DataGenerator:
    def __init__(self, records, batch_size, transform, batched_pipeline=True, shuffle=True):
        self.records = records
        self.batch_size = batch_size
        self.transform = transform
        self.shuffle = shuffle
        self.batched_pipeline = batched_pipeline

    def __iter__(self):
        if self.shuffle: np.random.shuffle(self.records)
        batch = []

        for output in map(self.transform, self.records):
            # TODO AS: Infer this? Or force pipeline to always return array
            if self.batched_pipeline:
                batched_output = output
                for output in zip(*batched_output):
                    batch.append(output)
                    if len(batch) >= self.batch_size:
                        split_outputs = list(zip(*batch))
                        yield map(np.stack, split_outputs)
                        batch = []
            else:
                batch.append(output)
                if len(batch) >= self.batch_size:
                    split_outputs = list(zip(*batch))
                    import pdb; pdb.set_trace()
                    yield map(np.stack, split_outputs)
                    batch = []

        if len(batch) > 0:
            split_outputs = list(zip(*batch))
            yield map(np.stack, split_outputs)

    def __len__(self):
        # TODO AS: 9 patches per image. Clean this up
        return math.ceil(len(self.records) / self.batch_size) * 9

def get_validation_generator(batch_size, limit=None):
    mask_db = get_mask_db('data/train_ship_segmentations.csv')
    images_with_ships = mask_db[mask_db['EncodedPixels'].notnull()]['ImageId'].unique()
    image_paths = 'data/train/' + images_with_ships
    _, image_paths, _ = get_train_validation_holdout_split(image_paths)
    transform = partial(pipeline, mask_db)
    return DataGenerator(image_paths[:limit], batch_size, transform)

def get_train_generator(batch_size, limit=None):
    mask_db = get_mask_db('data/train_ship_segmentations.csv')
    images_with_ships = mask_db[mask_db['EncodedPixels'].notnull()]['ImageId'].unique()
    image_paths = 'data/train/' + images_with_ships
    image_paths, _, _ = get_train_validation_holdout_split(image_paths)
    mask_db = get_mask_db('data/train_ship_segmentations.csv')
    transform = partial(pipeline, mask_db)
    return DataGenerator(image_paths[:limit], batch_size, transform)
