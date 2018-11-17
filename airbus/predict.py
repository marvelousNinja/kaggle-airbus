from itertools import product

import pandas as pd
import torch
from fire import Fire
from tqdm import tqdm

from airbus.callbacks.model_checkpoint import load_checkpoint
from airbus.generators import get_test_generator
from airbus.utils import as_cuda
from airbus.utils import encode_rle
from airbus.utils import extract_instance_masks_from_soft_mask
from airbus.utils import from_numpy
from airbus.utils import get_images_in
from airbus.utils import to_numpy

def flip_image_batch(image_batch):
    return image_batch.flip(dims=(3,))

def rotate_image_batch(image_batch, k):
    return image_batch.rot90(k=k, dims=(2,3))

import matplotlib; matplotlib.use('agg')
def predict(checkpoint_path, batch_size=1, limit=None, tta=False):
    model = load_checkpoint(checkpoint_path)
    model = as_cuda(model)
    torch.set_grad_enabled(False)
    model.eval()

    records = []
    ids = list(map(lambda path: path.split('/')[-1], get_images_in('data/test')))[:limit]
    test_generator = get_test_generator(batch_size, limit)
    for batch in tqdm(test_generator, total=len(test_generator)):
        batch = from_numpy(batch)
        masks = None
        if tta:
            accumulated_outputs = 0
            for i, should_flip in product(range(4), [False, True]):
                image_batch = batch['image']
                image_batch = rotate_image_batch(image_batch, i)
                if should_flip: image_batch = flip_image_batch(image_batch)

                outputs = torch.sigmoid(model({'image': image_batch})['mask'])
                if should_flip: outputs = flip_image_batch(outputs)
                outputs = rotate_image_batch(outputs, -i)
                accumulated_outputs += outputs
            accumulated_outputs /= 8
            masks = to_numpy(accumulated_outputs[:, 0, :, :])
        else:
            outputs = model(batch)
            outputs['mask'] = torch.sigmoid(outputs['mask'])
            masks = to_numpy(outputs['mask'][:, 0, :, :])
        for mask in masks:
            _id = ids.pop(0)
            instance_masks = extract_instance_masks_from_soft_mask(mask)

            if len(instance_masks) == 0:
                records.append((_id, None))
            else:
                for instance_mask in instance_masks:
                    records.append((_id, encode_rle(instance_mask)))

    image_ids, encoded_pixels = zip(*records)
    df = pd.DataFrame({'ImageId': image_ids, 'EncodedPixels': encoded_pixels})
    df.to_csv('./data/submissions/__latest.csv', index=False)

if __name__ == '__main__':
    Fire(predict)
