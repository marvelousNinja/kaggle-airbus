import glob
from functools import partial

import cv2
import numpy as np
import pandas as pd
import torch

def get_train_validation_holdout_split(records):
    np.random.shuffle(records)
    n = len(records)
    train = records[:int(n * .6)]
    validation = records[int(n * .6):int(n * .75)]
    holdout = records[int(n * .75)]
    return train, validation, holdout

def get_images_in(path):
    return np.sort(glob.glob(f'{path}/*.jpg'))

def read_image(path):
    return cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)

def normalize(image):
    return (image.astype(np.float32) / 255 - [0.485, 0.456, 0.406]) / [0.229, 0.224, 0.225]

def channels_first(image):
    return np.moveaxis(image, 2, 0)

def decode_rle_mask(shape, encoded_mask):
    if encoded_mask == 'nan': return np.zeros(shape)
    numbers = np.array(list(map(int, encoded_mask.split())))
    starts, lengths = numbers[::2], numbers[1::2]
    # Enumerates from 1
    starts -= 1
    ends = starts + lengths
    mask = np.zeros(shape[0] * shape[1], dtype=np.uint8)
    for start, end in zip(starts, ends): mask[start:end] += 1
    mask = np.clip(mask, a_min=0, a_max=2)
    return mask.reshape(shape).T

def load_mask(mask_db, shape, image_path):
    image_id = image_path.split('/')[-1]
    encoded_mask = ' '.join(mask_db[mask_db['ImageId'] == image_id]['EncodedPixels'].fillna('nan').values)
    return decode_rle_mask(shape, encoded_mask)

def combine_masks(masks):
    return np.clip(sum(masks), a_min=0, a_max=1)

def pipeline(mask_db, path):
    image = read_image(path)
    image = normalize(image)
    image = channels_first(image)
    mask = load_mask(mask_db, image.shape[1:], path)

    images = []
    masks = []
    for i in range(3):
        for j in range(3):
            images.append(image[:, i * 256: i * 256 + 256])
            masks.append(mask[i * 256: i * 256 + 256])
    return images, masks

def get_mask_db(path):
    return pd.read_csv(path)

def as_cuda(tensor):
    if torch.cuda.is_available():
        return tensor.cuda()
    return tensor

def from_numpy(obj):
    if torch.cuda.is_available():
        return torch.cuda.FloatTensor(obj)
    else:
        return torch.FloatTensor(obj)

def to_numpy(tensor):
    return tensor.data.cpu().numpy()

if __name__ == '__main__':
    import pdb; pdb.set_trace()
