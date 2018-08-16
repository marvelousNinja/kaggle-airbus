import glob
from functools import partial

import cv2
import numpy as np
import pandas as pd
import torch
from skimage.morphology import dilation
from skimage.morphology import square
from skimage.morphology import watershed

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
    i = 1
    labelled_mask = np.zeros(shape)
    for encoded_mask in mask_db[mask_db['ImageId'] == image_id]['EncodedPixels'].fillna('nan'):
        mask = decode_rle_mask(shape, encoded_mask)
        labelled_mask[mask > 0] = i
        i += 1

    return label_touching_borders(labelled_mask)

def label_touching_borders(labelled_mask):
    # 0 = background, 1 = ships, 2 = touching borders
    dilated = dilation(labelled_mask > 0, square(9))
    after_watershed = watershed(dilated, labelled_mask, mask=dilated, watershed_line=True) > 0
    watershed_line = dilated ^ after_watershed
    watershed_line = dilation(watershed_line, square(7))
    mask = np.zeros(labelled_mask.shape, dtype=np.uint8)
    mask[labelled_mask > 0] = 1
    for (x, y) in np.argwhere(watershed_line):
        diff = 1 if mask[x, y] > 0 else 3
        x_lo = np.clip(x - diff, 0, labelled_mask.shape[0])
        y_lo = np.clip(y - diff, 0, labelled_mask.shape[1])
        x_hi = np.clip(x + diff + 1, 0, labelled_mask.shape[0])
        y_hi = np.clip(y + diff + 1, 0, labelled_mask.shape[1])
        around = labelled_mask[x_lo:x_hi, y_lo:y_hi]
        around = around[around > 0]
        num_instances = len(np.unique(around))
        if num_instances < 2: continue
        mask[x, y] = 2
    return mask

def pipeline(mask_db, path):
    image = read_image(path)
    image = normalize(image)
    image = channels_first(image)
    mask = load_mask(mask_db, image.shape[1:], path)

    images = []
    masks = []
    for i, j in zip(range(3), range(3)):
        images.append(image[:, i * 256: j * 256 + 256])
        masks.append(mask[i * 256: j * 256 + 256])
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
