import glob

import cv2
import numpy as np
import pandas as pd
import torch
from scipy import ndimage

def get_fold_split(mask_db, num_folds, drop_duplicates=True, drop_empty=True, validation=False):
    # TODO AS: Utilize non-empty images
    np.random.seed(1991)
    location_db = pd.read_csv('data/location_ids_v2.csv')
    db = pd.merge(mask_db, location_db, left_on='ImageId', right_on='ImageId')
    location_ids = db['BigImageId'].unique()
    fold_ids = np.random.randint(0, num_folds, len(location_ids))
    mapping = dict(zip(location_ids, fold_ids))
    db['FoldId'] = db['BigImageId'].map(mapping)

    if validation:
        db = db.drop_duplicates('BigImageId')
        with_ships = db[db['EncodedPixels'].notnull()]
        without_ships = db[db['EncodedPixels'].isnull()]
        db = pd.concat([with_ships, without_ships.sample(int(len(with_ships) * 1.08))])
        db = db.sample(frac=1)
        return db['ImageId'].values, db['FoldId'].values

    if drop_duplicates:
        db = db.drop_duplicates(['ImageId'])
    if drop_empty:
        db = db[db['EncodedPixels'].notnull()]

    db = db.sample(frac=1)

    return db['ImageId'].values, db['FoldId'].values

def get_images_in(path):
    return np.sort(glob.glob(f'{path}/*.jpg'))

def read_image(path):
    return cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)

def encode_rle(mask):
    pixels = mask.T.flatten()
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return ' '.join(str(x) for x in runs)

def decode_rle(shape, encoded_mask):
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

def extract_instance_masks_from_binary_mask(binary_mask):
    masks = []
    labelled_mask = ndimage.label(binary_mask)[0]
    for label in np.unique(labelled_mask):
        if label == 0: continue
        mask = np.zeros(labelled_mask.shape)
        mask[labelled_mask == label] = 1
        area = mask.sum()
        if area >= 50: masks.append(mask)
    return masks

def extract_instance_masks_from_labelled_mask(labelled_mask):
    masks = []
    for label in np.unique(labelled_mask):
        if label == 0: continue
        mask = np.zeros(labelled_mask.shape)
        mask[labelled_mask == label] = 1
        masks.append(mask)
    return masks

def load_mask(mask_db, shape, image_path):
    image_id = image_path.split('/')[-1]
    labelled_mask = np.zeros(shape)

    for i, encoded_mask in enumerate(mask_db[mask_db['ImageId'] == image_id]['EncodedPixels'].fillna('nan')):
        labelled_mask[decode_rle(shape, encoded_mask) == 1] = i + 1
    return labelled_mask.astype(np.uint8)

def get_mask_db(path):
    return pd.read_csv(path)

def as_cuda(tensor):
    if torch.cuda.is_available():
        return tensor.cuda()
    return tensor

def from_numpy(obj):
    if isinstance(obj, dict):
        return {key: from_numpy(value) for key, value in obj.items()}

    if torch.cuda.is_available():
        if isinstance(obj, torch.Tensor): return obj.float().cuda(non_blocking=True)
        return torch.cuda.FloatTensor(obj)
    else:
        if isinstance(obj, torch.Tensor): return obj.float()
        return torch.FloatTensor(obj)

def to_numpy(obj):
    if isinstance(obj, dict):
        return {key: to_numpy(value) for key, value in obj.items()}
    if isinstance(obj, tuple):
        return tuple(map(to_numpy, obj))
    return obj.data.cpu().numpy()

if __name__ == '__main__':
    import pdb; pdb.set_trace()
