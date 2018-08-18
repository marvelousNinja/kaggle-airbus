import glob
from functools import partial
from itertools import product

import cv2
import numpy as np
import pandas as pd
import torch
from skimage.morphology import dilation
from skimage.morphology import square
from skimage.morphology import watershed
from tabulate import tabulate

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
    labelled_mask = np.zeros(shape)
    for encoded_mask in mask_db[mask_db['ImageId'] == image_id]['EncodedPixels'].fillna('nan'):
        labelled_mask += decode_rle_mask(shape, encoded_mask)
    labelled_mask[labelled_mask > 0] = 1
    return labelled_mask

def pipeline(mask_db, path):
    image = read_image(path)
    image = normalize(image)
    image = channels_first(image)
    mask = load_mask(mask_db, image.shape[1:], path)
    return image, mask

def confusion_matrix(pred_labels, true_labels, labels):
    pred_labels = pred_labels.reshape(-1)
    true_labels = true_labels.reshape(-1)
    columns = [list(map(lambda label: f'Pred {label}', labels))]
    for true_label in labels:
        counts = []
        preds_for_label = pred_labels[np.argwhere(true_labels == true_label)]
        for predicted_label in labels:
            counts.append((preds_for_label == predicted_label).sum())
        columns.append(counts)

    headers = list(map(lambda label: f'True {label}', labels))
    rows = np.column_stack(columns)
    return tabulate(rows, headers, 'grid')

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
