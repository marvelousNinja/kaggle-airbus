import glob
from functools import partial

import cv2
import numpy as np
import pandas as pd
import torch
from scipy import ndimage
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

def read_image_cached(cache, preprocess, path):
    image = cache.get(path)
    if image is not None:
        return image
    else:
        image = preprocess(read_image(path))
        cache[path] = image
        return image

def resize(size, image):
    return cv2.resize(image, size, interpolation=cv2.INTER_NEAREST)

def normalize(image):
    return (image.astype(np.float32) / 255 - [0.485, 0.456, 0.406]) / [0.229, 0.224, 0.225]

def channels_first(image):
    return np.moveaxis(image, 2, 0)

def encode_rle(mask):
    pixels = mask.flatten()
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

def load_mask_cached(cache, preprocess, mask_db, shape, path):
    mask = cache.get(path)
    if mask is not None:
        return mask
    else:
        mask = preprocess(load_mask(mask_db, shape, path))
        cache[path] = mask
        return mask

def crop(top, left, shape, image):
    return image[top:top + shape[0], left:left + shape[1]].copy()

def make_random_cropper(crop_shape, image_shape):
    top = np.random.randint(image_shape[0] - crop_shape[0])
    left = np.random.randint(image_shape[1] - crop_shape[1])
    return partial(crop, top, left, crop_shape)

def mask_to_bbox(mask):
    a = np.where(mask != 0)
    return np.array([np.min(a[0]), np.min(a[1]), np.max(a[1]), np.max(a[0])])

def random_crop_containing(crop_shape, image_shape, bbox):
    bbox_top, bbox_left, bbox_right, bbox_bottom = bbox
    top = max(bbox_bottom - crop_shape[0], 0)
    bottom = min(bbox_top + crop_shape[0], image_shape[0])
    left = max(bbox_right - crop_shape[1], 0)
    right = min(bbox_left + crop_shape[1], image_shape[1])
    top_shift = np.random.randint(max(bottom - top - crop_shape[0] + 1, 1))
    left_shift = np.random.randint(max(right - left - crop_shape[1] + 1, 1))
    return top + top_shift, left + left_shift, (crop_shape)

def pipeline(mask_db, cache, mask_cache, path):
    preprocess = lambda image: image
    image = read_image_cached(cache, preprocess, path)
    labelled_mask = load_mask_cached(mask_cache, preprocess, mask_db, (768, 768), path)
    random_mask_label = np.random.randint(labelled_mask.max()) + 1
    instance_mask = labelled_mask.copy()
    instance_mask[instance_mask != random_mask_label] = 0
    mask_bbox = mask_to_bbox(instance_mask)
    top, left, crop_shape = random_crop_containing((224, 224), (768, 768), mask_bbox)
    image = crop(top, left, crop_shape, image)
    image = normalize(image)
    image = channels_first(image)
    mask = crop(top, left, crop_shape, labelled_mask)
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
