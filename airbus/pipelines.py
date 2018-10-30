import cv2
import numpy as np
from albumentations import Compose, Crop, Normalize

from airbus.utils import load_mask, read_image

def channels_first(image):
    return np.moveaxis(image, 2, 0)

def resize(size, interpolation, image):
    return cv2.resize(image, size, interpolation=interpolation)

def read_image_and_mask(mask_db, target_shape, path):
    image = read_image(path)
    mask = load_mask(mask_db, image.shape[:2], path)
    image = resize(target_shape, cv2.INTER_LINEAR, image)
    if mask is not None:
        mask = resize(target_shape, cv2.INTER_NEAREST, mask)
    return image, mask

def read_image_and_mask_cached(cache, mask_db, target_shape, path):
    if cache.get(path): return cache[path]
    image, mask = read_image_and_mask(mask_db, target_shape, path)
    cache[path] = (image, mask)
    return image, mask

class ChannelsFirst:
    def __call__(self, **args):
        args['image'] = channels_first(args['image'])
        return args

def train_pipeline(cache, mask_db, path):
    image, mask = read_image_and_mask_cached(cache, mask_db, (768, 768), path)
    args = Compose([
        Crop(0, 0, 256, 256),
        Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ChannelsFirst()
    ])(image=image, mask=mask)
    return args['image'], args.get('mask')

def validation_pipeline(cache, mask_db, path):
    image, mask = read_image_and_mask_cached(cache, mask_db, (768, 768), path)
    args = Compose([
        Crop(0, 0, 256, 256),
        Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ChannelsFirst()
    ])(image=image, mask=mask)
    return args['image'], args.get('mask')