import cv2
import numpy as np

from albumentations import (Blur, Compose, Crop, GaussNoise,
                            HorizontalFlip, MedianBlur,
                            MotionBlur, Normalize, OneOf,
                            RandomBrightness, RandomGamma, RandomSizedCrop,
                            RandomRotate90, ShiftScaleRotate)

from airbus.utils import load_mask, read_image

def channels_first(image):
    return np.moveaxis(image, 2, 0)

def resize(size, interpolation, image):
    return cv2.resize(image, size, interpolation=interpolation)

def read_image_and_mask(mask_db, path):
    image = read_image(path)
    mask = load_mask(mask_db, image.shape[:2], path)
    return image, mask

def read_image_and_mask_cached(preproc, cache, mask_db, path):
    if cache.get(path): return cache[path]
    image, mask = read_image_and_mask(mask_db, path)
    args = preproc(image=image, mask=mask)
    cache[path] = (args.get('image').copy(), args.get('mask').copy())
    return cache.get(path)

class ChannelsFirst:
    def __call__(self, **args):
        args['image'] = channels_first(args['image'])
        return args

def train_pipeline(cache, mask_db, path):
    image, mask = read_image_and_mask_cached(Crop(0, 0, 256, 256), cache, mask_db, path)
    args = Compose([
        HorizontalFlip(p=0.5),
        RandomRotate90(p=0.5),
        OneOf([
            ShiftScaleRotate(),
            RandomSizedCrop(min_max_height=(200, 240), height=256, width=256)
        ], p=0.2),
        GaussNoise(p=0.2),
        OneOf([
            RandomBrightness(limit=0.2),
            RandomGamma(),
        ], p=0.5),
        OneOf([
            Blur(),
            MedianBlur(),
            MotionBlur()
        ], p=0.2),
        Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ChannelsFirst()
    ])(image=image, mask=mask)
    return {'image': args['image'], 'mask': args['mask']}

def validation_pipeline(cache, mask_db, path):
    image, mask = read_image_and_mask_cached(Crop(0, 0, 768, 768), cache, mask_db, path)
    args = Compose([
        Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ChannelsFirst()
    ])(image=image, mask=mask)
    return {'image': args['image'], 'mask': args['mask']}

def test_pipeline(path):
    image = read_image(path)
    args = Compose([
        Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ChannelsFirst()
    ])(image=image)
    return {'image': args['image']}
