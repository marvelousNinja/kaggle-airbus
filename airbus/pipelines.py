import cv2
import numpy as np
from skimage.measure import regionprops

from albumentations import (Blur, Compose, Crop, GaussNoise,
                            HorizontalFlip, MedianBlur,
                            MotionBlur, Normalize, OneOf,
                            RandomBrightness, RandomGamma, RandomSizedCrop, RandomCrop,
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

class ChannelsFirst:
    def __call__(self, **args):
        args['image'] = channels_first(args['image'])
        return args

class RandomCropWithBbox:
    def __init__(self, height, width):
        self.height = height
        self.width = width

    def __call__(self, **args):
        regions = regionprops(args['mask'])
        if len(regions) < 1: return RandomCrop(self.height, self.width)(**args)
        max_height, max_width, _ = args['image'].shape
        min_row, min_col, max_row, max_col = np.random.choice(regions).bbox

        boundary_min_row = max(max_row - self.height, 0)
        boundary_max_row = min(min_row + self.height, max_height)
        boundary_min_col = max(max_col - self.width, 0)
        boundary_max_col = min(min_col + self.width, max_width)

        return Compose([
            Crop(boundary_min_col, boundary_min_row, boundary_max_col, boundary_max_row),
            RandomCrop(self.height, self.width)
        ])(**args)

def train_pipeline(mask_db, path):
    image, mask = read_image_and_mask(mask_db, path)
    args = Compose([
        RandomCropWithBbox(256, 256),
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

def validation_pipeline(mask_db, path):
    image, mask = read_image_and_mask(mask_db, path)
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
