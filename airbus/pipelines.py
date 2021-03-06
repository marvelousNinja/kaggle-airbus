import cv2
import numpy as np
from skimage.measure import regionprops

from albumentations import (Blur, Compose, Crop, GaussNoise,
                            HorizontalFlip, MedianBlur, Resize,
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

def mask_to_bbox(mask):
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]
    return rmin, rmax, cmin, cmax

class RandomCropWithBbox:
    def __init__(self, height, width):
        self.height = height
        self.width = width

    def __call__(self, **args):
        region_labels = np.unique(args['mask'])
        region_labels = region_labels[region_labels > 0]
        if len(region_labels) < 1: return RandomCrop(self.height, self.width)(**args)
        max_height, max_width, _ = args['image'].shape
        region_label = np.random.choice(region_labels)
        min_row, max_row, min_col, max_col = mask_to_bbox(args['mask'] == region_label)

        bbox_height = max_row - min_row
        if bbox_height >= self.height:
            boundary_min_row = min_row
            boundary_max_row = max_row
        else:
            boundary_min_row = max(max_row - self.height, 0)
            boundary_max_row = min(min_row + self.height, max_height)

        bbox_width = max_col - min_col
        if bbox_width >= self.width:
            boundary_min_col = min_col
            boundary_max_col = max_col
        else:
            boundary_min_col = max(max_col - self.width, 0)
            boundary_max_col = min(min_col + self.width, max_width)

        return Compose([
            Crop(boundary_min_col, boundary_min_row, boundary_max_col, boundary_max_row),
            RandomCrop(self.height, self.width)
        ])(**args)

class LabelShipPresence:
    def __call__(self, **args):
        args['has_ships'] = 1 if args['mask'].max() > 0 else 0
        return args

def train_classification_pipeline(mask_db, path):
    image, mask = read_image_and_mask(mask_db, path)
    args = Compose([
        Resize(224, 224),
        HorizontalFlip(p=0.5),
        RandomRotate90(p=0.5),
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
        LabelShipPresence(),
        ChannelsFirst()
    ])(image=image, mask=mask)
    return {'image': args['image'], 'has_ships': args['has_ships']}

def validation_classification_pipeline(mask_db, path):
    image, mask = read_image_and_mask(mask_db, path)
    args = Compose([
        Resize(224, 224),
        Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        LabelShipPresence(),
        ChannelsFirst()
    ])(image=image, mask=mask)
    return {'image': args['image'], 'has_ships': args['has_ships']}

def train_pipeline(mask_db, path):
    image, mask = read_image_and_mask(mask_db, path)
    args = Compose([
        RandomCropWithBbox(256, 256),
        HorizontalFlip(p=0.5),
        RandomRotate90(p=0.5),
        #TODO AS: Too hard for simpler networks
        #OneOf([
        #    ShiftScaleRotate(),
        #    RandomSizedCrop(min_max_height=(200, 240), height=256, width=256)
        #], p=0.2),
        #GaussNoise(p=0.2),
        #OneOf([
        #    RandomBrightness(limit=0.2),
        #    RandomGamma(),
        #], p=0.5),
        #OneOf([
        #    Blur(),
        #    MedianBlur(),
        #    MotionBlur()
        #], p=0.2),
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

def test_classification_pipeline(path):
    image = read_image(path)
    args = Compose([
        Resize(224, 224),
        Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ChannelsFirst()
    ])(image=image)
    return {'image': args['image']}
