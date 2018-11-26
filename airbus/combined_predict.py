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

def predict(classifier_path, segmenter_path, directory_path='data/test', batch_size=1):
    torch.backends.cudnn.benchmark = True
    torch.set_grad_enabled(False)

    classifier = as_cuda(load_checkpoint(classifier_path))
    classifier.eval()

    segmenter = as_cuda(load_checkpoint(segmenter_path))
    segmenter.eval()

    import time
    inference_start = time.time()
    # TODO AS: Warmup
    # image_paths = list(map(lambda path: path.split('/')[-1], get_images_in(directory_path)))
    image_paths = get_images_in(directory_path)
    all_image_paths = list(image_paths[::-1].copy())
    positive_image_paths = []
    negative_image_paths = []
    test_generator = get_test_generator(image_paths, batch_size, limit=None, classification=True)
    for batch in tqdm(test_generator, total=len(test_generator)):
        batch = from_numpy(batch)
        # TODO AS: TTA as a switch?
        #batch['image'] = torch.cat([
        #    batch['image'],
        #    batch['image'].flip(dims=(3,))
        #], 0)

        probs = torch.sigmoid(classifier(batch)['has_ships'])
        pred_labels = probs.round().long()[:, 0]
        #pred_labels = ((probs[:int(len(probs) / 2), 0] + probs[int(len(probs) / 2):, 0]) / 2).round().long()

        for label in pred_labels:
            if label == 1:
                positive_image_paths.append(all_image_paths.pop())
            else:
                negative_image_paths.append(all_image_paths.pop())

    all_image_paths = list(positive_image_paths[::-1].copy())
    test_generator = get_test_generator(positive_image_paths, batch_size, limit=None, classification=False)
    records = []
    for batch in tqdm(test_generator, total=len(test_generator)):
        batch = from_numpy(batch)
        # TODO AS: TTA as a switch?
        #batch['image'] = torch.cat([
        #    batch['image'],
        #    batch['image'].flip(dims=(3,))
        #], 0)

        probs = torch.sigmoid(segmenter(batch)['mask'])
        probs = to_numpy(probs[:, 0])
        #probs = to_numpy((probs[:int(len(probs) / 2), 0] + probs[int(len(probs) / 2):, 0].flip(dims=(2,))) / 2)

        for mask in probs:
            _id = all_image_paths.pop().split('/')[-1]
            instance_masks = extract_instance_masks_from_soft_mask(mask)
            if len(instance_masks) < 1:
                records.append((_id, None))
            else:
                for instance_mask in instance_masks:
                    records.append((_id, encode_rle(instance_mask)))

    negative_ids = list(map(lambda path: path.split('/')[-1], negative_image_paths))
    for _id in negative_ids:
        records.append((_id, None))

    image_ids, encoded_pixels = zip(*records)
    df = pd.DataFrame({'ImageId': image_ids, 'EncodedPixels': encoded_pixels})
    df.to_csv('./data/submissions/__latest_combined.csv', index=False)
    inference_end = time.time()
    print('Inference Time: %0.2f Minutes'%((inference_end - inference_start)/60))

if __name__ == '__main__':
    Fire(predict)
