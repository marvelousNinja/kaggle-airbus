import numpy as np
import pandas as pd
import torch
from fire import Fire
from tqdm import tqdm

from airbus.model_checkpoint import load_checkpoint
from airbus.generators import get_test_generator
from airbus.utils import as_cuda
from airbus.utils import encode_rle
from airbus.utils import extract_instance_masks_from_binary_mask
from airbus.utils import from_numpy
from airbus.utils import get_images_in
from airbus.utils import resize
from airbus.utils import to_numpy

def predict(checkpoint_path, batch_size=1, limit=None):
    model = load_checkpoint(checkpoint_path)
    model = as_cuda(model)
    torch.set_grad_enabled(False)
    model.eval()

    records = []
    ids = list(map(lambda path: path.split('/')[-1], get_images_in('data/test')))[:limit]
    test_generator = get_test_generator(batch_size, limit)
    for inputs, _ in tqdm(test_generator, total=len(test_generator)):
        inputs = from_numpy(inputs)
        outputs = model(inputs)
        masks = to_numpy(torch.argmax(outputs, dim=1))
        for mask in masks:
            _id = ids.pop(0)
            instance_masks = extract_instance_masks_from_binary_mask(mask)

            if len(instance_masks) == 0:
                records.append((_id, None))
            else:
                for instance_mask in instance_masks:
                    records.append((_id, encode_rle(instance_mask)))

    image_ids, encoded_pixels = zip(*records)
    df = pd.DataFrame({'ImageId': image_ids, 'EncodedPixels': encoded_pixels})
    df.to_csv('./data/submissions/__latest.csv', index=False)

if __name__ == '__main__':
    Fire(predict)
