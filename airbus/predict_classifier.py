import pandas as pd
import torch
from fire import Fire
from tqdm import tqdm

from airbus.callbacks.model_checkpoint import load_checkpoint
from airbus.generators import get_test_generator
from airbus.utils import as_cuda
from airbus.utils import encode_rle
from airbus.utils import extract_instance_masks_from_binary_mask
from airbus.utils import from_numpy
from airbus.utils import get_images_in
from airbus.utils import to_numpy

def predict(checkpoint_path, submission_path, batch_size=1, limit=None, tta=False):
    model = load_checkpoint(checkpoint_path)
    model = as_cuda(model)
    torch.set_grad_enabled(False)
    model.eval()

    records = []
    ids = list(map(lambda path: path.split('/')[-1], get_images_in('data/test')))[:limit]
    submission = pd.read_csv(submission_path)
    print('Num masks before', len(submission))
    test_generator = get_test_generator(batch_size, limit, classification=True)
    for batch in tqdm(test_generator, total=len(test_generator)):
        batch = from_numpy(batch)
        outputs = model(batch)
        if tta:
            batch['image'] = batch['image'].flip(dims=(3,))
            flipped_outputs = model(batch)
            outputs['has_ships'] = (torch.sigmoid(outputs['has_ships']) + torch.sigmoid(flipped_outputs['has_ships'])) / 2
        else:
            outputs['has_ships'] = torch.sigmoid(outputs['has_ships'])
        pred_labels = to_numpy(outputs['has_ships'][:, 0].round().long())

        for pred in pred_labels:
            _id = ids.pop(0)
            if pred == 1: continue
            submission = submission[submission['ImageId'] != _id].copy()
            submission = submission.append({'ImageId': _id, 'EncodedPixels': None}, ignore_index=True)
    print('Num masks after', len(submission))
    submission.to_csv('./data/submissions/__latest_filtered.csv', index=False)

if __name__ == '__main__':
    Fire(predict)
