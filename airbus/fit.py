from functools import partial

import torch
import numpy as np
from fire import Fire
from tqdm import tqdm

from airbus.generators import get_train_generator
from airbus.generators import get_validation_generator
from airbus.linknet import Linknet
from airbus.model_checkpoint import ModelCheckpoint
from airbus.training import fit_model
from airbus.utils import as_cuda
from airbus.utils import confusion_matrix

def dice_loss(logits, labels):
    probs = torch.nn.functional.softmax(logits, dim=1)
    probs = probs[:, 1, :, :]
    labels = labels.float()
    intersection = (probs * labels).sum((1, 2))
    pred_volume = probs.sum((1, 2))
    true_volume = labels.sum((1, 2))
    return (1 - 2 * intersection / (pred_volume + true_volume + 1.0)).mean()

def generalized_dice_loss(logits, labels):
    labels = labels.long()
    probs = torch.nn.functional.softmax(logits, dim=1)
    numerator = 0
    denominator = 0
    for label in [0, 1]:
        pred_probs = probs[:, label, :, :]
        true_labels = (labels == label).float()
        weight = (true_labels.sum() + 1) ** (-2)
        numerator += weight * (pred_probs * true_labels).sum((1, 2))
        denominator += weight * (pred_probs.sum((1, 2)) + true_labels.sum((1, 2)) + 1)
    return (1 - 2 * numerator / denominator).mean()

def compute_loss(logits, labels):
    return generalized_dice_loss(logits, labels)

def after_validation(model_checkpoint, val_loss, outputs, gt):
    tqdm.write(confusion_matrix(np.argmax(outputs, axis=1), gt, [0, 1, 2]))
    model_checkpoint.step(val_loss)

def fit(num_epochs=100, limit=None, batch_size=16, lr=.001):
    torch.backends.cudnn.benchmark = True
    np.random.seed(1991)
    model = Linknet(2)
    model = as_cuda(model)
    optimizer = torch.optim.Adam(filter(lambda param: param.requires_grad, model.parameters()), lr)
    model_checkpoint = ModelCheckpoint(model, 'linknet', tqdm.write)

    fit_model(
        model=model,
        train_generator=get_train_generator(batch_size, limit),
        # TODO AS: Colab explodes with out of memory
        validation_generator=get_validation_generator(batch_size, 160),
        optimizer=optimizer,
        loss_fn=compute_loss,
        num_epochs=num_epochs,
        after_validation=partial(after_validation, model_checkpoint)
    )

def prof():
    import profile
    import pstats
    profile.run('fit(batch_size=4, limit=100, num_epochs=1)', 'fit.profile')
    stats = pstats.Stats('fit.profile')
    stats.sort_stats('cumulative').print_stats(30)
    import pdb; pdb.set_trace()

if __name__ == '__main__':
    Fire()
