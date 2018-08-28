import io
from functools import partial

import torch
import telegram_send
import numpy as np
import matplotlib; matplotlib.use('agg')
import matplotlib.pyplot as plt
from fire import Fire

from airbus.cyclic_lr import CyclicLR
from airbus.generators import get_train_generator
from airbus.generators import get_validation_generator
from airbus.f_score import f2_score
from airbus.linknet import Linknet
from airbus.losses import jaccard_loss
from airbus.model_checkpoint import ModelCheckpoint
from airbus.model_checkpoint import load_checkpoint
from airbus.loggers import make_loggers
from airbus.training import fit_model
from airbus.utils import as_cuda
from airbus.utils import confusion_matrix
from airbus.utils import extract_instance_masks_from_binary_mask
from airbus.utils import extract_instance_masks_from_labelled_mask
from airbus.utils import visualize_learning_curve
from airbus.utils import visualize_predictions

def compute_loss(logits, labels):
    labels[labels > 1] = 1
    return jaccard_loss(logits, labels) * 0.5 + torch.nn.functional.cross_entropy(logits, labels.long()) * 0.5

def calc_f2_score(logits, labelled_masks):
    binary_masks = np.argmax(logits, axis=1)
    pred_masks = list(extract_instance_masks_from_binary_mask(mask) for mask in binary_masks)
    true_masks = list(extract_instance_masks_from_labelled_mask(mask) for mask in labelled_masks)
    return f2_score(pred_masks, true_masks)

def on_validation_end(history, visualize, image_logger, logger, model_checkpoint, train_loss, val_loss, logits, gt):
    logger(f'F2 Score: {calc_f2_score(logits, gt)}')
    gt[gt > 1] = 1
    if visualize:
        history.setdefault('train_losses', []).append(train_loss)
        history.setdefault('val_losses', []).append(val_loss)
        visualize_predictions(image_logger, logits, gt)
        visualize_learning_curve(image_logger, history['train_losses'], history['val_losses'])
    logger(confusion_matrix(np.argmax(logits, axis=1), gt, [0, 1]))
    model_checkpoint.step(val_loss)

def fit(num_epochs=100, limit=None, batch_size=16, lr=.001, checkpoint_path=None, telegram=False, visualize=False):
    torch.backends.cudnn.benchmark = True
    np.random.seed(1991)
    logger, image_logger = make_loggers(telegram)

    if checkpoint_path:
        model = load_checkpoint(checkpoint_path)
    else:
        model = Linknet(2)

    model = as_cuda(model)
    optimizer = torch.optim.Adam(filter(lambda param: param.requires_grad, model.parameters()), lr)
    model_checkpoint = ModelCheckpoint(model, 'linknet', logger)
    train_generator = get_train_generator(batch_size, limit)
    cyclic_lr = CyclicLR(cycle_iterations=len(train_generator) * 2, min_lr=0.0001, max_lr=0.005, optimizer=optimizer, logger=logger)
    history = {}

    fit_model(
        model=model,
        train_generator=train_generator,
        validation_generator=get_validation_generator(batch_size, 160),
        optimizer=optimizer,
        loss_fn=compute_loss,
        num_epochs=num_epochs,
        on_validation_end=partial(on_validation_end, history, visualize, image_logger, logger, model_checkpoint),
        on_batch_end=cyclic_lr.step,
        logger=logger
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
