import torch
import numpy as np
import matplotlib; matplotlib.use('agg')
from fire import Fire

from airbus.callbacks.cyclic_lr import CyclicLR
from airbus.callbacks.learning_curve import LearningCurve
from airbus.callbacks.lr_on_plateau import LROnPlateau
from airbus.callbacks.lr_range_test import LRRangeTest
from airbus.callbacks.lr_schedule import LRSchedule
from airbus.callbacks.histogram import Histogram
from airbus.callbacks.model_checkpoint import ModelCheckpoint
from airbus.callbacks.model_checkpoint import load_checkpoint
from airbus.callbacks.prediction_grid import PredictionGrid
from airbus.callbacks.weight_grid import WeightGrid
from airbus.generators import get_train_generator
from airbus.generators import get_validation_generator
from airbus.loggers import make_loggers
from airbus.losses import lovasz_hinge_loss
from airbus.metrics import mean_iou
from airbus.metrics import f2_score
from airbus.models.devilnet import Devilnet
from airbus.training import fit_model
from airbus.utils import as_cuda

def pred_fn(outputs, batch):
    return torch.sigmoid(outputs['mask'])[:, 0, :, :], batch['mask']

def compute_loss(outputs, batch):
    true_masks = batch['mask'].clone()
    true_masks[true_masks > 1] = 1
    true_labels = (true_masks.sum(dim=(1, 2)) > 0)
    true_label_indices = true_labels.nonzero().view(-1)
    loss = torch.nn.functional.binary_cross_entropy_with_logits(outputs['presence'], true_labels.float()[:, None])
    if len(true_label_indices) > 0:
        loss += lovasz_hinge_loss(outputs['mask'][true_label_indices], true_masks[true_label_indices])
    return loss

def fit(
        num_epochs=100,
        limit=None,
        validation_limit=None,
        batch_size=16,
        lr=.005,
        checkpoint_path=None,
        telegram=False,
        visualize=False,
        num_folds=5,
        train_fold_ids=[0, 1, 2, 3],
        validation_fold_ids=[4]
    ):
    torch.backends.cudnn.benchmark = True
    np.random.seed(1991)
    logger, image_logger = make_loggers(telegram)

    if checkpoint_path:
        model = load_checkpoint(checkpoint_path)
    else:
        model = Devilnet(1)

    model = as_cuda(model)
    optimizer = torch.optim.SGD(filter(lambda param: param.requires_grad, model.parameters()), lr, weight_decay=1e-3, momentum=0.9, nesterov=True)
    train_generator = get_train_generator(num_folds, train_fold_ids, batch_size, limit)
    callbacks = [
        ModelCheckpoint(model, type(model).__name__.lower(), 'val_f2_score', 'max', logger),
        # CyclicLR(step_size=len(train_generator) * 2, min_lr=0.0001, max_lr=0.005, optimizer=optimizer, logger=logger),
        # LRSchedule(optimizer, [(0, 0.003), (2, 0.01), (12, 0.001), (17, 0.0001)], logger),
        # LRRangeTest(0.00001, 1.0, 20000, optimizer, image_logger),
        LROnPlateau('val_f2_score', optimizer, mode='max', factor=0.5, patience=4, min_lr=0, logger=logger),
        # ConfusionMatrix([0, 1], logger)
    ]

    if visualize:
        callbacks.extend([
            LearningCurve(['train_loss', 'val_loss', 'train_mean_iou', 'val_mean_iou', 'train_f2_score', 'val_f2_score'], image_logger),
            PredictionGrid(80, image_logger, mean_iou, pred_fn),
            Histogram(image_logger, mean_iou),
            WeightGrid(model, image_logger, 32)
        ])

    fit_model(
        model=model,
        train_generator=train_generator,
        validation_generator=get_validation_generator(num_folds, validation_fold_ids, 2, validation_limit),
        optimizer=optimizer,
        loss_fn=compute_loss,
        num_epochs=num_epochs,
        logger=logger,
        callbacks=callbacks,
        metrics=[mean_iou, f2_score]
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
