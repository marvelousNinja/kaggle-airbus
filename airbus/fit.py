from functools import partial

import torch
import torchvision
import numpy as np
from fire import Fire
from tqdm import tqdm

from airbus.generators import get_train_generator
from airbus.generators import get_validation_generator
from airbus.linknet import Linknet
from airbus.model_checkpoint import ModelCheckpoint
from airbus.training import fit_model
from airbus.utils import as_cuda

def compute_loss(logits, labels):
    return torch.nn.functional.cross_entropy(logits, labels.long())

def after_validation(model_checkpoint, val_loss):
    model_checkpoint.step(val_loss)

def fit(num_epochs=100, limit=None, batch_size=16, lr=.001):
    torch.backends.cudnn.benchmark = True
    np.random.seed(1991)
    model = Linknet(3)
    model = as_cuda(model)
    optimizer = torch.optim.SGD(filter(lambda param: param.requires_grad, model.parameters()), lr, momentum=.95, nesterov=True)
    model_checkpoint = ModelCheckpoint(model, 'linknet', tqdm.write)

    fit_model(
        model=model,
        train_generator=get_train_generator(batch_size, limit),
        validation_generator=get_validation_generator(batch_size, limit),
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
