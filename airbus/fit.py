from functools import partial

import torch
import torchvision
import numpy as np
from fire import Fire
from tqdm import tqdm

from airbus.generators import get_train_generator
from airbus.generators import get_validation_generator
from airbus.models import Unet
from airbus.training import fit_model
from airbus.utils import as_cuda

def compute_loss(logits, labels):
    return torch.nn.functional.cross_entropy(logits, labels.long())

def fit(num_epochs=100, limit=None, batch_size=16, lr=.001):
    np.random.seed(1991)
    model = Unet()
    model = as_cuda(model)
    optimizer = torch.optim.SGD(filter(lambda param: param.requires_grad, model.parameters()), lr, momentum=.95, nesterov=True)

    fit_model(
        model=model,
        train_generator=get_train_generator(batch_size, limit),
        validation_generator=get_validation_generator(batch_size, limit),
        optimizer=optimizer,
        loss_fn=compute_loss,
        num_epochs=num_epochs
    )

def prof():
    import profile
    import pstats
    profile.run('fit()', 'fit.profile')
    stats = pstats.Stats('fit.profile')
    stats.sort_stats('cumulative').print_stats(30)
    import pdb; pdb.set_trace()

if __name__ == '__main__':
    Fire()
