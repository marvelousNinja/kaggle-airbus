from datetime import datetime
from functools import partial

import torch

from airbus.utils import as_cuda

def save_checkpoint(model, path):
    torch.save(model, path)

def load_checkpoint(path):
    if torch.cuda.is_available():
        return as_cuda(torch.load(path))
    else:
        return torch.load(path, map_location='cpu')

def generate_checkpoint_path(prefix, timestamp, epoch, loss):
    name = f'{prefix}-{timestamp}-{epoch:02d}-{loss:.5f}.pt'
    return f'./data/models/{name}'

class ModelCheckpoint:
    def __init__(self, model, prefix, logger=None):
        self.epoch = 0
        self.loss = float("inf")
        self.model = model
        self.logger = logger

        timestamp = datetime.utcnow().strftime('%Y%m%d_%H%M')
        self.generate_checkpoint_path = partial(generate_checkpoint_path, prefix, timestamp)

    def step(self, loss):
        if loss < self.loss:
            checkpoint_path = self.generate_checkpoint_path(self.epoch, loss)
            save_checkpoint(self.model, checkpoint_path)
            self.loss = loss
            if self.logger: self.logger(f'Checkpoint saved {checkpoint_path}')
        self.epoch += 1
