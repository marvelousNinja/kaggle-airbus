import numpy as np
import torch
from tqdm import tqdm

from airbus.utils import from_numpy
from airbus.utils import to_numpy

def fit_model(
        model,
        train_generator,
        validation_generator,
        optimizer,
        loss_fn,
        num_epochs,
        logger,
        after_validation=None
    ):

    for epoch in tqdm(range(num_epochs)):
        train_loss = 0
        num_batches = len(train_generator)
        model.train()
        torch.set_grad_enabled(True)
        for inputs, gt in tqdm(train_generator, total=num_batches):
            inputs, gt = from_numpy(inputs), from_numpy(gt)
            optimizer.zero_grad()
            loss = loss_fn(model(inputs), gt)
            loss.backward()
            optimizer.step()
            train_loss += loss.data[0]
        train_loss /= num_batches

        val_loss = 0
        all_outputs = []
        all_gt = []
        num_batches = len(validation_generator)
        model.eval()
        torch.set_grad_enabled(False)
        for inputs, gt in tqdm(validation_generator, total=num_batches):
            all_gt.append(gt)
            inputs, gt = from_numpy(inputs), from_numpy(gt)
            outputs = model(inputs)
            val_loss += loss_fn(outputs, gt).data[0]
            all_outputs.append(to_numpy(outputs))
        val_loss /= num_batches

        if after_validation:
            after_validation(val_loss, np.concatenate(all_outputs), np.concatenate(all_gt))

        logger(f'epoch {epoch} train loss {train_loss:.5f} - val loss {val_loss:.5f}')
