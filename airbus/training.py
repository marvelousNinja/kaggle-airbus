import numpy as np
from tqdm import tqdm

from airbus.utils import from_numpy
from airbus.utils import to_numpy

def fit_model(
        model,
        train_generator,
        validation_generator,
        optimizer,
        loss_fn,
        num_epochs
    ):

    for _ in tqdm(range(num_epochs)):
        train_loss = 0
        num_batches = len(train_generator)
        model.train()
        for inputs, gt in tqdm(train_generator, total=num_batches):
            inputs, gt = from_numpy(inputs), from_numpy(gt)
            optimizer.zero_grad()
            loss = loss_fn(model(inputs), gt)
            loss.backward()
            optimizer.step()
            #train_loss += loss.item()
        #train_loss /= num_batches

        val_loss = 0
        num_batches = len(validation_generator)
        model.eval()
        for inputs, gt in tqdm(validation_generator, total=num_batches):
            inputs, gt = from_numpy(inputs), from_numpy(gt)
            outputs = model(inputs)
            #val_loss += loss_fn(outputs, gt).item()
        #val_loss /= num_batches

        tqdm.write(f'train loss {train_loss:.5f} - val loss {val_loss:.5f}')
