import torch
from tabulate import tabulate
from tqdm import tqdm

from airbus.utils import from_numpy

def fit_model(
        model,
        train_generator,
        validation_generator,
        optimizer,
        loss_fn,
        num_epochs,
        logger,
        callbacks=[],
        metrics=[]
    ):

    for epoch in tqdm(range(num_epochs)):
        num_batches = len(train_generator)
        logs = {}
        logs['train_loss'] = 0
        for func in metrics: logs[f'train_{func.__name__}'] = 0
        model.train()
        torch.set_grad_enabled(True)
        for callback in callbacks: callback.on_train_begin(logs)
        for batch in tqdm(train_generator, total=num_batches):
            batch = from_numpy(batch)
            optimizer.zero_grad()
            outputs = model(batch)
            loss = loss_fn(outputs, batch)
            loss.backward()
            optimizer.step()
            logs['train_loss'] += loss.data[0]
            logs['batch_loss'] = loss.data[0]
            for func in metrics: logs[f'train_{func.__name__}'] += func(outputs, batch)
            for callback in callbacks: callback.on_train_batch_end(logs, outputs, batch)

        logs.pop('batch_loss')
        logs['train_loss'] /= num_batches
        for func in metrics: logs[f'train_{func.__name__}'] /= num_batches

        logs['val_loss'] = 0
        for func in metrics: logs[f'val_{func.__name__}'] = 0
        num_batches = len(validation_generator)
        model.eval()
        torch.set_grad_enabled(False)
        for batch in tqdm(validation_generator, total=num_batches):
            batch = from_numpy(batch)
            outputs = model(batch)
            logs['val_loss'] += loss_fn(outputs, batch).data[0]
            for func in metrics: logs[f'val_{func.__name__}'] += func(outputs, batch)
            for callback in callbacks: callback.on_validation_batch_end(logs, outputs, batch)
        logs['val_loss'] /= num_batches
        for func in metrics: logs[f'val_{func.__name__}'] /= num_batches
        for callback in callbacks: callback.on_validation_end(logs)

        epoch_rows = [['epoch', epoch]]
        for name, value in logs.items():
            epoch_rows.append([name, f'{value:.3f}'])

        logger(tabulate(epoch_rows))
