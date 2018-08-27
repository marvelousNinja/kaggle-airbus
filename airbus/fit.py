import io
from functools import partial

import torch
import telegram_send
import numpy as np
import matplotlib; matplotlib.use('agg')
import matplotlib.pyplot as plt
from fire import Fire
from tqdm import tqdm

from airbus.cyclic_lr import CyclicLR
from airbus.generators import get_train_generator
from airbus.generators import get_validation_generator
from airbus.linknet import Linknet
from airbus.model_checkpoint import ModelCheckpoint
from airbus.model_checkpoint import load_checkpoint
from airbus.training import fit_model
from airbus.utils import as_cuda
from airbus.utils import confusion_matrix

def jaccard_loss(logits, labels):
    smooth = 1e-12
    probs = torch.nn.functional.softmax(logits, dim=1)
    probs = probs[:, 1, :, :]
    labels = labels.float()
    intersection = (probs * labels).sum((1, 2))
    union = probs.sum((1, 2)) + labels.sum((1, 2)) - intersection
    return (1 - (intersection + smooth) / (union + smooth)).mean()

def dice_loss(logits, labels):
    probs = torch.nn.functional.softmax(logits, dim=1)
    probs = probs[:, 1, :, :]
    labels = labels.float()
    intersection = (probs * labels).sum((1, 2))
    pred_volume = probs.sum((1, 2))
    true_volume = labels.sum((1, 2))
    return (1 - 2 * intersection / (pred_volume + true_volume + 1.0)).mean()

# https://arxiv.org/pdf/1707.03237.pdf
def generalized_dice_loss(logits, labels):
    labels = labels.long()
    probs = torch.nn.functional.softmax(logits, dim=1)
    numerator = 0
    denominator = 0
    for label in [0, 1]:
        pred_probs = probs[:, label, :, :]
        true_labels = (labels == label).float()
        weight = (true_labels.sum((1, 2)) + 1) ** (-2)
        numerator += weight * (pred_probs * true_labels).sum((1, 2))
        denominator += weight * (pred_probs.sum((1, 2)) + true_labels.sum((1, 2)) + 1)
    return (1 - 2 * numerator / denominator).mean()

# https://arxiv.org/pdf/1803.11078.pdf
def asymmetric_similarity_loss(logits, labels):
    beta = 1.5
    probs = torch.nn.functional.softmax(logits, dim=1)[:, 1]
    intersection = (probs * labels).sum((1, 2))
    false_positives = ((1 - labels) * probs).sum((1, 2)) * (beta ** 2) / (1 + beta ** 2)
    false_negatives = ((1 - probs) * labels).sum((1, 2)) / (1 + beta ** 2)
    return (1 - intersection / (intersection + false_negatives + false_positives)).mean()

def compute_loss(logits, labels):
    return jaccard_loss(logits, labels) * 0.5 + torch.nn.functional.cross_entropy(logits, labels.long()) * 0.5

def send_telegram_message(message):
    telegram_send.send(conf='./telegram.conf', messages=[f'`{message}`'], parse_mode='markdown')

def send_telegram_figure(figure):
    buf = io.BytesIO()
    figure.savefig(buf, format='png')
    buf.seek(0)
    telegram_send.send(conf='./telegram.conf', images=[buf])
    plt.close()

def plot_figure(figure):
    plt.show()
    plt.close()

def visualize_losses(image_logger, outputs, gt):
    num_samples = min(len(gt), 8)
    outputs = outputs[:num_samples]
    outputs -= np.expand_dims(np.max(outputs, axis=1), axis=1)
    gt = gt[:num_samples]

    probs = (np.exp(outputs) / np.expand_dims(np.sum(np.exp(outputs), axis=1), axis=1))[:, 1, :, :]
    beta = 1.5
    intersection = (probs * gt).sum((1, 2))
    false_positives = ((1 - gt) * probs).sum((1, 2)) * (beta ** 2) / (1 + beta ** 2)
    false_negatives = ((1 - probs) * gt).sum((1, 2)) / (1 + beta ** 2)
    losses = (1 - intersection / (intersection + false_negatives + false_positives))

    num_samples = min(len(gt), 8)
    plt.figure(figsize=(10, 5))
    for i in range(num_samples):
        plt.subplot(2, num_samples, i + 1)
        plt.gca().set_title(f'L {losses[i]:.2f}\nI {intersection[i]:.2f}\nFP {false_positives[i]:.2f}\nFN {false_negatives[i]:.2f}')
        plt.imshow(probs[i])
        plt.subplot(2, num_samples, num_samples + i + 1)
        plt.imshow(gt[i])
    plt.gcf().tight_layout()
    plt.subplots_adjust(hspace=0.1, wspace=0.1)
    image_logger(plt.gcf())

def visualize_learning_curve(image_logger, train_losses, val_losses):
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss')
    plt.plot(np.array(train_losses) - np.array(val_losses), label='Generalization Error')
    plt.legend()
    image_logger(plt.gcf())

def on_validation_end(history, visualize, image_logger, logger, model_checkpoint, train_loss, val_loss, outputs, gt):
    if visualize:
        history.setdefault('train_losses', []).append(train_loss)
        history.setdefault('val_losses', []).append(val_loss)
        visualize_losses(image_logger, outputs, gt)
        visualize_learning_curve(image_logger, history['train_losses'], history['val_losses'])
    logger(confusion_matrix(np.argmax(outputs, axis=1), gt, [0, 1]))
    model_checkpoint.step(val_loss)

def fit(num_epochs=100, limit=None, batch_size=16, lr=.001, checkpoint_path=None, telegram=False, visualize=False):
    torch.backends.cudnn.benchmark = True
    np.random.seed(1991)

    if telegram:
        logger = send_telegram_message
        image_logger = send_telegram_figure
    else:
        logger = tqdm.write
        image_logger = plot_figure

    if checkpoint_path:
        model = load_checkpoint(checkpoint_path)
    else:
        model = Linknet(2)

    model = as_cuda(model)
    optimizer = torch.optim.Adam(filter(lambda param: param.requires_grad, model.parameters()), lr)
    model_checkpoint = ModelCheckpoint(model, 'linknet', logger)
    train_generator = get_train_generator(batch_size, limit)
    cyclic_lr = CyclicLR(cycle_iterations=len(train_generator) * 2, min_lr=0.0001, max_lr=0.005, optimizer=optimizer, logger=logger)

    fit_model(
        model=model,
        train_generator=train_generator,
        # TODO AS: Colab explodes with out of memory
        validation_generator=get_validation_generator(batch_size, 160),
        optimizer=optimizer,
        loss_fn=compute_loss,
        num_epochs=num_epochs,
        on_validation_end=partial(on_validation_end, {}, visualize, image_logger, logger, model_checkpoint),
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
