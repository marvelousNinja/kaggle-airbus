import torch

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
