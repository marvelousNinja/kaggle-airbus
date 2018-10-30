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

def flat_lovasz_hinge_grad(gt_sorted):
    gts = gt_sorted.sum()
    intersection = gts - gt_sorted.float().cumsum(0)
    union = gts + (1 - gt_sorted).float().cumsum(0)
    jaccard = 1. - intersection / union
    jaccard[1:] = jaccard[1:] - jaccard[0:-1]
    return jaccard

def flat_lovasz_hinge_loss(logits, labels):
    signs = 2. * labels.float() - 1.
    errors = 1 - signs * logits
    errors_sorted, indicies = torch.sort(errors, dim=0, descending=True)
    labels_sorted = labels[indicies]
    grad = flat_lovasz_hinge_grad(labels_sorted)
    return torch.dot(torch.nn.functional.elu(errors_sorted) + 1, grad)

def lovasz_hinge_loss(logits, labels, average=True):
    losses = []
    for sample_logits, sample_labels in zip(logits, labels):
        losses.append(flat_lovasz_hinge_loss(sample_logits.view(-1), sample_labels.view(-1)))
    if average: return sum(losses) / len(losses)
    return torch.cat(losses)