from itertools import product

import numpy as np
import torch

from airbus.utils import extract_instance_masks_from_binary_mask
from airbus.utils import extract_instance_masks_from_labelled_mask
from airbus.utils import to_numpy

def mean_iou(outputs, gt, average=True):
    smooth = 1e-12
    preds = torch.sigmoid(outputs).round().long()
    pred_masks = preds[:, 0, :, :]
    true_masks = gt.long()
    intersection = (pred_masks & true_masks).sum(dim=(1, 2)).float()
    union = (pred_masks | true_masks).sum(dim=(1, 2)).float()
    values = ((intersection + smooth) / (union + smooth))
    return values.mean() if average else values

def calculate_iou(masks, other_masks):
    ious = []
    for mask, other_mask in product(masks, other_masks):
        intersection = np.count_nonzero(np.logical_and(mask, other_mask))
        union = np.count_nonzero(np.logical_or(mask, other_mask))
        ious.append(intersection / union)
    return np.array(ious).reshape((len(masks), len(other_masks)))

def f_score_with_threshold(beta, threshold, ious):
    # gt mask paired with predicted mask with IoU > threshold
    true_positives = 0
    for _ in range(ious.shape[0]):
        if ious.shape[1] == 0: continue

        gt_index, pred_index = np.unravel_index(ious.argmax(), ious.shape)
        max_iou = ious[gt_index, pred_index]
        if max_iou > threshold:
            true_positives += 1
            ious = np.delete(ious, gt_index, axis=0)
            ious = np.delete(ious, pred_index, axis=1)
            continue

    # leftover predicted masks
    false_positives = ious.shape[1]

    # gt mask not associated with pred mask
    false_negatives = ious.shape[0]
    numerator = (1 + beta ** 2) * true_positives
    denominator = ((1 + beta ** 2) * true_positives + (beta ** 2) * false_negatives + false_positives)
    if denominator == 0: return 1.0
    return numerator / denominator

def f_score(beta, thresholds, pred_masks, gt_masks):
    scores = []
    for image_pred_masks, image_gt_masks in zip(pred_masks, gt_masks):
        ious = calculate_iou(image_pred_masks, image_gt_masks)
        threshold_scores = [f_score_with_threshold(beta, threshold, ious) for threshold in thresholds]
        image_score = np.mean(threshold_scores)
        scores.append(image_score)
    return np.mean(scores)

def f2_score(outputs, gt):
    pred_masks = to_numpy(torch.sigmoid(outputs).round().long()[:, 0, :, :])
    pred_instance_masks = list(map(extract_instance_masks_from_binary_mask, pred_masks))
    gt_masks = list(map(lambda sample_gt: extract_instance_masks_from_labelled_mask(to_numpy(sample_gt)), gt))
    return f_score(2, [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95], pred_masks, gt_masks)
