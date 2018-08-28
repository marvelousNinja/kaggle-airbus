from itertools import product

import numpy as np

def calculate_iou(masks, other_masks):
    ious = []
    for mask, other_mask in product(masks, other_masks):
        intersection = np.count_nonzero(np.logical_and(mask, other_mask))
        union = np.count_nonzero(np.logical_or(mask, other_mask))
        ious.append(intersection / union)
    return np.array(ious).reshape((len(masks), len(other_masks)))

def f2_score_with_threshold(threshold, ious):
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
    numerator = 5 * true_positives
    denominator = (5 * true_positives + 4 * false_negatives + false_positives)
    if denominator == 0: return 1.0
    return numerator / denominator

def f2_score(thresholds, pred_masks, gt_masks):
    scores = []
    for image_pred_masks, image_gt_masks in zip(pred_masks, gt_masks):
        ious = calculate_iou(image_pred_masks, image_gt_masks)
        threshold_scores = [f2_score_with_threshold(threshold, ious) for threshold in thresholds]
        image_score = np.mean(threshold_scores)
        scores.append(image_score)
    return np.mean(scores)
