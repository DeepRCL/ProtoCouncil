"""Weak supervision for the detectors.

The heatmap is supervised only inside the needle track (needle trace AND prostate): the mean predicted cancer
probability of the track should match the core's pathology involvement (proportion BCE, ProstNFound+), with a
small entropy penalty. The class decoder is trained with three-class cross-entropy (benign / GG1-2 / GG>=3).
"""
import math

import torch
from torch.nn import functional as F


def proportion_bce(logits, target, pos_weight=1., entropy_weight=.01):
    """BCE between the mean sigmoid of a bag of logits and a target proportion, computed stably in log space (FP32)."""
    z = logits.float().reshape(-1)
    t = target.float()
    log_n = math.log(z.numel())
    log_p = torch.logsumexp(F.logsigmoid(z), 0) - log_n
    log_q = torch.logsumexp(F.logsigmoid(-z), 0) - log_n  # log(1 - mean sigmoid(z))
    loss = -(t * log_p + (1 - t) * log_q)
    if float(t) > 0:
        loss = loss * pos_weight
    if entropy_weight:
        p = torch.sigmoid(z)
        loss = loss + entropy_weight * (F.softplus(z) - z * p).mean()
    return loss


def involvement_loss(heatmap_logits, prostate_mask, needle_mask, involvement, pos_weight=1., entropy_weight=.01):
    """Sum over cores of proportion BCE inside the needle track. Cores with an empty track or unknown involvement
    contribute nothing."""
    region = (prostate_mask > .5) & (needle_mask > .5)
    total = heatmap_logits.new_zeros((), dtype=torch.float32)
    for i in range(len(heatmap_logits)):
        if not region[i].any() or not torch.isfinite(involvement[i]):
            continue
        total = total + proportion_bce(heatmap_logits[i][region[i]], involvement[i], pos_weight, entropy_weight)
    return total


def balanced_cross_entropy(logits, target, num_classes=3):
    """Cross-entropy with per-batch inverse-frequency class weights n / (K * n_c)."""
    counts = torch.bincount(target, minlength=num_classes).float()
    weight = torch.where(counts > 0, len(target) / (num_classes * counts.clamp_min(1)), torch.ones_like(counts))
    return F.cross_entropy(logits.float(), target, weight=weight)


def needle_mean_probability(heatmap_logits, prostate_mask, needle_mask):
    """Mean heatmap probability inside the needle track per core (NaN when the track is empty)."""
    region = ((prostate_mask > .5) & (needle_mask > .5)).flatten(1).float()
    p = torch.sigmoid(heatmap_logits.float()).flatten(1)
    n = region.sum(1)
    return torch.where(n > 0, (p * region).sum(1) / n.clamp_min(1), torch.full_like(n, float('nan')))
