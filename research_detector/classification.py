"""Shared label and probability helpers for image-level classification (three-class Prost-RL convention)."""

from __future__ import annotations

import torch
from torch.nn import functional as F


def grade_group_to_three_class(grade_group: torch.Tensor) -> torch.Tensor:
    """Map grade groups to benign, non-csPCa cancer, and csPCa classes."""
    grade_group = torch.as_tensor(grade_group)
    if grade_group.is_floating_point():
        grade_group = torch.nan_to_num(grade_group, nan=0.0)

    labels = torch.zeros_like(grade_group, dtype=torch.long)
    labels = torch.where(grade_group > 0, torch.ones_like(labels), labels)
    labels = torch.where(grade_group > 2, torch.full_like(labels, 2), labels)
    return labels


def image_classification_probabilities(
    logits: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return P(PCA) and P(csPCa) for binary or three-class logits."""
    num_classes = logits.shape[-1]
    probabilities = F.softmax(logits, dim=-1)

    if num_classes == 2:
        positive_probability = probabilities[..., 1]
        return positive_probability, positive_probability
    if num_classes == 3:
        pca_probability = probabilities[..., 1] + probabilities[..., 2]
        cspca_probability = probabilities[..., 2]
        return pca_probability, cspca_probability

    raise ValueError(
        f"Expected 2 or 3 image-level classes, got {num_classes}."
    )
