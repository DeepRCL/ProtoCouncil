"""Differentiable pooling along the biopsy-needle longitudinal axis."""

from __future__ import annotations

import torch


def pool_logits_along_needle_axis(
    logits: torch.Tensor,
    valid_mask: torch.Tensor,
    num_bins: int = 32,
) -> torch.Tensor:
    """Pool a 2-D valid-region heatmap into ordered longitudinal needle bins.

    Pathology involvement is a fraction of biopsy-core *length*, not a fraction
    of the rendered two-dimensional needle-mask area.  The mask geometry is
    non-learned, so PCA is used only to obtain a stable longitudinal coordinate;
    gradients flow through the mean logit pooled in every occupied bin.
    """
    if num_bins < 2:
        raise ValueError("num_bins must be at least 2.")
    if logits.ndim == 3 and logits.shape[0] == 1:
        logits = logits[0]
    if valid_mask.ndim == 3 and valid_mask.shape[0] == 1:
        valid_mask = valid_mask[0]
    if logits.ndim != 2 or valid_mask.ndim != 2:
        raise ValueError("logits and valid_mask must describe one 2-D core.")

    mask = valid_mask.bool()
    coordinates = mask.nonzero(as_tuple=False)
    values = logits[mask]
    if values.numel() == 0:
        return logits.new_empty((0,))
    if values.numel() == 1:
        return values

    # Geometry is deliberately computed in float32 because eigh is not
    # implemented for half precision and does not need prediction gradients.
    # Explicitly leave the surrounding CUDA autocast context. Merely calling
    # ``float()`` is insufficient because autocast may cast the matrix
    # operation back to FP16, while CUDA eigh has no half-precision kernel.
    with torch.autocast(device_type=coordinates.device.type, enabled=False):
        geometry = coordinates.detach().to(dtype=torch.float32)
        centered = geometry - geometry.mean(0, keepdim=True)
        covariance = centered.T @ centered / max(len(geometry) - 1, 1)
        _, eigenvectors = torch.linalg.eigh(covariance.float())
        longitudinal = centered @ eigenvectors[:, -1]
    span = longitudinal.max() - longitudinal.min()
    if float(span) <= 1e-6:
        return values.mean().reshape(1)

    normalized = (longitudinal - longitudinal.min()) / span
    bin_indices = torch.clamp((normalized * num_bins).long(), max=num_bins - 1)
    sums = values.new_zeros(num_bins).scatter_add_(0, bin_indices, values)
    counts = values.new_zeros(num_bins).scatter_add_(
        0, bin_indices, torch.ones_like(values)
    )
    occupied = counts > 0
    return sums[occupied] / counts[occupied]
