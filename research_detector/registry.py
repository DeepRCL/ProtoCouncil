"""Support code for the research detector: model registry, encoder builders and small utilities.

The research code expects a model registry (``create_model``) and a few helpers from the authors' internal library.
They are reproduced here; the DINOv3 / V-JEPA 2.1 backbones are the ProtoCouncil encoders, which have the same
parameter names and outputs as the research wrappers.
"""
import itertools
import logging
import random
from typing import Sequence

import numpy as np
import pandas as pd
import torch
from einops import rearrange, repeat
from torch import nn

try:  # optional: only used when ``wandb_enabled: true``
    import wandb
except ImportError:  # pragma: no cover
    wandb = None

_MODELS = {}


def register_model(fn):
    _MODELS[fn.__name__] = fn
    return fn


def create_model(name, **kwargs):
    if name not in _MODELS:
        from . import model  # noqa: F401  (registers protoprostate_medsam)
    if name not in _MODELS:
        raise KeyError(f'unknown model {name}; available: {sorted(_MODELS)}')
    return _MODELS[name](**kwargs)


@register_model
def dinov3_backbone_wrapper_for_feature_maps(*, backbone_cfg: dict, **_unused):
    from protocouncil.models.encoders import DINOv3Encoder
    return DINOv3Encoder(weights=backbone_cfg.get('weights'), arch=backbone_cfg.get('name', 'dinov3_vitl16'))


@register_model
def vjepa2_backbone_wrapper_for_feature_maps(*, backbone_cfg: dict, input_size: int = 256, **_unused):
    from protocouncil.models.encoders import VJEPAEncoder
    return VJEPAEncoder(weights=backbone_cfg.get('weights'), input_size=input_size)


def set_global_seed(seed):
    logging.info(f'Setting global seed to {seed}')
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class MaskedPredictionModule(nn.Module):
    """Logits inside a mask, flattened, with the index of the core they belong to."""

    def forward(self, heatmap_logits, mask):
        B, C, H, W = heatmap_logits.shape
        assert mask.shape == (B, 1, H, W), f'Expected mask shape to be {(B, 1, H, W)}, got {mask.shape} instead.'
        core_idx = repeat(torch.arange(B, device=heatmap_logits.device), 'b -> b h w', h=H, w=W)
        core_idx_flattened = rearrange(core_idx, 'b h w -> (b h w)')
        mask_flattened = rearrange(mask, 'b c h w -> (b h w) c')[..., 0]
        logits_flattened = rearrange(heatmap_logits, 'b c h w -> (b h w) c', h=H, w=W)
        return logits_flattened[mask_flattened], core_idx_flattened[mask_flattened]


def get_bags_of_predictions(predictions_map, *masks):
    """Per-core bags of logits inside the intersection of the given masks."""
    masks = [mask.to(predictions_map.device) for mask in masks]
    _, _, h, w = predictions_map.shape
    masks = [nn.functional.interpolate(mask, size=(h, w), mode='nearest') for mask in masks]
    final_masks = []
    for i in range(len(predictions_map)):
        final_mask = torch.ones(predictions_map[i].shape, device=predictions_map[i].device).bool()
        for mask in masks:
            final_mask &= (mask[i] > 0.5)
        final_masks.append(final_mask)
    predictions, batch_idx = MaskedPredictionModule()(predictions_map, torch.stack(final_masks))
    return [predictions[batch_idx == i] for i in range(len(predictions_map))]


class DataFrameCollector:
    """Concatenates dicts of per-batch values and returns a DataFrame (2-D arrays are split into columns)."""

    def __init__(self):
        self.reset()

    def __call__(self, data_dict):
        for k, v in data_dict.items():
            if isinstance(v, torch.Tensor):
                v = v.detach().cpu()
            elif isinstance(v, np.ndarray):
                pass
            elif not isinstance(v, Sequence):
                v = [v]
            self._data.setdefault(k, []).append(v)

    def update(self, data_dict):
        self(data_dict)

    def compute(self):
        out = {}
        for k, v in self._data.items():
            out[k] = torch.concat(v) if isinstance(v[0], torch.Tensor) else list(itertools.chain(*v))
        out_new = {}
        for k, v in out.items():
            if isinstance(v, list):
                v = np.array(v)
            if isinstance(v, torch.Tensor):
                v = v.detach().cpu().numpy()
            if isinstance(v, np.ndarray) and v.ndim == 2:
                for i in range(v.shape[1]):
                    out_new[f'{k}_{i}'] = v[:, i]
            else:
                out_new[k] = v
        return pd.DataFrame(out_new)

    def reset(self):
        self._data = {}
