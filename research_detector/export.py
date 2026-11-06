"""Export a trained research detector so that ``protocouncil.extract`` can read its token grids.

    python -m research_detector.export --checkpoint runs/research/dinov3/fold0/best_interpretable.pth \
        --out runs/detectors/dinov3/fold0/ctx.pth

Token grids, as used for the paper's DINOv3 and V-JEPA 2.1 branches:
    enc  image-encoder output [B, 256 tokens, 1024]
    ctx  image tokens of the two prompt-conditioned class decoders (all-cancer and conditional csPCa), first call,
         concatenated [B, 256 tokens, 512]
For DINOv3 the paper read ctx from the heatmap-weight-1.0 detector (configs/dinov3.yaml) and enc from the
heatmap-weight-0.2 detector (configs/dinov3_heatmap02.yaml); for V-JEPA 2.1 both come from one detector
(configs/vjepa.yaml). Use ``protocouncil.extract --detector <ctx export> --enc-detector <enc export>``.
"""
import argparse
import tempfile
from pathlib import Path

import torch
from omegaconf import OmegaConf
from torch import nn

from protocouncil.import_prostrl import PROSTRL_PROMPT_STATS
from protocouncil.models.detector import PROMPTS

ENCODER_NAMES = {'dinov3_backbone_wrapper_for_feature_maps': 'dinov3', 'vjepa2_backbone_wrapper_for_feature_maps': 'vjepa'}


class ResearchDetector(nn.Module):
    """Research detector with the interface of ``protocouncil.models.detector.Detector`` used for extraction."""

    def __init__(self, model):
        super().__init__()
        self.model = model
        self.encoder = model.image_encoder

    def forward(self, image, prompts, return_tokens=False, prostate_mask=None, needle_mask=None):
        assert prostate_mask is not None and needle_mask is not None, 'the research detector needs both masks'
        # As in the research pipeline, cores without any needle-track pixel are not passed to the model; their
        # outputs are NaN (protocouncil.extract excludes them, as the prototype heads cannot score them either).
        keep = ((prostate_mask > .5) & (needle_mask > .5)).flatten(1).any(1)
        if not bool(keep.all()):
            res = {}
            if keep.any():
                part = self(image[keep], prompts[keep], return_tokens, prostate_mask[keep], needle_mask[keep])
                for k, v in part.items():
                    full = v.new_full((len(image),) + tuple(v.shape[1:]), float('nan'))
                    full[keep] = v
                    res[k] = full
            return res or self._nan_outputs(image, return_tokens)
        head = self.model.prost_rl_endpoint_head
        rec = {'pca': [], 'gg3': [], 'enc': []}
        hooks = [head.pca_decoder.transformer.register_forward_hook(lambda m, i, o: rec['pca'].append(o[1])),
                 head.cspca_decoder.transformer.register_forward_hook(lambda m, i, o: rec['gg3'].append(o[1])),
                 self.model.image_encoder.register_forward_hook(lambda m, i, o: rec['enc'].append(o))]
        try:
            out = self.model(image, prostate_mask, needle_mask, prompts)
        finally:
            for h in hooks:
                h.remove()
        res = dict(heatmap_logits=out['mask_logits'], class_logits=out['cls_outputs'][0])
        if return_tokens:
            res['enc'] = rec['enc'][0].float().flatten(2).transpose(1, 2)
            res['ctx'] = torch.cat([rec['pca'][0], rec['gg3'][0]], -1)
        return res

    def _nan_outputs(self, image, return_tokens):
        b = len(image)
        res = dict(heatmap_logits=_nan_like((b, 1, 64, 64), image), class_logits=_nan_like((b, 2), image))
        if return_tokens:
            res['enc'] = _nan_like((b, 256, self.model.image_encoder.out_dim), image)
            res['ctx'] = _nan_like((b, 256, 512), image)
        return res


def _nan_like(shape, ref):
    return ref.new_full(shape, float('nan'))


def build(config, state_dict=None):
    """Rebuild the research model from its training config (backbone weights are restored from ``state_dict``)."""
    from .anchors import random_bank
    from .registry import create_model
    import numpy as np
    cfg = OmegaConf.create(config)
    with tempfile.TemporaryDirectory() as tmp:
        bank = Path(tmp) / 'bank.npz'  # placeholder of the right shape; the trained bank is part of the state dict
        np.savez_compressed(bank, **random_bank())
        kw = OmegaConf.to_container(cfg.model_kw, resolve=True)
        kw['backbone_kw']['backbone_cfg']['weights'] = None
        kw['semantic_anchor_bank_path'] = str(bank)
        model = create_model(cfg.model, **kw)
    if state_dict is not None:
        model.load_state_dict(state_dict, strict=True)
    return ResearchDetector(model.eval())


def load(ck):
    det = build(ck['config'], ck['state_dict'])
    return det


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--checkpoint', required=True, help='best_interpretable.pth written by research_detector.train')
    p.add_argument('--out', required=True)
    a = p.parse_args()
    ck = torch.load(a.checkpoint, map_location='cpu', weights_only=False)
    cfg = ck['args']
    load(dict(config=cfg, state_dict=ck['model']))  # check that the checkpoint rebuilds strictly
    data = cfg['data']
    assert data.get('prompt_stats', 'prost_rl') == 'prost_rl', 'export expects Prost-RL prompt scaling (data.prompt_stats: prost_rl)'
    out = Path(a.out); out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(dict(kind='research', config=cfg, state_dict=ck['model'], encoder=ENCODER_NAMES[cfg['model_kw']['backbone']],
                    prompts=PROMPTS, prompt_stats=PROSTRL_PROMPT_STATS, mean=tuple(data['mean']), std=tuple(data['std']),
                    image_size=int(data['image_size']), fold=int(data['fold']), n_folds=int(data['n_folds']),
                    exclude_benign_from_positive_patients=bool(data['exclude_benign_cores_from_positive_patients'])), out)
    print(f'{a.checkpoint} -> {out}')


if __name__ == '__main__':
    main()
