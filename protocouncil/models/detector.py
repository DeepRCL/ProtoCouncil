"""Prompt-conditioned cancer detector (ProstNFound+-style) with a swappable foundation-model encoder.

    image --encoder--> enc grid [B, D, 16, 16] --(1x1 conv if D != 256)--> image tokens [B, 256, 16, 16]
    clinical values --per-variable MLPs--> one sparse prompt token each
    mask decoder (SAM)  : image tokens + prompt tokens -> cancer heatmap logits [B, 1, 64, 64]
    class decoder (SAM) : image tokens + prompt tokens -> class logits [B, 3] (benign, GG1-2, GG>=3)
                          and the prompt-conditioned image tokens = ctx grid [B, 256 tokens, 256]

After training, the detector is frozen and its two token grids (enc and ctx) feed the anatomical prototype heads.
"""
import torch
from torch import nn
from torch.nn import functional as F

from . import sam
from .encoders import build_encoder

PROMPTS = ('age', 'psa', 'psad', 'base_apex', 'mid_lateral')
NUM_CLASSES = 3


class Detector(nn.Module):
    def __init__(self, encoder: nn.Module, prompts=PROMPTS, num_classes=NUM_CLASSES):
        super().__init__()
        self.encoder = encoder
        self.prompts = tuple(prompts)
        width = sam.EMBED_DIM
        if encoder.out_dim == width:
            self.neck = nn.Identity()
        else:
            self.neck = nn.Conv2d(encoder.out_dim, width, kernel_size=1)
            nn.init.orthogonal_(self.neck.weight)
            nn.init.zeros_(self.neck.bias)
        self.prompt_encoder = sam.prompt_encoder()
        self.mask_decoder = sam.mask_decoder()
        self.class_decoder = sam.class_decoder(num_classes)
        self.prompt_mlps = nn.ModuleDict({
            name: nn.Sequential(nn.Linear(1, 128), nn.ReLU(), nn.Linear(128, width)) for name in self.prompts})

    @property
    def mean(self):
        return self.encoder.mean

    @property
    def std(self):
        return self.encoder.std

    def load_sam_decoders(self, medsam_checkpoint):
        """Initialize the prompt encoder and mask decoder from the MedSAM release (class decoder stays random)."""
        state = sam.load_sam_state(medsam_checkpoint)
        for name in ('prompt_encoder', 'mask_decoder'):
            msg = getattr(self, name).load_state_dict(sam.substate(state, name + '.'), strict=True)
            assert not msg.missing_keys and not msg.unexpected_keys, (name, msg)

    def load_decoders_from(self, detector_state):
        """Copy prompt encoder, prompt MLPs, mask decoder and class decoder from another detector's state dict
        (e.g. the MedSAM / Prost-RL detector of the same fold). Tensors whose shape differs (the output layer of a
        class decoder with a different number of classes) keep their initialization; their names are returned."""
        keys = ('prompt_encoder.', 'prompt_mlps.', 'mask_decoder.', 'class_decoder.')
        own = self.state_dict()
        part = {k: v for k, v in detector_state.items() if k.startswith(keys) and k in own and v.shape == own[k].shape}
        skipped = sorted(k for k in own if k.startswith(keys) and k not in part)
        last = {f'class_decoder.output_cls_mlps.{i}.layers.{len(m.layers) - 1}.{w}'
                for i, m in enumerate(self.class_decoder.output_cls_mlps) for w in ('weight', 'bias')}
        bad = [k for k in skipped if k not in last or k not in detector_state]
        assert not bad, ('decoder tensors missing or incompatible (only the class output layer may differ)', bad)
        self.load_state_dict(part, strict=False)
        return skipped

    def encoder_parameters(self):
        return [p for p in self.encoder.parameters() if p.requires_grad]

    def head_parameters(self):
        enc = {id(p) for p in self.encoder.parameters()}
        return [p for p in self.parameters() if p.requires_grad and id(p) not in enc]

    def forward(self, image, prompts, return_tokens=False, prostate_mask=None, needle_mask=None):
        """image [B, 3, 256, 256]; prompts [B, len(self.prompts)] (normalized values). Masks are not used here
        (accepted for interface compatibility with the research detector)."""
        enc = self.encoder(image)
        x = self.neck(enc)
        b = len(x)
        sparse, dense = self.prompt_encoder(None, None, None)
        sparse = sparse.expand(b, -1, -1)
        if dense.shape[-2:] != x.shape[-2:]:
            dense = F.interpolate(dense, size=x.shape[-2:])
        dense = dense.expand(b, -1, -1, -1)
        pe = self.prompt_encoder.get_dense_pe()
        if pe.shape[-2:] != x.shape[-2:]:
            pe = F.interpolate(pe, size=x.shape[-2:])
        tokens = [self.prompt_mlps[name](prompts[:, i:i + 1].to(x.dtype))[:, None] for i, name in enumerate(self.prompts)]
        sparse = torch.cat([sparse.to(x.dtype)] + tokens, dim=1)
        heatmap, _ = self.mask_decoder(x, pe, sparse, dense, multimask_output=False)
        cls, ctx = self.class_decoder(x, pe, sparse, dense, return_tokens=True)
        out = dict(heatmap_logits=heatmap, class_logits=cls[0])
        if return_tokens:
            out['enc'] = enc.flatten(2).transpose(1, 2)  # [B, 256 tokens, D], row-major over the 16x16 grid
            out['ctx'] = ctx  # [B, 256 tokens, 256]
        return out


def build_detector(encoder, encoder_weights=None, medsam_checkpoint=None):
    """Fresh detector. ``encoder_weights``: MedSAM checkpoint (medsam) or adapted backbone (dinov3 / vjepa)."""
    det = Detector(build_encoder(encoder, encoder_weights))
    if medsam_checkpoint is not None:
        det.load_sam_decoders(medsam_checkpoint)
    return det


def load_detector(path, map_location='cpu'):
    """Rebuild a trained detector from a checkpoint written by ``protocouncil.train_detector``."""
    ck = torch.load(path, map_location=map_location, weights_only=False)
    if ck.get('kind') == 'research':  # exported by research_detector.export (exact paper recipe)
        from research_detector.export import load
        return load(ck), ck
    det = Detector(build_encoder(ck['encoder']), prompts=ck['prompts'], num_classes=ck.get('num_classes', NUM_CLASSES))
    det.load_state_dict(ck['state_dict'], strict=True)
    return det, ck
