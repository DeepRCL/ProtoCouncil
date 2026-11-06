"""Image encoders. Each returns a 16x16 token grid [B, D, 16, 16] for a 256x256 input image.

* ``medsam``: MedSAM ViT-B (SAM image encoder with its 256-channel neck) with bottleneck adapters after every
  attention and MLP block. Only the adapters are trained; all MedSAM weights stay frozen.
* ``dinov3``: DINOv3 ViT-L/16 (normalized patch tokens, 1024 channels). Fully fine-tuned.
* ``vjepa``: V-JEPA 2.1 ViT-L/16 image path (1024 channels, interpolated RoPE at 256 px). Fully fine-tuned.
"""
import os
import sys

import torch
from torch import nn
from torch.nn import functional as F

from . import sam

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


# ----------------------------------------------------------------------------- MedSAM with adapters
class Adapter(nn.Module):
    def __init__(self, dim, adapter_dim=256, init_scale=1e-3):
        super().__init__()
        self.down_project = nn.Linear(dim, adapter_dim)
        self.up_project = nn.Linear(adapter_dim, dim)
        self.act = nn.GELU()
        nn.init.uniform_(self.down_project.weight, -init_scale, init_scale)
        nn.init.uniform_(self.up_project.weight, -init_scale, init_scale)
        nn.init.zeros_(self.down_project.bias)
        nn.init.zeros_(self.up_project.bias)

    def forward(self, x):
        return self.up_project(self.act(self.down_project(x))) + x


class AdapterAttn(nn.Module):
    """Attention sub-block followed by a residual adapter (attribute names match ProstNFound+ / Prost-RL)."""

    def __init__(self, attn, dim, adapter_dim=256):
        super().__init__()
        self.attn = attn
        self.adapter = Adapter(dim, adapter_dim)

    def forward(self, x):
        return self.adapter(self.attn(x))


class AdapterMLPBlock(nn.Module):
    """MLP sub-block followed by a residual adapter."""

    def __init__(self, mlp, dim, adapter_dim=256):
        super().__init__()
        self.mlp = mlp
        self.adapter = Adapter(dim, adapter_dim)

    def forward(self, x):
        return self.adapter(self.mlp(x))


class MedSAMEncoder(nn.Module):
    out_dim = 256
    mean, std = (0., 0., 0.), (1., 1., 1.)

    def __init__(self, medsam_checkpoint=None, adapter_dim=256):
        super().__init__()
        enc = sam.image_encoder_vit_b()
        if medsam_checkpoint is not None:
            msg = enc.load_state_dict(sam.substate(sam.load_sam_state(medsam_checkpoint), 'image_encoder.'), strict=True)
            assert not msg.missing_keys and not msg.unexpected_keys, msg
        for block in enc.blocks:
            dim = block.attn.proj.in_features
            block.attn = AdapterAttn(block.attn, dim, adapter_dim)
            block.mlp = AdapterMLPBlock(block.mlp, dim, adapter_dim)
        for name, p in enc.named_parameters():
            p.requires_grad_('adapter' in name)
        self.encoder = enc

    def forward(self, x):
        return self.encoder(x)


# ----------------------------------------------------------------------------- DINOv3
def _import_dinov3():
    try:
        import dinov3.hub.backbones  # noqa: F401
    except ImportError:
        repo = os.environ.get('DINOV3_REPO')
        if repo is None:
            raise ImportError('DINOv3 is not installed. Install https://github.com/facebookresearch/dinov3 '
                              '(pip install -e <clone>) or set DINOV3_REPO to a local clone.')
        sys.path.append(repo)
        import dinov3.hub.backbones  # noqa: F401
    import dinov3.hub.backbones as backbones
    return backbones


class DINOv3Encoder(nn.Module):
    out_dim = 1024
    mean, std = IMAGENET_MEAN, IMAGENET_STD

    def __init__(self, weights=None, arch='dinov3_vitl16'):
        super().__init__()
        backbones = _import_dinov3()
        self.backbone = backbones.__dict__[arch](pretrained=False)
        if weights is not None:
            state = torch.load(weights, map_location='cpu', weights_only=False)
            if isinstance(state, dict) and 'encoder_state_dict' in state:  # exported by protocouncil.adapt
                state = state['encoder_state_dict']
                state = {k[len('backbone.'):]: v for k, v in state.items() if k.startswith('backbone.')}
            msg = self.backbone.load_state_dict(state, strict=True)
            assert not msg.missing_keys and not msg.unexpected_keys, msg

    def forward(self, x):
        tokens = self.backbone.forward_features(x)['x_norm_patchtokens']
        b, n, c = tokens.shape
        h = int(n ** .5)
        return tokens.transpose(1, 2).reshape(b, c, h, h)


# ----------------------------------------------------------------------------- V-JEPA 2.1
# Parameters outside the single-frame image path (video patch embedding, video modality embedding and the
# hierarchical distillation norms). They never receive gradients and are kept frozen.
VJEPA_INACTIVE = ('encoder.patch_embed.', 'encoder.video_mod_embed', 'encoder.norms_block.0.',
                  'encoder.norms_block.1.', 'encoder.norms_block.2.')


class VJEPAEncoder(nn.Module):
    out_dim = 1024
    mean, std = IMAGENET_MEAN, IMAGENET_STD

    def __init__(self, weights=None, arch='vjepa2_1_vit_large_384', input_size=256):
        super().__init__()
        from .vjepa import vjepa2_1_encoder
        self.encoder = vjepa2_1_encoder(arch)
        self.input_size = int(input_size)
        if weights is not None:
            state = torch.load(weights, map_location='cpu', weights_only=False)
            if 'ema_encoder' in state:  # official V-JEPA 2.1 release checkpoint
                state = {k.replace('module.', '').replace('backbone.', ''): v for k, v in state['ema_encoder'].items()}
            elif 'encoder_state_dict' in state:  # exported by protocouncil.adapt
                state = {k[len('encoder.'):]: v for k, v in state['encoder_state_dict'].items() if k.startswith('encoder.')}
            msg = self.encoder.load_state_dict(state, strict=True)
            assert not msg.missing_keys and not msg.unexpected_keys, msg
        for name, p in self.named_parameters():
            if name.startswith(VJEPA_INACTIVE):
                p.requires_grad_(False)

    def forward(self, x):
        x = x.float()
        if tuple(x.shape[-2:]) != (self.input_size, self.input_size):
            x = F.interpolate(x, size=(self.input_size, self.input_size), mode='bilinear', align_corners=False)
        z = self.encoder(x.unsqueeze(2))  # single frame through the image patch embedding
        if isinstance(z, (list, tuple)):
            z = z[0]
        b, n, c = z.shape
        h = int(n ** .5)
        return z.transpose(1, 2).reshape(b, c, h, h)


ENCODERS = {'medsam': MedSAMEncoder, 'dinov3': DINOv3Encoder, 'vjepa': VJEPAEncoder}


def build_encoder(name, weights=None):
    """``weights``: MedSAM checkpoint for ``medsam``; official or adapted backbone weights for ``dinov3`` / ``vjepa``."""
    if name == 'medsam':
        return MedSAMEncoder(medsam_checkpoint=weights)
    return ENCODERS[name](weights=weights)
