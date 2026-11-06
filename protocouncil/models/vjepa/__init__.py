"""Minimal V-JEPA 2.1 image encoder, vendored from https://github.com/facebookresearch/vjepa2 (MIT License, see LICENSE).

Upstream commit 204698b45b3712590f06245fbfba32d3be539812 (app/vjepa_2_1/models). The only change is that the
package-absolute imports in ``vision_transformer.py`` are relative.
"""
from .vision_transformer import vit_base, vit_large  # noqa: F401

UPSTREAM_COMMIT = '204698b45b3712590f06245fbfba32d3be539812'

# Encoder keyword arguments of upstream ``src.hub.backbones._make_vjepa2_1_model`` for the 384-px release models.
# Upstream passes the misspelled ``use_SiLU`` / ``wide_SiLU`` keys, which the ViT swallows through ``**kwargs``;
# they are reproduced verbatim so that construction is identical.
ENCODER_KWARGS = dict(patch_size=16, img_size=(384, 384), num_frames=64, tubelet_size=2, use_sdpa=True,
                      use_SiLU=False, wide_SiLU=True, uniform_power=False, use_rope=True,
                      img_temporal_dim_size=1, interpolate_rope=True)
ARCHITECTURES = {'vjepa2_1_vit_base_384': vit_base, 'vjepa2_1_vit_large_384': vit_large}


def vjepa2_1_encoder(arch='vjepa2_1_vit_large_384'):
    return ARCHITECTURES[arch](**ENCODER_KWARGS)
