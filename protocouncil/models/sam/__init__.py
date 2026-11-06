"""Minimal Segment Anything (ViT-B) components used by the MedSAM encoder and the detector decoders.

Vendored from https://github.com/facebookresearch/segment-anything (Apache License 2.0, see LICENSE).
"""
from functools import partial

import torch

from .image_encoder import ImageEncoderViT
from .mask_decoder import ClassDecoder, MaskDecoder
from .prompt_encoder import PromptEncoder
from .transformer import TwoWayTransformer

EMBED_DIM = 256


def image_encoder_vit_b() -> ImageEncoderViT:
    return ImageEncoderViT(depth=12, embed_dim=768, img_size=1024, mlp_ratio=4,
                           norm_layer=partial(torch.nn.LayerNorm, eps=1e-6), num_heads=12, patch_size=16,
                           qkv_bias=True, use_rel_pos=True, global_attn_indexes=(2, 5, 8, 11), window_size=14,
                           out_chans=EMBED_DIM)


def prompt_encoder() -> PromptEncoder:
    return PromptEncoder(embed_dim=EMBED_DIM, image_embedding_size=(64, 64), input_image_size=(1024, 1024),
                         mask_in_chans=16)


def two_way_transformer() -> TwoWayTransformer:
    return TwoWayTransformer(depth=2, embedding_dim=EMBED_DIM, mlp_dim=2048, num_heads=8)


def mask_decoder() -> MaskDecoder:
    return MaskDecoder(num_multimask_outputs=3, transformer=two_way_transformer(), transformer_dim=EMBED_DIM,
                       iou_head_depth=3, iou_head_hidden_dim=256)


def class_decoder(num_classes: int) -> ClassDecoder:
    return ClassDecoder(transformer_dim=EMBED_DIM, transformer=two_way_transformer(), num_cls_tokens=1,
                        cls_output_dims=[num_classes])


def load_sam_state(path):
    """Official SAM / MedSAM ViT-B checkpoint as a flat state dict (keys image_encoder.*, prompt_encoder.*, mask_decoder.*)."""
    state = torch.load(path, map_location='cpu', weights_only=False)
    if isinstance(state, dict) and 'model' in state and isinstance(state['model'], dict):
        state = state['model']
    return state


def substate(state, prefix):
    return {k[len(prefix):]: v for k, v in state.items() if k.startswith(prefix)}
