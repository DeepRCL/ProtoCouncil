"""Projection-aware ProtoViT training on the Prost-RL weak-label pipeline.

There is no reinforcement learning or test-time adaptation here. The primary
task is involvement-proportion learning inside the strict intersection of the
prostate and needle masks. Binary cancer and csPCa classifiers are auxiliary
prototype heads and never replace the weak spatial objective.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
from pathlib import Path
from typing import Any

import numpy as np
import torch
from .registry import wandb
from omegaconf import DictConfig, OmegaConf
from sklearn.metrics import balanced_accuracy_score, cohen_kappa_score, roc_auc_score
from torch.nn import functional as F
from tqdm import tqdm

from .registry import create_model
from .model import ProtoProstate
from .registry import set_global_seed
from .evaluator import CancerLogitsHeatmapsEvaluator
from .data import get_dataloaders
from .losses import build_loss
from .prototype_utils import (
    clinical_prompt_tensor,
    filter_invalid_region_samples,
    forward_batch,
    hyperbolic_needle_regularization,
    json_safe_metrics,
    needle_axis_prototype_regularization,
    pathology_prototype_transport_regularization,
    project_prototypes,
    prototype_regularization,
    prototype_target_labels,
)
from .ordinal_domain import ordinal_domain_regularization
from .pathology_manifold import pathology_manifold_regularization
from .semantic_anchor import semantic_anchor_regularization
from .metrics import calculate_metrics


_PATHOLOGY_GEOMETRY_CACHE: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
_PATHOLOGY_GRADE_TARGET_CACHE: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}


def _pathology_grade_distribution_regularization(
    model: ProtoProstate,
    data: dict[str, Any],
    cfg: DictConfig,
    stage: str,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Distill training-only PANDA grade uncertainty into the existing grade cone.

    Unlike V11--V14, this introduces no cross-modal projector and does not force
    ultrasound features to imitate pathology features. It supervises logits that
    already participate in inference, using only a fixed grade-conditional table.
    """
    weight = float(cfg.get("pathology_grade_distill_weight", 0.0))
    logits = data.get("hyperbolic_grade_logits")
    if weight <= 0 or logits is None or stage != "joint":
        zero = data["prototype_activations"].new_zeros(())
        return zero, {}
    path = str(cfg.pathology_grade_target_path)
    if path not in _PATHOLOGY_GRADE_TARGET_CACHE:
        archive = np.load(path, allow_pickle=False)
        grades = torch.as_tensor(archive["grades"], dtype=torch.long)
        targets = torch.as_tensor(archive["targets"], dtype=torch.float)
        if targets.shape != (len(grades), len(grades)):
            raise ValueError(
                f"Pathology grade target must be square, got {targets.shape}."
            )
        _PATHOLOGY_GRADE_TARGET_CACHE[path] = (grades, targets)
        logging.info(
            "Loaded pathology grade-distribution target %s with grades=%s.",
            path,
            grades.tolist(),
        )
    teacher_grades, teacher_targets = _PATHOLOGY_GRADE_TARGET_CACHE[path]
    target_grades = torch.as_tensor(
        model.hyperbolic_needle_encoder.target_values,
        dtype=torch.long,
        device=logits.device,
    )
    row_matches = target_grades[:, None] == teacher_grades.to(logits.device)[None]
    if not bool(row_matches.any(1).all()):
        raise ValueError("Pathology grade target misses a model grade.")
    indices = row_matches.float().argmax(1).long()
    restricted = teacher_targets.to(logits.device)[indices][:, indices]
    restricted = restricted / restricted.sum(1, keepdim=True).clamp_min(1e-8)
    sample_grades = data["grade_group"].long().to(logits.device)
    sample_matches = sample_grades[:, None] == target_grades[None]
    if not bool(sample_matches.any(1).all()):
        unknown = sample_grades[~sample_matches.any(1)].unique().tolist()
        raise ValueError(f"No pathology soft target for ultrasound grades {unknown}.")
    sample_rows = sample_matches.float().argmax(1).long()
    grade_targets = restricted[sample_rows]

    teacher_sample_matches = (
        sample_grades[:, None] == teacher_grades.to(logits.device)[None]
    )
    teacher_sample_rows = teacher_sample_matches.float().argmax(1).long()
    full_targets = teacher_targets.to(logits.device)[teacher_sample_rows]
    class_targets = torch.stack(
        (
            full_targets[:, 0],
            full_targets[:, 1:3].sum(1),
            full_targets[:, 3:].sum(1),
        ),
        dim=1,
    )
    class_targets /= class_targets.sum(1, keepdim=True).clamp_min(1e-8)
    temperature = float(cfg.get("pathology_grade_student_temperature", 1.0))

    def distill(
        branch_logits: torch.Tensor, targets: torch.Tensor
    ) -> torch.Tensor:
        return F.kl_div(
            F.log_softmax(branch_logits.float() / temperature, dim=1),
            targets,
            reduction="batchmean",
        ) * temperature**2

    grade_conditioned = distill(logits, grade_targets)
    grade_image_logits = data.get("hyperbolic_grade_logits_image")
    grade_image = (
        distill(grade_image_logits, grade_targets)
        if grade_image_logits is not None
        else grade_conditioned.new_zeros(())
    )
    class_conditioned = distill(
        data["prototype_class_logits"], class_targets
    )
    class_image = distill(
        data["prototype_class_logits_image"], class_targets
    )
    image_weight = float(cfg.get("pathology_grade_image_weight", 0.5))
    grade_kd = (
        (1.0 - image_weight) * grade_conditioned + image_weight * grade_image
    )
    class_kd = (
        (1.0 - image_weight) * class_conditioned + image_weight * class_image
    )
    class_weight = float(cfg.get("pathology_grade_class_weight", 0.5))
    kd = (1.0 - class_weight) * grade_kd + class_weight * class_kd
    accuracy = (logits.argmax(1) == sample_rows).float().mean()
    class_accuracy = (
        data["prototype_class_logits"].argmax(1)
        == class_targets.argmax(1)
    ).float().mean()
    entropy = -(
        grade_targets * grade_targets.clamp_min(1e-8).log()
    ).sum(1).mean()
    return weight * kd, {
        "pathology_grade_kd": kd,
        "pathology_grade_native_kd": grade_kd,
        "pathology_class_kd": class_kd,
        "pathology_grade_conditioned_kd": grade_conditioned,
        "pathology_grade_image_kd": grade_image,
        "pathology_class_conditioned_kd": class_conditioned,
        "pathology_class_image_kd": class_image,
        "pathology_grade_target_entropy": entropy,
        "pathology_grade_accuracy": accuracy,
        "pathology_class_accuracy": class_accuracy,
    }


def _center_adversarial_regularization(
    model: ProtoProstate,
    data: dict[str, Any],
    cfg: DictConfig,
    stage: str,
    epoch: int,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Remove center information conditional on benign/non-csPCa/csPCa state."""
    weight = float(cfg.get("center_adversarial_loss_weight", 0.0))
    if weight <= 0 or model.center_discriminator is None or stage != "joint":
        zero = data["prototype_activations"].new_zeros(())
        return zero, {}
    center_names = tuple(str(value) for value in cfg.center_adversarial_centers)
    raw_centers = data.get("center")
    if raw_centers is None:
        raw_centers = [str(value).split("-")[0] for value in data["core_id"]]
    raw_centers = [str(value) for value in raw_centers]
    unknown = sorted(set(raw_centers) - set(center_names))
    if unknown:
        raise ValueError(f"Unknown centers for adversarial loss: {unknown}")
    lookup = {name: index for index, name in enumerate(center_names)}
    targets = torch.as_tensor(
        [lookup[value] for value in raw_centers],
        device=data["prototype_activations"].device,
        dtype=torch.long,
    )
    states = prototype_target_labels(model, data).to(targets.device)
    progress = min(
        1.0,
        (float(epoch) + 1.0) / max(float(cfg.get("joint_epochs", 1)), 1.0),
    )
    reversal = float(cfg.get("center_adversarial_reversal_max", 1.0)) * (
        2.0 / (1.0 + np.exp(-10.0 * progress)) - 1.0
    )
    logits = model.center_adversarial_logits(
        data["prototype_activations"],
        data["hyperbolic_direction"],
        states,
        reversal,
    )
    counts = torch.bincount(targets, minlength=len(center_names)).float()
    class_weights = targets.numel() / (len(center_names) * counts.clamp_min(1.0))
    class_weights[counts == 0] = 0
    ce = F.cross_entropy(logits, targets, weight=class_weights)
    accuracy = (logits.argmax(1) == targets).float().mean()
    return weight * ce, {
        "center_adversarial_ce": ce,
        "center_adversarial_accuracy": accuracy,
        "center_adversarial_reversal": logits.new_tensor(reversal),
    }


def pathology_geometry_regularization(
    data: dict[str, Any], cfg: DictConfig
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Transfer grade relationships from unpaired pathology representations.

    The loss never compares individual pathology and ultrasound cases. It maps
    each ultrasound grade to a training-only pathology grade centroid and also
    matches pairwise centroid similarities within the ultrasound minibatch.
    """
    weight = float(cfg.get("pathology_geometry_loss_weight", 0.0))
    embedding = data.get("pathology_geometry_embedding")
    if weight <= 0 or embedding is None:
        zero = data["prototype_activations"].new_zeros(())
        return zero, {}
    path = str(cfg.pathology_geometry_teacher_path)
    if path not in _PATHOLOGY_GEOMETRY_CACHE:
        archive = np.load(path)
        _PATHOLOGY_GEOMETRY_CACHE[path] = (
            torch.as_tensor(archive["grades"], dtype=torch.long),
            F.normalize(torch.as_tensor(archive["centroids"], dtype=torch.float), dim=1),
        )
        logging.info(
            "Loaded unpaired pathology geometry teacher %s with grades=%s and "
            "centroid_shape=%s.",
            path,
            archive["grades"].tolist(),
            tuple(archive["centroids"].shape),
        )
    grade_values, centroids = _PATHOLOGY_GEOMETRY_CACHE[path]
    grade_values = grade_values.to(embedding.device)
    centroids = centroids.to(embedding.device)
    grades = data["grade_group"].long().to(embedding.device)
    matches = grades[:, None] == grade_values[None]
    if not bool(matches.any(1).all()):
        unknown = grades[~matches.any(1)].unique().tolist()
        raise ValueError(f"Pathology teacher has no centroid for grades {unknown}")
    targets = matches.float().argmax(1).long()
    target_centroids = centroids[targets]
    student = F.normalize(embedding.float(), dim=1)
    alignment = (1.0 - (student * target_centroids).sum(1)).mean()
    student_similarity = student @ student.T
    teacher_similarity = target_centroids @ target_centroids.T
    off_diagonal = ~torch.eye(len(student), dtype=torch.bool, device=student.device)
    relation = (
        F.smooth_l1_loss(
            student_similarity[off_diagonal], teacher_similarity[off_diagonal]
        )
        if bool(off_diagonal.any())
        else alignment.new_zeros(())
    )
    temperature = float(cfg.get("pathology_geometry_temperature", 0.1))
    centroid_ce = F.cross_entropy(student @ centroids.T / temperature, targets)
    total = weight * (
        float(cfg.get("pathology_geometry_alignment_weight", 1.0)) * alignment
        + float(cfg.get("pathology_geometry_relation_weight", 1.0)) * relation
        + float(cfg.get("pathology_geometry_centroid_ce_weight", 0.25)) * centroid_ce
    )
    return total, {
        "pathology_geometry": total,
        "pathology_geometry_alignment": alignment,
        "pathology_geometry_relation": relation,
        "pathology_geometry_centroid_ce": centroid_ce,
    }


def _device(cfg: DictConfig) -> torch.device:
    requested = str(cfg.get("device", "cuda"))
    if requested.startswith("cuda") and not torch.cuda.is_available():
        logging.warning("CUDA is unavailable; falling back to CPU.")
        return torch.device("cpu")
    return torch.device(requested)


def _set_trainable(
    model: ProtoProstate,
    stage: str,
    unfreeze_adapters: bool,
    hyperbolic_only: bool = False,
    encoder_unfreeze_mode: str = "none",
    pathology_transport_only: bool = False,
) -> None:
    for parameter in model.parameters():
        parameter.requires_grad = False
    if pathology_transport_only and stage in {"warmup", "joint"}:
        encoder = getattr(model, "hyperbolic_needle_encoder", None)
        head = getattr(encoder, "pathology_transport_head", None)
        if head is None:
            raise ValueError(
                "pathology_transport_only_training requires a transport head."
            )
        for parameter in head.parameters():
            parameter.requires_grad = True
        return
    if stage in {"warmup", "joint"}:
        if not hyperbolic_only:
            model.prototype_layer.prototype_vectors.requires_grad = not bool(
                getattr(model, "lock_projected_prototypes", False)
            )
            for parameter in model.evidence_layer.parameters():
                parameter.requires_grad = True
            for parameter in model.prototype_classifier.parameters():
                parameter.requires_grad = True
            if model.core_classifier is not None:
                for parameter in model.core_classifier.parameters():
                    parameter.requires_grad = True
            if model.grade_classifier is not None:
                for parameter in model.grade_classifier.parameters():
                    parameter.requires_grad = True
            if model.occurrence_matcher is not None:
                for parameter in model.occurrence_matcher.parameters():
                    parameter.requires_grad = True
            if model.uncertainty_head is not None:
                for parameter in model.uncertainty_head.parameters():
                    parameter.requires_grad = True
            if model.clinical_head is not None:
                for parameter in model.clinical_head.parameters():
                    parameter.requires_grad = True
            if model.prompt_conditioner is not None:
                for parameter in model.prompt_conditioner.parameters():
                    parameter.requires_grad = True
        if model.hyperbolic_needle_encoder is not None:
            for parameter in model.hyperbolic_needle_encoder.parameters():
                parameter.requires_grad = True
            if model.hyperbolic_needle_encoder.structured_needle_selection:
                # Structured V20 burden is the occupied-bin selection fraction;
                # the legacy scalar involvement_head is bypassed in forward().
                # Do not advertise or optimize a disconnected parameter path.
                for parameter in model.hyperbolic_needle_encoder.involvement_head.parameters():
                    parameter.requires_grad = False
            if model.hyperbolic_needle_encoder.fusion_fixed_scale is not None:
                model.hyperbolic_needle_encoder.fusion_logit.requires_grad = False
        if model.dense_endpoint_head is not None:
            for parameter in model.dense_endpoint_head.parameters():
                parameter.requires_grad = True
        if model.prost_rl_endpoint_head is not None:
            for parameter in model.prost_rl_endpoint_head.parameters():
                parameter.requires_grad = True
            # Point/box/mask prompt paths are not called by the aligned head.
            # Keep only the no-mask dense embedding trainable so every declared
            # trainable parameter participates in the deployed computation.
            for parameter in model.prost_rl_endpoint_head.prompt_encoder.parameters():
                parameter.requires_grad = False
            model.prost_rl_endpoint_head.prompt_encoder.no_mask_embed.weight.requires_grad = True
        if model.pathology_geometry_projector is not None:
            for parameter in model.pathology_geometry_projector.parameters():
                parameter.requires_grad = True
        if model.semantic_anchor_core_projector is not None:
            for parameter in model.semantic_anchor_core_projector.parameters():
                parameter.requires_grad = True
            for parameter in model.semantic_anchor_prototype_projector.parameters():
                parameter.requires_grad = True
            model.semantic_anchor_log_temperature.requires_grad = True
            if model.semantic_anchor_fusion_logit is not None:
                model.semantic_anchor_fusion_logit.requires_grad = True
        if model.branch_fusion_logit is not None:
            model.branch_fusion_logit.requires_grad = True
        if model.center_discriminator is not None:
            for parameter in model.center_discriminator.parameters():
                parameter.requires_grad = True
    if stage == "joint" and not hyperbolic_only:
        mode = str(encoder_unfreeze_mode).lower()
        if mode == "full":
            count = model.unfreeze_encoder()
            # The dense DINOv3 wrapper never supplies masked-token positions,
            # so ``mask_token`` is connected only through an explicit
            # zero-multiplier in ``prepare_tokens_with_masks``.  In the
            # prostate-adapted checkpoint, every attention ``bias_mask`` is
            # also exactly zero, which makes the corresponding qkv bias
            # mathematically absent from F.linear.  Do not advertise or apply
            # AdamW decay to parameters that cannot receive a task gradient.
            encoder_buffers = dict(model.image_encoder.named_buffers())
            has_dino_bias_masks = any(
                name.endswith(".attn.qkv.bias_mask")
                for name in encoder_buffers
            )
            if has_dino_bias_masks:
                for name, parameter in model.image_encoder.named_parameters():
                    inactive = name.endswith("mask_token")
                    if name.endswith(".attn.qkv.bias"):
                        mask_name = name[: -len("bias")] + "bias_mask"
                        bias_mask = encoder_buffers.get(mask_name)
                        inactive = inactive or (
                            bias_mask is not None
                            and int(torch.count_nonzero(bias_mask).item()) == 0
                        )
                    if inactive:
                        parameter.requires_grad = False
                count = sum(
                    parameter.numel()
                    for parameter in model.image_encoder.parameters()
                    if parameter.requires_grad
                )
            # Wrappers may declare parameters outside their forward path (for
            # example V-JEPA's video patch embedding when only images are fed).
            inactive_prefixes = tuple(
                getattr(model.image_encoder, "inactive_parameter_prefixes", ())
            )
            if inactive_prefixes:
                for name, parameter in model.image_encoder.named_parameters():
                    if name.startswith(inactive_prefixes):
                        parameter.requires_grad = False
                count = sum(
                    parameter.numel()
                    for parameter in model.image_encoder.parameters()
                    if parameter.requires_grad
                )
            logging.info("Unfroze %d complete image-encoder parameters.", count)
        elif mode == "lora":
            count = model.unfreeze_lora()
            logging.info("Unfroze %d DINOv3 LoRA parameters.", count)
        elif mode in {"register", "registers"}:
            count = model.unfreeze_registers()
            logging.info("Unfroze %d DINOv3 register-token parameters.", count)
        elif mode == "adapters" or unfreeze_adapters:
            count = model.unfreeze_adapters()
            logging.info("Unfroze %d image-encoder adapter parameters.", count)
        elif mode not in {"none", "frozen"}:
            raise ValueError(f"Unknown encoder_unfreeze_mode={encoder_unfreeze_mode!r}")
    if stage == "slots":
        model.prototype_layer.slot_logits.requires_grad = True
    if stage == "last_layer":
        for parameter in model.prototype_classifier.parameters():
            parameter.requires_grad = True
        if model.core_classifier is not None:
            for parameter in model.core_classifier.parameters():
                parameter.requires_grad = True
        if model.grade_classifier is not None:
            for parameter in model.grade_classifier.parameters():
                parameter.requires_grad = True
        if model.prompt_conditioner is not None:
            for parameter in model.prompt_conditioner.classifier.parameters():
                parameter.requires_grad = True
        if model.hyperbolic_needle_encoder is not None:
            model.hyperbolic_needle_encoder.fusion_logit.requires_grad = (
                model.hyperbolic_needle_encoder.fusion_fixed_scale is None
            )
            if model.hyperbolic_needle_encoder.band_endpoint_readout is not None:
                for parameter in (
                    model.hyperbolic_needle_encoder.band_endpoint_readout.parameters()
                ):
                    parameter.requires_grad = True
        if model.dense_endpoint_head is not None:
            for parameter in model.dense_endpoint_head.parameters():
                parameter.requires_grad = True
        if model.prost_rl_endpoint_head is not None:
            for parameter in model.prost_rl_endpoint_head.parameters():
                parameter.requires_grad = True
            for parameter in model.prost_rl_endpoint_head.prompt_encoder.parameters():
                parameter.requires_grad = False
            model.prost_rl_endpoint_head.prompt_encoder.no_mask_embed.weight.requires_grad = True
        if model.branch_fusion_logit is not None:
            model.branch_fusion_logit.requires_grad = True
        if model.semantic_anchor_fusion_logit is not None:
            model.semantic_anchor_fusion_logit.requires_grad = True
        if model.semantic_anchor_core_projector is not None:
            # Final real-example projection changes occurrence vectors.  Keep
            # the semantic bridge trainable in the deployed post-projection
            # stage so the exported ultrasound-to-pathology correspondence is
            # not silently invalidated after joint training.
            for parameter in model.semantic_anchor_core_projector.parameters():
                parameter.requires_grad = True
            for parameter in model.semantic_anchor_prototype_projector.parameters():
                parameter.requires_grad = True
            model.semantic_anchor_log_temperature.requires_grad = True

    if (
        model.prost_rl_endpoint_head is not None
        and stage in {"warmup", "joint", "last_layer"}
    ):
        if not model.clinical_prompt_names:
            # Image-only uses null tokens; clinical MLPs are disconnected.
            for parameter in model.prost_rl_endpoint_head.floating_point_prompt_modules.parameters():
                parameter.requires_grad = False
        # Adding the same learned key bias to every key changes every
        # query-key logit by the same constant and therefore cancels exactly
        # under softmax.  SAM's Attention keeps these biases for architectural
        # symmetry, but they are structurally unidentifiable.  Excluding them
        # keeps the optimizer and strict gradient/update audit honest.
        for name, parameter in model.prost_rl_endpoint_head.named_parameters():
            if name.endswith(".k_proj.bias"):
                parameter.requires_grad = False


def _optimizer(model: ProtoProstate, cfg: DictConfig, stage: str):
    groups = model.get_params_groups()
    parameters: list[dict[str, Any]] = []
    if stage in {"warmup", "joint"}:
        parameters.extend(
            [
                {"params": groups["prototypes"], "lr": float(cfg.prototype_lr)},
                {"params": groups["last_layers"], "lr": float(cfg.head_lr)},
            ]
        )
        encoder_named = [
            (name, parameter)
            for name, parameter in model.image_encoder.named_parameters()
            if parameter.requires_grad
        ]
        adapters = [
            parameter for name, parameter in encoder_named
            if "adapter" in name.lower()
        ]
        encoder_base = [
            parameter for name, parameter in encoder_named
            if "adapter" not in name.lower()
        ]
        if encoder_base:
            parameters.append(
                {
                    "params": encoder_base,
                    "lr": float(cfg.get("encoder_lr", cfg.adapter_lr)),
                }
            )
        if adapters:
            parameters.append({"params": adapters, "lr": float(cfg.adapter_lr)})
    elif stage == "slots":
        parameters.append({"params": groups["slots"], "lr": float(cfg.slot_lr)})
    elif stage == "last_layer":
        parameters.append(
            {"params": groups["classification_layers"], "lr": float(cfg.last_layer_lr)}
        )
    # Optimizer membership is a hard correctness invariant.  V5--V20 declared
    # the occurrence temperature trainable but silently omitted it here.  Keep
    # only trainable tensors, reject duplicates, and fail before an expensive
    # job if any requires-grad parameter is absent.
    for group in parameters:
        group["params"] = [
            parameter for parameter in group["params"] if parameter.requires_grad
        ]
    parameters = [group for group in parameters if group["params"]]
    actual = [parameter for group in parameters for parameter in group["params"]]
    actual_ids = [id(parameter) for parameter in actual]
    if len(actual_ids) != len(set(actual_ids)):
        duplicates = sorted(
            name
            for name, parameter in model.named_parameters()
            if actual_ids.count(id(parameter)) > 1
        )
        raise RuntimeError(
            f"Optimizer stage {stage!r} contains duplicate parameters: {duplicates}"
        )
    expected = {
        id(parameter): name
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    }
    missing = sorted(expected[parameter_id] for parameter_id in expected.keys() - set(actual_ids))
    unexpected_ids = set(actual_ids) - expected.keys()
    if missing or unexpected_ids:
        raise RuntimeError(
            f"Optimizer stage {stage!r} membership mismatch: missing={missing}, "
            f"unexpected_trainable_ids={len(unexpected_ids)}"
        )
    logging.info(
        "Optimizer audit stage=%s trainable_tensors=%d trainable_parameters=%d",
        stage,
        len(actual),
        sum(parameter.numel() for parameter in actual),
    )
    return torch.optim.AdamW(
        parameters, weight_decay=float(cfg.get("weight_decay", 0))
    )


def _prototype_stats(model: ProtoProstate) -> dict[str, float]:
    slots = model.prototype_layer.slot_weights().detach()
    anchors = model.prototype_layer.projection_anchors.detach()
    prototypes = model.prototype_layer.prototype_vectors.detach()
    stats = {
        "slot_mean": float(slots.mean()),
        "slot_min": float(slots.min()),
        "slot_max": float(slots.max()),
        "active_slots": float((slots >= 0.5).sum()),
        "slot_temperature": float(model.prototype_layer.slot_temperature),
    }
    if bool(model.prototype_layer.has_projection_anchors.item()):
        cosine = F.cosine_similarity(prototypes, anchors, dim=1)
        stats["prototype_anchor_cosine"] = float(cosine.mean())
        stats["prototype_anchor_cosine_min"] = float(cosine.min())
    if model.occurrence_matcher is not None:
        stats["occurrence_temperature"] = float(
            model.occurrence_matcher.logit_scale.detach().exp().reciprocal()
        )
    return stats


def _criterion_with_components(criterion, data, task_weights=None):
    """Evaluate the reused Prost-RL loss while exposing its task components."""
    if not hasattr(criterion, "losses"):
        loss = criterion(data)
        return loss, {"prost_rl_loss": loss.detach()}
    parts = []
    metrics = {}
    task_weights = dict(task_weights or {})
    for index, (loss_module, weight) in enumerate(
        zip(criterion.losses, criterion.weights)
    ):
        name = type(loss_module).__name__
        task_name = "classification" if name == "ImageLevelClassificationLoss" else "heatmap"
        value = loss_module(data) * weight * float(task_weights.get(task_name, 1.0))
        parts.append(value)
        metrics[f"task_{index}_{name}"] = value.detach()
    return sum(parts), metrics


def _ordinal_grade_loss(
    model: ProtoProstate,
    data: dict[str, Any],
    cfg: DictConfig,
    stage: str = "fine",
    epoch: int = 0,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Cancer-conditional, progressive ordinal likelihood for GG2--GG5."""
    if model.grade_classifier is None or "grade_group_logits" not in data:
        zero = data["prototype_activations"].new_zeros(())
        return zero, {"grade_ordinal": zero, "grade_monotonic": zero}
    logits = data["grade_group_logits"]
    grades = data["grade_group"].to(logits.device).long()
    cancer = grades >= 2
    if not bool(cancer.any()):
        zero = logits.new_zeros(())
        return zero, {"grade_ordinal": zero, "grade_monotonic": zero}
    logits = logits[cancer]
    grades = grades[cancer]
    thresholds = torch.as_tensor(
        model.grade_thresholds, device=logits.device, dtype=grades.dtype
    )
    targets = (grades[:, None] >= thresholds[None, :]).to(logits.dtype)
    pos_weights = torch.as_tensor(
        list(cfg.get("grade_pos_weights", [1.0] * len(model.grade_thresholds))),
        device=logits.device,
        dtype=logits.dtype,
    )
    if bool(cfg.get("progressive_grading", True)) and stage == "warmup":
        active_thresholds = 0
    elif bool(cfg.get("progressive_grading", True)) and stage == "joint" and epoch < int(
        cfg.get("coarse_grade_epochs", 2)
    ):
        active_thresholds = 1
    else:
        active_thresholds = logits.shape[1]
    ordinal = (
        F.binary_cross_entropy_with_logits(
            logits[:, :active_thresholds],
            targets[:, :active_thresholds],
            pos_weight=pos_weights[:active_thresholds],
        )
        if active_thresholds
        else logits.new_zeros(())
    )
    monotonic = (
        F.relu(logits[:, 1:active_thresholds] - logits[:, : active_thresholds - 1]).mean()
        if active_thresholds > 1
        else logits.new_zeros(())
    )
    total = (
        float(cfg.get("grade_loss_weight", 0.0)) * ordinal
        + float(cfg.get("grade_monotonic_weight", 0.0)) * monotonic
    )
    return total, {
        "grade_ordinal": ordinal,
        "grade_monotonic": monotonic,
        "grade_active_thresholds": logits.new_tensor(float(active_thresholds)),
    }


def _uncertainty_abstention_loss(
    data: dict[str, Any], cfg: DictConfig
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """ProtoASNet virtual-prediction abstention without hiding eval cases."""
    if "uncertainty_logit" not in data:
        zero = data["prototype_activations"].new_zeros(())
        return zero, {"abstention": zero, "uncertainty_mean": zero}
    probabilities = torch.softmax(data["prototype_class_logits"], dim=-1)
    labels = data["label"].long().to(probabilities.device)
    targets = F.one_hot(labels, num_classes=2).to(probabilities.dtype)
    alpha = torch.sigmoid(data["uncertainty_logit"]).clamp(1e-6, 1 - 1e-6)
    virtual = (1.0 - alpha[:, None]) * probabilities + alpha[:, None] * targets
    fit = -torch.log(virtual.gather(1, labels[:, None]).squeeze(1).clamp_min(1e-7)).mean()
    penalty = -torch.log1p(-alpha).mean()
    raw = fit + float(cfg.get("abstention_penalty", 0.1)) * penalty
    total = float(cfg.get("abstention_loss_weight", 0.0)) * raw
    return total, {
        "abstention": raw,
        "abstention_fit": fit,
        "abstention_penalty": penalty,
        "uncertainty_mean": alpha.mean(),
    }


def _patient_structure_loss(
    model: ProtoProstate,
    data: dict[str, Any],
    cfg: DictConfig,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Use repeated training cores without assuming one diagnosis per patient.

    Same-patient cores with the same three-state label are encouraged to use a
    similar distribution over prototypes. Differently labelled cores are only
    ranked by expected severity, preserving focal disease heterogeneity.
    Patient identity is not a model input and this loss is absent at inference.
    """
    activations = data["prototype_activations"]
    zero = activations.new_zeros(())
    same_weight = float(cfg.get("patient_same_state_weight", 0.0))
    rank_weight = float(cfg.get("patient_rank_weight", 0.0))
    if (same_weight <= 0 and rank_weight <= 0) or "patient_id" not in data:
        return zero, {
            "patient_same_state": zero,
            "patient_rank": zero,
            "patient_pairs": zero,
        }

    patient_ids = [str(value) for value in data["patient_id"]]
    labels = prototype_target_labels(model, data).to(activations.device)
    probabilities = torch.softmax(data["prototype_class_logits"], dim=-1)
    severity_values = torch.arange(
        probabilities.shape[1], device=probabilities.device, dtype=probabilities.dtype
    )
    severity = (probabilities * severity_values[None]).sum(1)
    prototype_mixture = torch.softmax(activations / 0.25, dim=1)
    same_terms = []
    rank_terms = []
    pair_count = 0
    margin = float(cfg.get("patient_rank_margin", 0.15))
    for first in range(len(patient_ids)):
        for second in range(first + 1, len(patient_ids)):
            if patient_ids[first] != patient_ids[second]:
                continue
            pair_count += 1
            if int(labels[first]) == int(labels[second]):
                mean = 0.5 * (
                    prototype_mixture[first] + prototype_mixture[second]
                )
                same_terms.append(
                    0.5
                    * (
                        F.kl_div(
                            mean.clamp_min(1e-8).log(),
                            prototype_mixture[first],
                            reduction="sum",
                        )
                        + F.kl_div(
                            mean.clamp_min(1e-8).log(),
                            prototype_mixture[second],
                            reduction="sum",
                        )
                    )
                )
            else:
                direction = torch.sign((labels[first] - labels[second]).float())
                rank_terms.append(
                    F.relu(
                        activations.new_tensor(margin)
                        - direction * (severity[first] - severity[second])
                    )
                )
    same = torch.stack(same_terms).mean() if same_terms else zero
    rank = torch.stack(rank_terms).mean() if rank_terms else zero
    total = same_weight * same + rank_weight * rank
    return total, {
        "patient_same_state": same,
        "patient_rank": rank,
        "patient_pairs": activations.new_tensor(float(pair_count)),
    }


def _set_fold_grade_weights(cfg: DictConfig, train_loader, model: ProtoProstate) -> None:
    """Compute stable fold-specific sqrt inverse-frequency ordinal weights."""
    if model.grade_classifier is None:
        return
    from .data import grades_of

    grades = grades_of(train_loader)
    cancer_grades = grades[grades >= 2]
    max_weight = float(cfg.get("grade_pos_weight_max", 10.0))
    weights = []
    threshold_counts = {}
    for threshold in model.grade_thresholds:
        positives = int((cancer_grades >= threshold).sum())
        negatives = int(len(cancer_grades) - positives)
        weight = np.sqrt(negatives / max(positives, 1))
        weights.append(float(np.clip(weight, 1.0, max_weight)))
        threshold_counts[f"gg_ge_{threshold}"] = positives
    cfg.grade_pos_weights = weights
    cfg.grade_threshold_positive_counts = threshold_counts
    logging.info(
        "Observed fold Grade Groups: %s; ordinal pos_weights=%s",
        dict(zip(*np.unique(grades, return_counts=True))),
        [round(value, 3) for value in weights],
    )


def _set_ordinal_frequency_weights(
    cfg: DictConfig, train_loader, model: ProtoProstate
) -> None:
    """Record fold-specific inverse-frequency weights for the V17 branches."""
    if float(cfg.get("ordinal_domain_loss_weight", 0.0)) <= 0:
        return
    from .data import grades_of

    grades = grades_of(train_loader)
    states = np.where(grades == 0, 0, np.where(grades >= 3, 2, 1))
    target_values = np.asarray(model.hyperbolic_needle_encoder.target_values, dtype=int)

    def inverse(values: np.ndarray, categories: np.ndarray) -> list[float]:
        counts = np.asarray([(values == category).sum() for category in categories], dtype=float)
        raw = len(values) / np.maximum(counts, 1.0)
        raw /= raw.mean()
        raw = np.clip(raw, 0.2, float(cfg.get("ordinal_inverse_frequency_max", 5.0)))
        return [float(value) for value in raw]

    cfg.ordinal_local_inverse_frequency_weights = inverse(
        states, np.arange(model.num_classes)
    )
    cfg.ordinal_grade_inverse_frequency_weights = inverse(grades, target_values)
    cfg.ordinal_grade_counts = {
        str(int(value)): int((grades == value).sum()) for value in target_values
    }
    cfg.ordinal_state_counts = {
        str(int(value)): int((states == value).sum()) for value in range(model.num_classes)
    }
    logging.info(
        "V17 state counts=%s grade counts=%s inverse weights local=%s grade=%s",
        cfg.ordinal_state_counts, cfg.ordinal_grade_counts,
        cfg.ordinal_local_inverse_frequency_weights,
        cfg.ordinal_grade_inverse_frequency_weights,
    )


def _grade_metrics(
    probabilities: np.ndarray, grades: np.ndarray
) -> tuple[dict[str, float], dict[str, np.ndarray]]:
    """Evaluate ordinal discrimination without pretending GG1 was observed."""
    observed_grades = np.asarray([0, 2, 3, 4, 5])
    probabilities = np.clip(probabilities, 0.0, 1.0)
    probabilities /= probabilities.sum(axis=1, keepdims=True).clip(min=1e-8)
    target_indices = np.asarray(
        [{int(value): index for index, value in enumerate(observed_grades)}[int(grade)] for grade in grades]
    )
    predicted_indices = probabilities.argmax(axis=1)
    predicted_grades = observed_grades[predicted_indices]
    metrics = {
        "grade_group_balanced_accuracy": float(
            balanced_accuracy_score(target_indices, predicted_indices)
        ),
        "grade_group_qwk": float(
            cohen_kappa_score(
                target_indices, predicted_indices, weights="quadratic"
            )
        ),
        "grade_group_mae": float(np.abs(predicted_grades - grades).mean()),
    }
    threshold_aucs = []
    survival = np.column_stack(
        [probabilities[:, 1:].sum(1), probabilities[:, 2:].sum(1),
         probabilities[:, 3:].sum(1), probabilities[:, 4:].sum(1)]
    )
    for threshold_idx, threshold in enumerate((2, 3, 4, 5)):
        labels = (grades >= threshold).astype(int)
        if np.unique(labels).size == 2:
            auc = float(roc_auc_score(labels, survival[:, threshold_idx]))
            metrics[f"grade_auc_ge_{threshold}"] = auc
            threshold_aucs.append(auc)
    metrics["grade_ordinal_macro_auc"] = float(np.mean(threshold_aucs))
    if np.unique(target_indices).size == len(observed_grades):
        metrics["grade_group_macro_ovr_auc"] = float(
            roc_auc_score(
                target_indices, probabilities, multi_class="ovr", average="macro"
            )
        )
    return metrics, {
        "predicted_grades": predicted_grades,
        "probabilities": probabilities,
    }


def train_epoch(
    model: ProtoProstate,
    loader,
    criterion,
    optimizer,
    device: torch.device,
    cfg: DictConfig,
    stage: str,
    epoch: int,
) -> dict[str, float]:
    model.train()
    if bool(cfg.get("pathology_transport_only_training", False)):
        # Keep every frozen V18 component, including Transformer/dropout and
        # prompt conditioning, in deterministic evaluation mode. Only the new
        # shared cross-modal head is stochastic/trainable.
        model.eval()
        model.hyperbolic_needle_encoder.pathology_transport_head.train()
    if not any(parameter.requires_grad for parameter in model.image_encoder.parameters()):
        model.image_encoder.eval()
    amp_dtype_name = str(cfg.get("amp_dtype", "float16")).lower()
    amp_dtypes = {"float16": torch.float16, "fp16": torch.float16,
                  "bfloat16": torch.bfloat16, "bf16": torch.bfloat16}
    if amp_dtype_name not in amp_dtypes:
        raise ValueError(f"Unsupported amp_dtype={amp_dtype_name!r}")
    amp_dtype = amp_dtypes[amp_dtype_name]
    amp_enabled = bool(cfg.use_amp and device.type == "cuda")
    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=bool(amp_enabled and amp_dtype == torch.float16),
        init_scale=float(cfg.get("amp_grad_scaler_init_scale", 2.0**16)),
    )
    totals: dict[str, float] = {}
    batches = 0
    skipped_cores = 0
    optimizer_steps = 0
    amp_skipped_steps = 0
    consecutive_amp_skipped_steps = 0
    initial_amp_scale = float(scaler.get_scale())
    accumulation = int(cfg.accumulate_grad_steps)
    optimizer.zero_grad(set_to_none=True)
    coefficients = dict(cfg.prototype_loss_weights)
    if stage == "slots":
        coefficients.update(dict(cfg.get("slot_loss_weights", {})))

    def finish_optimizer_step(step_label: str) -> bool:
        """Clip and step without turning recoverable AMP overflow into a crash.

        GradScaler intentionally permits early FP16 overflows while it calibrates
        its scale.  After ``unscale_`` it records the overflow and ``step`` skips
        the optimizer update; calling ``clip_grad_norm_`` with
        ``error_if_nonfinite=True`` before that skip incorrectly makes the
        expected calibration event fatal.  We therefore inspect unscaled
        gradients first, let GradScaler lower its scale for recoverable FP16
        overflow, and retain a bounded hard failure for persistent or non-AMP
        non-finite gradients.
        """
        nonlocal optimizer_steps
        nonlocal amp_skipped_steps
        nonlocal consecutive_amp_skipped_steps

        named_gradients = [
            (name, parameter)
            for name, parameter in model.named_parameters()
            if parameter.requires_grad and parameter.grad is not None
        ]
        if scaler.is_enabled():
            scaler.unscale_(optimizer)
        nonfinite = [
            name
            for name, parameter in named_gradients
            if not bool(torch.isfinite(parameter.grad).all())
        ]
        if nonfinite:
            if not scaler.is_enabled():
                raise RuntimeError(
                    f"Non-finite {stage} gradients at {step_label}: "
                    f"parameters={nonfinite[:20]}"
                )
            previous_scale = float(scaler.get_scale())
            # The overflow status captured by unscale_ makes this a no-op.
            scaler.step(optimizer)
            scaler.update()
            updated_scale = float(scaler.get_scale())
            optimizer.zero_grad(set_to_none=True)
            amp_skipped_steps += 1
            consecutive_amp_skipped_steps += 1
            logging.warning(
                "Skipped recoverable AMP optimizer step for %s at %s: "
                "scale %.1f -> %.1f; nonfinite_parameters=%s",
                stage,
                step_label,
                previous_scale,
                updated_scale,
                nonfinite[:20],
            )
            maximum_skips = int(cfg.get("max_consecutive_amp_skipped_steps", 8))
            if updated_scale >= previous_scale or consecutive_amp_skipped_steps > maximum_skips:
                raise RuntimeError(
                    f"Persistent non-finite {stage} AMP gradients at {step_label}: "
                    f"scale={updated_scale}, consecutive_skips="
                    f"{consecutive_amp_skipped_steps}, parameters={nonfinite[:20]}"
                )
            return False

        clip_norm = float(cfg.get("gradient_clip_norm", 0.0))
        if clip_norm > 0:
            gradient_norm = torch.nn.utils.clip_grad_norm_(
                [parameter for _, parameter in named_gradients],
                clip_norm,
                error_if_nonfinite=True,
            )
            if not bool(torch.isfinite(gradient_norm)):
                raise RuntimeError(
                    f"Non-finite {stage} gradient norm at {step_label}."
                )
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)
        optimizer_steps += 1
        consecutive_amp_skipped_steps = 0
        return True

    for batch_idx, batch in enumerate(tqdm(loader, desc=f"{stage} {epoch}")):
        with torch.autocast(
            device_type=device.type,
            dtype=amp_dtype,
            enabled=amp_enabled,
        ):
            data = forward_batch(model, batch, device)
            if data is None:
                skipped_cores += len(batch["bmode"])
                continue
            skipped_cores += int(data["invalid_region_cores_filtered"])
            configured_task_weights = cfg.get("task_loss_weights", None)
            weak_loss, task_components = _criterion_with_components(
                criterion, data, configured_task_weights
            )
            proto_loss, components = prototype_regularization(
                model,
                data,
                coefficients,
                float(cfg.get("prototype_ce_weight", 1.0)),
                bool(cfg.get("involvement_weight_classification", True)),
            )
            grade_loss, grade_components = _ordinal_grade_loss(
                model, data, cfg, stage, epoch
            )
            uncertainty_loss, uncertainty_components = _uncertainty_abstention_loss(
                data, cfg
            )
            patient_loss, patient_components = _patient_structure_loss(
                model, data, cfg
            )
            hyperbolic_loss, hyperbolic_components = (
                hyperbolic_needle_regularization(model, data, cfg)
            )
            needle_prototype_loss, needle_prototype_components = (
                needle_axis_prototype_regularization(model, data, cfg)
            )
            pathology_transport_loss, pathology_transport_components = (
                pathology_prototype_transport_regularization(model, data, cfg)
            )
            pathology_geometry_loss, pathology_geometry_components = (
                pathology_geometry_regularization(data, cfg)
            )
            semantic_anchor_loss, semantic_anchor_components = (
                semantic_anchor_regularization(model, data, cfg, stage)
            )
            pathology_manifold_loss, pathology_manifold_components = (
                pathology_manifold_regularization(model, data, cfg, stage)
            )
            pathology_grade_loss, pathology_grade_components = (
                _pathology_grade_distribution_regularization(model, data, cfg, stage)
            )
            center_loss, center_components = _center_adversarial_regularization(
                model, data, cfg, stage, epoch
            )
            ordinal_domain_loss, ordinal_domain_components = (
                ordinal_domain_regularization(model, data, cfg, stage, epoch)
            )
            full_loss = (
                (
                    1.0
                    if configured_task_weights is not None
                    else float(cfg.get("weak_loss_weight", 1.0))
                )
                * weak_loss
                + proto_loss
                + grade_loss
                + uncertainty_loss
                + patient_loss
                + hyperbolic_loss
                + needle_prototype_loss
                + pathology_transport_loss
                + pathology_geometry_loss
                + semantic_anchor_loss
                + pathology_manifold_loss
                + pathology_grade_loss
                + center_loss
                + ordinal_domain_loss
            )
            loss = full_loss / accumulation
        if not bool(torch.isfinite(full_loss)):
            raise RuntimeError(
                f"Non-finite {stage} loss at epoch={epoch} batch={batch_idx}."
            )
        scaler.scale(loss).backward()
        batches += 1
        if batches % accumulation == 0:
            finish_optimizer_step(f"epoch={epoch} batch={batch_idx}")

        values = {"loss": full_loss.detach(), "weak": weak_loss.detach()}
        values.update(task_components)
        values.update({name: value.detach() for name, value in components.items()})
        values.update(
            {name: value.detach() for name, value in grade_components.items()}
        )
        values.update(
            {name: value.detach() for name, value in uncertainty_components.items()}
        )
        values.update(
            {name: value.detach() for name, value in patient_components.items()}
        )
        values.update(
            {name: value.detach() for name, value in hyperbolic_components.items()}
        )
        values.update(
            {
                name: value.detach()
                for name, value in needle_prototype_components.items()
            }
        )
        values.update(
            {
                name: value.detach()
                for name, value in pathology_transport_components.items()
            }
        )
        values.update(
            {
                name: value.detach()
                for name, value in pathology_geometry_components.items()
            }
        )
        values.update(
            {name: value.detach() for name, value in semantic_anchor_components.items()}
        )
        values.update(
            {
                name: value.detach()
                for name, value in pathology_manifold_components.items()
            }
        )
        values.update(
            {name: value.detach() for name, value in pathology_grade_components.items()}
        )
        values.update(
            {name: value.detach() for name, value in center_components.items()}
        )
        values.update(
            {
                name: value.detach() for name, value in ordinal_domain_components.items()
            }
        )
        for name, value in values.items():
            totals[name] = totals.get(name, 0.0) + float(value)
        if cfg.get("debug", False) and batch_idx >= 2:
            break

    if batches and batches % accumulation:
        finish_optimizer_step(f"trailing epoch={epoch}")
    if batches and optimizer_steps == 0:
        raise RuntimeError(
            f"No {stage} optimizer step completed at epoch={epoch}; "
            f"AMP-skipped steps={amp_skipped_steps}."
        )
    metrics = {name: value / max(batches, 1) for name, value in totals.items()}
    metrics["invalid_region_cores_filtered"] = float(skipped_cores)
    metrics["optimizer_steps"] = float(optimizer_steps)
    metrics["amp_skipped_steps"] = float(amp_skipped_steps)
    metrics["amp_scale_initial"] = initial_amp_scale
    metrics["amp_scale_final"] = float(scaler.get_scale())
    metrics.update(_prototype_stats(model))
    return metrics


@torch.no_grad()
def evaluate(model, loader, device: torch.device, cfg: DictConfig):
    model.eval()
    evaluator = CancerLogitsHeatmapsEvaluator(
        log_images=False,
        include_patient_metrics=bool(cfg.get("include_patient_metrics", False)),
    )
    skipped = 0
    grade_probabilities = []
    grade_labels = []
    prototype_probabilities = []
    prototype_pca_probabilities = []
    isup_probabilities = []
    image_only_probabilities = []
    conditioned_image_probabilities = []
    conditioned_image_pca_probabilities = []
    prompt_only_probabilities = []
    prompt_only_pca_probabilities = []
    prompt_gate_means = []
    prompt_gate_stds = []
    clinical_probabilities = []
    clinical_contributions = []
    uncertainty_values = []
    hyperbolic_grade_probabilities = []
    hyperbolic_state_probabilities = []
    hyperbolic_predicted_involvement = []
    hyperbolic_radii = []
    pathology_transport_grade_probabilities = []
    pathology_transport_state_probabilities = []
    pathology_transport_severities = []
    semantic_anchor_probabilities = []
    needle_selection_entropies = []
    needle_selection_cohesions = []
    needle_selection_active_fractions = []
    for batch in tqdm(loader, desc="Evaluate"):
        data = forward_batch(model, batch, device)
        if data is None:
            skipped += len(batch["bmode"])
            continue
        skipped += int(data["invalid_region_cores_filtered"])
        evaluator(data)
        prototype_class_probs = torch.softmax(
            data["prototype_class_logits"], dim=-1
        )
        image_class_probs = torch.softmax(
            data["prototype_class_logits_image"], dim=-1
        )
        if model.classification_target in {"three_way", "isup"}:
            cspca_start = 3 if model.classification_target == "isup" else 2
            prototype_probabilities.append(
                prototype_class_probs[:, cspca_start:].sum(1).detach().cpu().float()
            )
            prototype_pca_probabilities.append(
                prototype_class_probs[:, 1:].sum(1).detach().cpu().float()
            )
            image_only_probabilities.append(
                image_class_probs[:, cspca_start:].sum(1).detach().cpu().float()
            )
            if model.classification_target == "isup":
                isup_probabilities.append(
                    prototype_class_probs.detach().cpu().float()
                )
        else:
            prototype_probabilities.append(
                prototype_class_probs[:, 1].detach().cpu().float()
            )
            image_only_probabilities.append(
                image_class_probs[:, 1].detach().cpu().float()
            )
        if "prompt_class_logits" in data:
            conditioned_probs = torch.softmax(
                data["prototype_class_logits_conditioned"], dim=-1
            )
            prompt_probs = torch.softmax(data["prompt_class_logits"], dim=-1)
            if model.classification_target in {"three_way", "isup"}:
                cspca_start = 3 if model.classification_target == "isup" else 2
                conditioned_image_probabilities.append(
                    conditioned_probs[:, cspca_start:].sum(1).detach().cpu().float()
                )
                conditioned_image_pca_probabilities.append(
                    conditioned_probs[:, 1:].sum(1).detach().cpu().float()
                )
                prompt_only_probabilities.append(
                    prompt_probs[:, cspca_start:].sum(1).detach().cpu().float()
                )
                prompt_only_pca_probabilities.append(
                    prompt_probs[:, 1:].sum(1).detach().cpu().float()
                )
            else:
                conditioned_image_probabilities.append(
                    conditioned_probs[:, 1].detach().cpu().float()
                )
                prompt_only_probabilities.append(
                    prompt_probs[:, 1].detach().cpu().float()
                )
            gate = data["prompt_spatial_gate"].detach().cpu().float()
            prompt_gate_means.append(gate.flatten(1).mean(1))
            prompt_gate_stds.append(gate.flatten(1).std(1))
        if "clinical_logit" in data:
            clinical_probabilities.append(
                torch.sigmoid(data["clinical_logit"]).detach().cpu().float()
            )
            clinical_contributions.append(
                data["clinical_contributions"].detach().cpu().float()
            )
        if "grade_group_probabilities" in data:
            grade_probabilities.append(
                data["grade_group_probabilities"].detach().cpu().float()
            )
            grade_labels.append(data["grade_group"].detach().cpu().long())
        if "uncertainty_logit" in data:
            uncertainty_values.append(
                torch.sigmoid(data["uncertainty_logit"]).detach().cpu().float()
            )
        if "hyperbolic_grade_logits" in data:
            hyperbolic_grade_probabilities.append(
                torch.softmax(data["hyperbolic_grade_logits"], dim=1)
                .detach().cpu().float()
            )
            hyperbolic_state_probabilities.append(
                torch.softmax(data["hyperbolic_three_way_logits"], dim=1)
                .detach().cpu().float()
            )
            hyperbolic_predicted_involvement.append(
                data["hyperbolic_predicted_involvement"].detach().cpu().float()
            )
            hyperbolic_radii.append(
                data["hyperbolic_radius"].detach().cpu().float()
            )
            if model.hyperbolic_needle_encoder.structured_needle_selection:
                selections = data["needle_selection_probabilities"].detach().float()
                occupied = data["needle_axis_occupied_bins"].bool()
                clipped = selections.clamp(1e-6, 1 - 1e-6)
                entropy_per_bin = -(
                    clipped * clipped.log()
                    + (1 - clipped) * (1 - clipped).log()
                )
                needle_selection_entropies.append(
                    (entropy_per_bin * occupied).sum(1).div(
                        occupied.sum(1).clamp_min(1)
                    ).cpu()
                )
                adjacent = occupied[:, :-1] & occupied[:, 1:]
                differences = (selections[:, 1:] - selections[:, :-1]).abs()
                needle_selection_cohesions.append(
                    (differences * adjacent).sum(1).div(
                        adjacent.sum(1).clamp_min(1)
                    ).cpu()
                )
                needle_selection_active_fractions.append(
                    (((selections > 0.5) & occupied).sum(1) / occupied.sum(1).clamp_min(1)).cpu()
                )
        if "pathology_transport_grade_logits" in data:
            pathology_transport_grade_probabilities.append(
                torch.softmax(data["pathology_transport_grade_logits"], dim=1)
                .detach().cpu().float()
            )
            pathology_transport_state_probabilities.append(
                torch.softmax(data["pathology_transport_three_way_logits"], dim=1)
                .detach().cpu().float()
            )
            pathology_transport_severities.append(
                data["pathology_transport_predicted_severity"]
                .detach().cpu().float()
            )
        if "semantic_anchor_class_logits" in data:
            semantic_anchor_probabilities.append(
                torch.softmax(data["semantic_anchor_class_logits"], dim=1)
                .detach().cpu().float()
            )
        if cfg.get("debug", False) and evaluator.iter >= 3:
            break
    if evaluator.iter == 0:
        raise RuntimeError("No cores with a valid prostate/needle intersection were evaluated.")
    metrics = evaluator.aggregate_metrics()
    table = evaluator.results_table
    prototype_scores = torch.cat(prototype_probabilities).numpy()
    table["prototype_class_probability"] = prototype_scores
    prototype_labels = (
        (table.grade_group.values > 2).astype(int)
        if model.classification_target in {"cspca", "three_way", "isup"}
        else table.label.values
    )
    prototype_metrics = calculate_metrics(
        prototype_scores, prototype_labels, log_images=False
    )
    metrics.update({f"prototype_{key}": value for key, value in prototype_metrics.items()})
    if model.classification_target in {"cspca", "three_way", "isup"}:
        table["prototype_cspca_probability"] = prototype_scores
        metrics.update(
            {f"prototype_cspca_{key}": value for key, value in prototype_metrics.items()}
        )
    if model.classification_target in {"three_way", "isup"}:
        pca_scores = torch.cat(prototype_pca_probabilities).numpy()
        pca_metrics = calculate_metrics(
            pca_scores, table.label.values, log_images=False
        )
        table["prototype_pca_probability"] = pca_scores
        metrics.update(
            {f"prototype_pca_{key}": value for key, value in pca_metrics.items()}
        )
        metrics["prototype_endpoint_mean_auc"] = float(
            0.5 * (pca_metrics["core_auc"] + prototype_metrics["core_auc"])
        )
    image_only_scores = torch.cat(image_only_probabilities).numpy()
    table[f"image_only_{model.classification_target}_probability"] = image_only_scores
    image_only_metrics = calculate_metrics(
        image_only_scores, prototype_labels, log_images=False
    )
    metrics.update(
        {f"image_only_{key}": value for key, value in image_only_metrics.items()}
    )
    if conditioned_image_probabilities:
        conditioned_scores = torch.cat(conditioned_image_probabilities).numpy()
        prompt_scores = torch.cat(prompt_only_probabilities).numpy()
        table["conditioned_image_cspca_probability"] = conditioned_scores
        table["prompt_only_cspca_probability"] = prompt_scores
        for prefix, scores in (
            ("conditioned_image_cspca", conditioned_scores),
            ("prompt_only_cspca", prompt_scores),
        ):
            result = calculate_metrics(scores, prototype_labels, log_images=False)
            metrics.update({f"{prefix}_{key}": value for key, value in result.items()})
        if model.classification_target in {"three_way", "isup"}:
            conditioned_pca = torch.cat(
                conditioned_image_pca_probabilities
            ).numpy()
            prompt_pca = torch.cat(prompt_only_pca_probabilities).numpy()
            table["conditioned_image_pca_probability"] = conditioned_pca
            table["prompt_only_pca_probability"] = prompt_pca
            for prefix, scores in (
                ("conditioned_image_pca", conditioned_pca),
                ("prompt_only_pca", prompt_pca),
            ):
                result = calculate_metrics(scores, table.label.values, log_images=False)
                metrics.update(
                    {f"{prefix}_{key}": value for key, value in result.items()}
                )
        metrics["prompt_spatial_gate_mean"] = float(
            torch.cat(prompt_gate_means).mean()
        )
        metrics["prompt_spatial_gate_std"] = float(
            torch.cat(prompt_gate_stds).mean()
        )
    if clinical_probabilities:
        clinical_scores = torch.cat(clinical_probabilities).numpy()
        table["clinical_only_cancer_probability"] = clinical_scores
        prompt_metrics = calculate_metrics(
            clinical_scores, table.label.values, log_images=False
        )
        metrics.update(
            {f"clinical_only_{key}": value for key, value in prompt_metrics.items()}
        )
        contributions = torch.cat(clinical_contributions).numpy()
        for index, name in enumerate(model.clinical_prompt_names):
            table[f"clinical_contribution_{name}"] = contributions[:, index]
    metrics["core_accuracy"] = float(
        ((table.average_needle_heatmap_value.values > 0.5) == table.label.values).mean()
    )
    if "image_level_cancer_logits" in table.columns:
        cspca_labels = (table.grade_group.values > 2).astype(int)
        cspca_scores = (
            table.image_level_cspca_logits.values
            if "image_level_cspca_logits" in table.columns
            else table.image_level_cancer_logits.values
        )
        metrics["cspca_accuracy"] = float(
            ((cspca_scores > 0.5) == cspca_labels).mean()
        )
    if model.classification_target == "isup":
        probabilities = torch.cat(isup_probabilities).numpy()
        grades = table.grade_group.values.astype(int)
        predicted_grades = probabilities.argmax(1)
        metrics["isup_balanced_accuracy"] = float(
            balanced_accuracy_score(grades, predicted_grades)
        )
        metrics["isup_qwk"] = float(
            cohen_kappa_score(grades, predicted_grades, weights="quadratic")
        )
        metrics["isup_mae"] = float(np.abs(predicted_grades - grades).mean())
        if np.unique(grades).size == 6:
            metrics["isup_macro_ovr_auc"] = float(
                roc_auc_score(grades, probabilities, multi_class="ovr", average="macro")
            )
        table["predicted_grade_group"] = predicted_grades
        for grade in range(6):
            table[f"isup_probability_{grade}"] = probabilities[:, grade]
    if grade_probabilities:
        probabilities = torch.cat(grade_probabilities).numpy()
        grades = torch.cat(grade_labels).numpy()
        grade_metrics, grade_outputs = _grade_metrics(probabilities, grades)
        metrics.update(grade_metrics)
        table["predicted_grade_group"] = grade_outputs["predicted_grades"]
        for index, grade in enumerate((0, 2, 3, 4, 5)):
            table[f"grade_group_probability_{grade}"] = grade_outputs[
                "probabilities"
            ][:, index]
        cspca_scores = probabilities[:, 2:].sum(1)
        cspca_metrics = calculate_metrics(
            cspca_scores, (grades >= 3).astype(int), log_images=False
        )
        metrics.update(
            {f"hierarchical_cspca_{key}": value for key, value in cspca_metrics.items()}
        )
        table["hierarchical_cspca_probability"] = cspca_scores
    if hyperbolic_grade_probabilities:
        hyper_probabilities = torch.cat(hyperbolic_grade_probabilities).numpy()
        hyper_state_probabilities = torch.cat(
            hyperbolic_state_probabilities
        ).numpy()
        predicted_q = torch.cat(hyperbolic_predicted_involvement).numpy()
        radii = torch.cat(hyperbolic_radii).numpy()
        pca_scores = hyper_state_probabilities[:, 1:].sum(1)
        cspca_scores = hyper_state_probabilities[:, 2]
        target_values = tuple(model.hyperbolic_needle_encoder.target_values)
        table["hyperbolic_predicted_involvement"] = predicted_q
        table["hyperbolic_radius"] = radii
        table["hyperbolic_pca_probability"] = pca_scores
        table["hyperbolic_cspca_probability"] = cspca_scores
        for index, name in enumerate(("benign", "non_cspca", "cspca")):
            table[f"hyperbolic_class_probability_{name}"] = (
                hyper_state_probabilities[:, index]
            )
        if model.hyperbolic_needle_encoder.target_mode != "three_way":
            target_names = model.hyperbolic_needle_encoder.target_names
            for index, grade in enumerate(target_values):
                label = target_names[index].lower().replace("-", "_")
                table[f"hyperbolic_target_probability_{label}"] = (
                    hyper_probabilities[:, index]
                )
        for prefix, scores, labels in (
            ("hyperbolic_pca", pca_scores, table.label.values),
            (
                "hyperbolic_cspca",
                cspca_scores,
                (table.grade_group.values > 2).astype(int),
            ),
        ):
            endpoint_metrics = calculate_metrics(scores, labels, log_images=False)
            metrics.update(
                {f"{prefix}_{key}": value for key, value in endpoint_metrics.items()}
            )
        true_q = table.involvement.values.astype(float)
        q_valid = (
            table.involvement_valid.values.astype(bool)
            if "involvement_valid" in table.columns
            else np.ones_like(true_q, dtype=bool)
        )
        metrics["hyperbolic_involvement_mae"] = float(
            np.abs(predicted_q[q_valid] - true_q[q_valid]).mean()
        )
        metrics["hyperbolic_involvement_pearson"] = float(
            np.corrcoef(predicted_q[q_valid], true_q[q_valid])[0, 1]
        )
        metrics["hyperbolic_involvement_spearman"] = float(
            table.loc[q_valid].assign(_hyper_q=predicted_q[q_valid])[
                ["_hyper_q", "involvement"]
            ]
            .corr(method="spearman").iloc[0, 1]
        )
        metrics["hyperbolic_benign_radius_mean"] = float(
            radii[table.label.values == 0].mean()
        )
        metrics["hyperbolic_cancer_radius_mean"] = float(
            radii[table.label.values == 1].mean()
        )
        metrics["hyperbolic_fusion_scale"] = float(
            model.hyperbolic_needle_encoder.fusion_scale.detach().cpu()
        )
        metrics["hyperbolic_base_logit_scale"] = float(
            model.hyperbolic_base_logit_scale
        )
        metrics["branch_fusion_mode"] = str(model.branch_fusion_mode)
        metrics["branch_fusion_global_weight"] = float(
            model.branch_fusion_global_weight.detach().cpu()
        )
        if needle_selection_entropies:
            selection_entropy = torch.cat(needle_selection_entropies).numpy()
            selection_cohesion = torch.cat(needle_selection_cohesions).numpy()
            active_fraction = torch.cat(
                needle_selection_active_fractions
            ).numpy()
            table["hyperbolic_hard_selected_fraction"] = active_fraction
            metrics["hyperbolic_selection_entropy"] = float(
                selection_entropy.mean()
            )
            metrics["hyperbolic_selection_total_variation"] = float(
                selection_cohesion.mean()
            )
            metrics["hyperbolic_hard_selected_fraction_mean"] = float(
                active_fraction.mean()
            )
    if pathology_transport_state_probabilities:
        state_probabilities = torch.cat(
            pathology_transport_state_probabilities
        ).numpy()
        path_grade_probabilities = torch.cat(
            pathology_transport_grade_probabilities
        ).numpy()
        path_severity = torch.cat(pathology_transport_severities).numpy()
        path_pca_scores = state_probabilities[:, 1:].sum(1)
        path_cspca_scores = state_probabilities[:, 2]
        table["pathology_transport_pca_probability"] = path_pca_scores
        table["pathology_transport_cspca_probability"] = path_cspca_scores
        table["pathology_transport_predicted_grade"] = path_grade_probabilities.argmax(1)
        table["pathology_transport_predicted_severity"] = path_severity
        for grade in range(6):
            table[f"pathology_transport_grade_probability_{grade}"] = (
                path_grade_probabilities[:, grade]
            )
        for prefix, scores, labels in (
            ("pathology_transport_pca", path_pca_scores, table.label.values),
            (
                "pathology_transport_cspca",
                path_cspca_scores,
                (table.grade_group.values > 2).astype(int),
            ),
        ):
            endpoint_metrics = calculate_metrics(scores, labels, log_images=False)
            metrics.update(
                {f"{prefix}_{key}": value for key, value in endpoint_metrics.items()}
            )
        metrics["pathology_transport_endpoint_mean_auc"] = float(
            0.5
            * (
                metrics["pathology_transport_pca_core_auc"]
                + metrics["pathology_transport_cspca_core_auc"]
            )
        )
        true_grades = table.grade_group.values.astype(int)
        predicted_grades = path_grade_probabilities.argmax(1)
        metrics["pathology_transport_grade_mae"] = float(
            np.abs(predicted_grades - true_grades).mean()
        )
        metrics["pathology_transport_grade_qwk"] = float(
            cohen_kappa_score(true_grades, predicted_grades, weights="quadratic")
        )
        metrics["pathology_transport_severity_grade_spearman"] = float(
            table.assign(_path_severity=path_severity)[
                ["_path_severity", "grade_group"]
            ].corr(method="spearman").iloc[0, 1]
        )
        transport_head = model.hyperbolic_needle_encoder.pathology_transport_head
        mapping = data["pathology_transport_mapping"].detach().cpu().float()
        flattened_mapping = mapping.flatten(0, 1)
        mapped_embeddings = data[
            "pathology_transport_prototype_embeddings"
        ].detach().cpu().float().flatten(0, 1)
        singular_values = torch.linalg.svdvals(
            mapped_embeddings - mapped_embeddings.mean(0, keepdim=True)
        )
        normalized_singular_values = singular_values / singular_values.sum().clamp_min(1e-8)
        metrics["pathology_transport_mapping_confidence"] = float(
            flattened_mapping.max(1).values.mean()
        )
        metrics["pathology_transport_mapping_effective_anchors"] = float(
            torch.exp(
                -(
                    flattened_mapping.clamp_min(1e-8)
                    * flattened_mapping.clamp_min(1e-8).log()
                ).sum(1)
            ).mean()
        )
        metrics["pathology_transport_mapping_unique_top1"] = float(
            flattened_mapping.argmax(1).unique().numel()
        )
        metrics["pathology_transport_prototype_effective_rank"] = float(
            torch.exp(
                -(normalized_singular_values * normalized_singular_values.clamp_min(1e-8).log()).sum()
            )
        )
        metrics["pathology_transport_fusion_scale"] = float(
            transport_head.fusion_scale.detach().cpu()
        )
    if semantic_anchor_probabilities:
        anchor_probabilities = torch.cat(semantic_anchor_probabilities).numpy()
        anchor_pca_scores = anchor_probabilities[:, 1:].sum(1)
        anchor_cspca_scores = anchor_probabilities[:, 2]
        table["semantic_anchor_pca_probability"] = anchor_pca_scores
        table["semantic_anchor_cspca_probability"] = anchor_cspca_scores
        for prefix, scores, labels in (
            ("semantic_anchor_pca", anchor_pca_scores, table.label.values),
            (
                "semantic_anchor_cspca",
                anchor_cspca_scores,
                (table.grade_group.values > 2).astype(int),
            ),
        ):
            endpoint_metrics = calculate_metrics(scores, labels, log_images=False)
            metrics.update(
                {f"{prefix}_{key}": value for key, value in endpoint_metrics.items()}
            )
        metrics["semantic_anchor_endpoint_mean_auc"] = float(
            0.5
            * (
                metrics["semantic_anchor_pca_core_auc"]
                + metrics["semantic_anchor_cspca_core_auc"]
            )
        )
        metrics["semantic_anchor_fusion_scale"] = float(
            model.semantic_anchor_fusion_scale.detach().cpu()
        )
    if uncertainty_values:
        uncertainty = torch.cat(uncertainty_values).numpy()
        table["uncertainty_probability"] = uncertainty
        predicted = prototype_scores >= 0.5
        correct = predicted == table.label.values
        metrics["uncertainty_mean"] = float(uncertainty.mean())
        metrics["uncertainty_correct_mean"] = float(uncertainty[correct].mean())
        metrics["uncertainty_error_mean"] = (
            float(uncertainty[~correct].mean()) if bool((~correct).any()) else float("nan")
        )
    metrics["invalid_region_cores_filtered"] = skipped
    if "patient_id" in table.columns:
        patient_table = table.assign(
            prototype_target=prototype_labels,
            prototype_score=prototype_scores,
        ).groupby("patient_id").agg(
            prototype_target=("prototype_target", "max"),
            prototype_score_max=("prototype_score", "max"),
            prototype_score_mean=("prototype_score", "mean"),
        )
        for pooling in ("max", "mean"):
            patient_metrics = calculate_metrics(
                patient_table[f"prototype_score_{pooling}"].values,
                patient_table.prototype_target.values,
                log_images=False,
            )
            metrics.update(
                {
                    f"prototype_{key}_patient_{pooling}": value
                    for key, value in patient_metrics.items()
                }
            )
    metrics.update(_prototype_stats(model))
    return json_safe_metrics(metrics), table


def _balanced_weights(labels: torch.Tensor, num_classes: int = 2) -> torch.Tensor:
    counts = torch.bincount(labels, minlength=num_classes).float()
    weights = labels.numel() / (num_classes * counts.clamp_min(1.0))
    weights[counts == 0] = 0
    return weights.to(labels.device)


def _save(
    path: Path,
    model: ProtoProstate,
    cfg: DictConfig,
    metrics: dict[str, Any] | None = None,
    projection_info: list[dict[str, Any]] | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    state = model.state_dict()
    omit_frozen_encoder = bool(
        cfg.get("omit_frozen_encoder_from_intermediate_checkpoints", False)
    ) and path.name != "best_interpretable.pth"
    if omit_frozen_encoder:
        if any(parameter.requires_grad for parameter in model.image_encoder.parameters()):
            raise RuntimeError(
                "Cannot omit a trainable image encoder from an intermediate checkpoint."
            )
        state = {
            key: value
            for key, value in state.items()
            if not key.startswith("image_encoder.")
        }
    torch.save(
        {
            "model": state,
            "args": OmegaConf.to_container(cfg, resolve=True),
            "metrics": metrics or {},
            "projection_info": projection_info or [],
            "frozen_encoder_omitted": omit_frozen_encoder,
        },
        path,
    )


def _restore_model(path: Path, model: ProtoProstate) -> dict[str, Any]:
    checkpoint = torch.load(path, map_location=model.device, weights_only=False)
    if bool(checkpoint.get("frozen_encoder_omitted", False)):
        incompatible = model.load_state_dict(checkpoint["model"], strict=False)
        unexpected = list(incompatible.unexpected_keys)
        non_encoder_missing = [
            key
            for key in incompatible.missing_keys
            if not key.startswith("image_encoder.")
        ]
        if unexpected or non_encoder_missing:
            raise RuntimeError(
                "Intermediate checkpoint is incomplete beyond its frozen encoder: "
                f"missing={non_encoder_missing}, unexpected={unexpected}"
            )
    else:
        model.load_state_dict(checkpoint["model"], strict=True)
    return checkpoint


def _wandb_log(run, prefix: str, values: dict[str, Any], global_epoch: int) -> None:
    if run is None:
        return
    numeric = {
        f"{prefix}/{key}": value
        for key, value in values.items()
        if isinstance(value, (float, int, bool))
    }
    numeric["global_epoch"] = global_epoch
    run.log(numeric)


def _project(
    model: ProtoProstate,
    projection_loader,
    device: torch.device,
    output_dir: Path,
    event_name: str,
) -> list[dict[str, Any]]:
    path = output_dir / "projection_events" / f"{event_name}.json"
    records = project_prototypes(
        model,
        projection_loader,
        device,
        path,
        min_cancer_involvement=float(
            model.prototype_layer.coverage_min_cancer_involvement
        ),
    )
    logging.info("Projected all prototypes for %s and refreshed anchors.", event_name)
    return records


def _projection_stats(records: list[dict[str, Any]]) -> dict[str, float]:
    present = [record for record in records if not record.get("missing")]
    stats = {
        "projection_missing_prototypes": float(len(records) - len(present)),
        "projection_unique_cores": float(len({record["core_id"] for record in present})),
    }
    for prototype_class, name in ((0, "benign"), (1, "cancer")):
        class_records = [
            record
            for record in present
            if int(record["prototype_class"]) == prototype_class
        ]
        stats[f"projection_unique_{name}_cores"] = float(
            len({record["core_id"] for record in class_records})
        )
        if class_records:
            stats[f"projection_mean_{name}_involvement"] = float(
                sum(record.get("involvement", 0.0) for record in class_records)
                / len(class_records)
            )
    for prototype_class in sorted(
        {int(record["prototype_class"]) for record in present}
    ):
        class_records = [
            record
            for record in present
            if int(record["prototype_class"]) == prototype_class
        ]
        unique_cores = len({record["core_id"] for record in class_records})
        stats[f"projection_class_{prototype_class}_prototypes"] = float(
            len(class_records)
        )
        stats[f"projection_class_{prototype_class}_unique_cores"] = float(
            unique_cores
        )
        stats[f"projection_class_{prototype_class}_duplicate_cores"] = float(
            len(class_records) - unique_cores
        )
    for grade_group in (0, 2, 3, 4, 5):
        grade_records = [
            record
            for record in present
            if int(record.get("prototype_grade_target", -1)) == grade_group
        ]
        if grade_records:
            stats[f"projection_grade_{grade_group}_prototypes"] = float(
                len(grade_records)
            )
            stats[f"projection_grade_{grade_group}_unique_cores"] = float(
                len({record["core_id"] for record in grade_records})
            )
    return stats


def _train_prototype_stage(
    stage: str,
    epochs: int,
    model: ProtoProstate,
    train_loader,
    val_loader,
    projection_loader,
    criterion,
    device: torch.device,
    cfg: DictConfig,
    output_dir: Path,
    history: dict[str, Any],
    run,
    global_epoch: int,
    projection_info: list[dict[str, Any]],
) -> tuple[int, list[dict[str, Any]]]:
    if epochs <= 0:
        return global_epoch, projection_info
    _set_trainable(
        model,
        stage,
        bool(cfg.unfreeze_adapters),
        bool(cfg.get("hyperbolic_only_training", False))
        and stage in {"warmup", "joint"},
        str(cfg.get("encoder_unfreeze_mode", "none")),
        bool(cfg.get("pathology_transport_only_training", False))
        and stage in {"warmup", "joint"},
    )
    optimizer = _optimizer(model, cfg, stage)
    best_path = output_dir / f"best_{stage}.pth"
    selection_metric = str(cfg.selection_metric)
    best_score = -float("inf")
    stale_epochs = 0
    patience = int(cfg.get("early_stopping_patience", epochs))
    project_every = int(
        cfg.get(
            f"{stage}_projection_every_n_epochs",
            cfg.get("projection_every_n_epochs", 0),
        )
    )

    for epoch in range(epochs):
        if stage == "slots":
            fraction = epoch / max(epochs - 1, 1)
            start = float(cfg.slot_temperature_start)
            end = float(cfg.slot_temperature_end)
            model.prototype_layer.set_slot_temperature(start + fraction * (end - start))

        train_metrics = train_epoch(
            model, train_loader, criterion, optimizer, device, cfg, stage, epoch
        )
        history[f"{stage}/{epoch}/train"] = train_metrics
        _wandb_log(run, f"{stage}/train", train_metrics, global_epoch)

        should_project = (
            stage in {"warmup", "joint"}
            and project_every > 0
            and (epoch + 1) % project_every == 0
        )
        if should_project:
            before_metrics, _ = evaluate(model, val_loader, device, cfg)
            projection_info = _project(
                model,
                projection_loader,
                device,
                output_dir,
                f"{stage}_{epoch:02d}",
            )
            after_metrics, _ = evaluate(model, val_loader, device, cfg)
            tracked = str(cfg.projection_tracked_metric)
            projection_drop = float(
                before_metrics.get(tracked, float("nan"))
                - after_metrics.get(tracked, float("nan"))
            )
            after_metrics[f"projection_drop_{tracked}"] = projection_drop
            after_metrics.update(_projection_stats(projection_info))
            history[f"{stage}/{epoch}/pre_projection"] = before_metrics
            history[f"{stage}/{epoch}/val"] = after_metrics
            _wandb_log(run, f"{stage}/pre_projection", before_metrics, global_epoch)
            _wandb_log(run, f"{stage}/val", after_metrics, global_epoch)
            val_metrics = after_metrics
        else:
            val_metrics, _ = evaluate(model, val_loader, device, cfg)
            history[f"{stage}/{epoch}/val"] = val_metrics
            _wandb_log(run, f"{stage}/val", val_metrics, global_epoch)

        score = float(val_metrics.get(selection_metric, -float("inf")))
        logging.info(
            "%s epoch %d: train_loss=%.6f %s=%.6f",
            stage,
            epoch,
            train_metrics.get("loss", float("nan")),
            selection_metric,
            score,
        )
        _save(output_dir / "latest.pth", model, cfg, val_metrics, projection_info)
        if (not best_path.exists()) or score > best_score + float(
            cfg.get("early_stopping_min_delta", 0.0)
        ):
            best_score = score
            stale_epochs = 0
            _save(best_path, model, cfg, val_metrics, projection_info)
        else:
            stale_epochs += 1
        global_epoch += 1
        if stale_epochs >= patience:
            logging.info("Early stopping %s after %d stale epochs.", stage, stale_epochs)
            break

    best_checkpoint = _restore_model(best_path, model)
    return global_epoch, best_checkpoint.get("projection_info", projection_info)


def train_convex_last_layers(
    model: ProtoProstate,
    train_loader,
    val_loader,
    device: torch.device,
    cfg: DictConfig,
    output_dir: Path,
    history: dict[str, Any],
    run,
    global_epoch: int,
    projection_info: list[dict[str, Any]],
) -> int:
    """Optimize only fixed-feature classifiers; preserve q-calibrated evidence."""
    if int(cfg.last_layer_epochs) <= 0:
        logging.info(
            "Skipping last-layer optimization because last_layer_epochs=%d.",
            int(cfg.last_layer_epochs),
        )
        return global_epoch
    _set_trainable(model, "last_layer", False)
    optimizer = _optimizer(model, cfg, "last_layer")
    best_path = output_dir / "best_last_layer.pth"
    best_score = -float("inf")
    stale_epochs = 0
    selection_metric = str(cfg.last_layer_selection_metric)
    patience = int(cfg.get("last_layer_patience", cfg.last_layer_epochs))

    for epoch in range(int(cfg.last_layer_epochs)):
        model.eval()
        model.prototype_classifier.train()
        if model.core_classifier is not None:
            model.core_classifier.train()
        if model.grade_classifier is not None:
            model.grade_classifier.train()
        total = 0.0
        batches = 0
        skipped = 0
        for batch in tqdm(train_loader, desc=f"last-layer {epoch}"):
            original_batch_size = len(batch["bmode"])
            # Run the deployed forward path with autograd.  All representation
            # parameters are already frozen by _set_trainable, so this updates
            # only the selected classifiers while retaining dense, semantic,
            # and future fusion branches.  The previous hand reconstruction
            # silently skipped V20's actual final heads.
            data = forward_batch(model, batch, device)
            if data is None:
                skipped += original_batch_size
                continue
            skipped += int(data.get("invalid_region_cores_filtered", 0))
            batch = data
            activations = data["prototype_activations"]
            classification = data
            labels = prototype_target_labels(model, data).to(device)
            cspca_labels = (data["grade_group"].to(device) > 2).long()
            pca_logits = classification["prototype_class_logits"]
            cspca_logits = (
                pca_logits
                if model.core_classifier is None
                else model.core_classifier(activations)
            )
            grade_logits = (
                model.grade_classifier(activations)
                if model.grade_classifier is not None
                else None
            )
            objective = F.cross_entropy(
                pca_logits,
                labels,
                weight=_balanced_weights(labels, model.num_classes),
            )
            if model.classification_target in {"three_way", "isup"}:
                pca_binary_logits = torch.stack(
                    (
                        pca_logits[:, 0],
                        torch.logsumexp(pca_logits[:, 1:], dim=1),
                    ),
                    dim=1,
                )
                cspca_binary_logits = (
                    torch.stack(
                        (
                            torch.logsumexp(pca_logits[:, :3], dim=1),
                            torch.logsumexp(pca_logits[:, 3:], dim=1),
                        ),
                        dim=1,
                    )
                    if model.classification_target == "isup"
                    else torch.stack(
                        (
                            torch.logsumexp(pca_logits[:, :2], dim=1),
                            pca_logits[:, 2],
                        ),
                        dim=1,
                    )
                )
                objective = objective + float(
                    cfg.prototype_loss_weights.get("pca_endpoint_ce", 0.0)
                ) * F.cross_entropy(
                    pca_binary_logits,
                    batch["label"].long().to(device),
                    weight=_balanced_weights(batch["label"].long().to(device)),
                )
                objective = objective + float(
                    cfg.prototype_loss_weights.get("cspca_endpoint_ce", 0.0)
                ) * F.cross_entropy(
                    cspca_binary_logits,
                    cspca_labels,
                    weight=_balanced_weights(cspca_labels),
                )
                q_valid = batch.get("involvement_valid")
                q_valid = (
                    q_valid.bool().to(device)
                    if q_valid is not None
                    else torch.ones_like(cspca_labels, dtype=torch.bool)
                )
                high_keep = (
                    batch["label"].long().to(device) == 0
                ) | (
                    q_valid
                    & (batch["involvement"].float().to(device) >= 0.4)
                )
                high_objective = (
                    0.5
                    * (
                        F.cross_entropy(
                            pca_binary_logits[high_keep],
                            batch["label"].long().to(device)[high_keep],
                            weight=_balanced_weights(
                                batch["label"].long().to(device)[high_keep]
                            ),
                        )
                        + F.cross_entropy(
                            cspca_binary_logits[high_keep],
                            cspca_labels[high_keep],
                            weight=_balanced_weights(cspca_labels[high_keep]),
                        )
                    )
                    if bool(high_keep.any())
                    else pca_logits.new_zeros(())
                )
                objective = objective + float(
                    cfg.prototype_loss_weights.get(
                        "high_involvement_endpoint_ce", 0.0
                    )
                ) * high_objective
            if model.core_classifier is not None:
                objective = objective + F.cross_entropy(
                    cspca_logits,
                    cspca_labels,
                    weight=_balanced_weights(cspca_labels),
                )
            if "clinical_logit" in classification or "prompt_class_logits" in classification:
                objective = objective + float(
                    cfg.prototype_loss_weights.get("image_only_ce", 0.0)
                ) * F.cross_entropy(
                    classification["prototype_class_logits_image"],
                    labels,
                    weight=_balanced_weights(labels, model.num_classes),
                )
            if "prompt_class_logits" in classification:
                objective = objective + float(
                    cfg.prototype_loss_weights.get("conditioned_image_ce", 0.0)
                ) * F.cross_entropy(
                    classification["prototype_class_logits_conditioned"],
                    labels,
                    weight=_balanced_weights(labels, model.num_classes),
                )
                objective = objective + float(
                    cfg.prototype_loss_weights.get("prompt_only_ce", 0.0)
                ) * F.cross_entropy(
                    classification["prompt_class_logits"],
                    labels,
                    weight=_balanced_weights(labels, model.num_classes),
                )
            l1 = (
                model.prototype_classifier.weight.abs().sum()
                + (
                    model.core_classifier.weight.abs().sum()
                    if model.core_classifier is not None
                    else 0.0
                )
            )
            objective = objective + float(cfg.last_layer_l1) * l1
            if model.prompt_conditioner is not None:
                objective = objective + float(cfg.last_layer_l1) * (
                    model.prompt_conditioner.classifier.weight.abs().sum()
                )
            if grade_logits is not None:
                grade_data = {
                    "grade_group_logits": grade_logits,
                    "grade_group": batch["grade_group"],
                    "prototype_activations": activations,
                }
                grade_objective, _ = _ordinal_grade_loss(
                    model, grade_data, cfg, "fine", epoch
                )
                objective = objective + grade_objective
                objective = objective + float(cfg.last_layer_l1) * (
                    model.grade_classifier.weight.abs().sum()
                )
            semantic_objective, _ = semantic_anchor_regularization(
                model, data, cfg, "last_layer"
            )
            objective = objective + semantic_objective
            optimizer.zero_grad(set_to_none=True)
            objective.backward()
            optimizer.step()
            total += float(objective.detach())
            batches += 1
            if cfg.get("debug", False) and batches >= 3:
                break

        train_metrics = {
            "objective": total / max(batches, 1),
            "invalid_region_cores_filtered": skipped,
        }
        val_metrics, _ = evaluate(model, val_loader, device, cfg)
        history[f"last_layer/{epoch}/train"] = train_metrics
        history[f"last_layer/{epoch}/val"] = val_metrics
        _wandb_log(run, "last_layer/train", train_metrics, global_epoch)
        _wandb_log(run, "last_layer/val", val_metrics, global_epoch)
        score = float(val_metrics.get(selection_metric, -float("inf")))
        logging.info(
            "Last-layer epoch %d objective %.6f %s=%.6f",
            epoch,
            train_metrics["objective"],
            selection_metric,
            score,
        )
        if (not best_path.exists()) or score > best_score + float(
            cfg.get("early_stopping_min_delta", 0.0)
        ):
            best_score = score
            stale_epochs = 0
            _save(best_path, model, cfg, val_metrics, projection_info)
        else:
            stale_epochs += 1
        global_epoch += 1
        if stale_epochs >= patience:
            logging.info("Early stopping last layer after %d stale epochs.", stale_epochs)
            break

    _restore_model(best_path, model)
    return global_epoch


def _write_results_summary(
    output_dir: Path,
    pre: dict[str, Any],
    post: dict[str, Any],
    final: dict[str, Any],
    tracked: str,
) -> None:
    if os.environ.get("PROTOPROSTATE_NO_MARKDOWN") == "1":
        return
    drop = float(pre.get(tracked, float("nan")) - post.get(tracked, float("nan")))
    three_way = "prototype_pca_core_auc" in final
    if three_way:
        header = (
            "| Stage | Prototype cancer AUROC | Cancer Sens@80%Spec | "
            "Prototype csPCa AUROC | csPCa Sens@80%Spec | Endpoint mean AUROC | "
            "Cancer heatmap AUROC |"
        )
        divider = "|---|---:|---:|---:|---:|---:|---:|"
    else:
        header = (
            "| Stage | Prototype csPCa AUROC | csPCa Sens@80%Spec | "
            "Cancer heatmap AUROC | Thresholded heatmap AUROC | "
            "Involvement Pearson | Patient-max csPCa AUROC |"
        )
        divider = "|---|---:|---:|---:|---:|---:|---:|"
    lines = [
        "# ProtoProstate fold result",
        "",
        header,
        divider,
    ]
    for name, metrics in (("Pre-projection", pre), ("Post-projection", post), ("Final", final)):
        if three_way:
            lines.append(
                f"| {name} | {metrics.get('prototype_pca_core_auc', float('nan')):.4f} | "
                f"{metrics.get('prototype_pca_sens_at_80_spe', float('nan')):.4f} | "
                f"{metrics.get('prototype_cspca_core_auc', float('nan')):.4f} | "
                f"{metrics.get('prototype_cspca_sens_at_80_spe', float('nan')):.4f} | "
                f"{metrics.get('prototype_endpoint_mean_auc', float('nan')):.4f} | "
                f"{metrics.get('core_auc', float('nan')):.4f} |"
            )
        else:
            lines.append(
                f"| {name} | {metrics.get('prototype_core_auc', float('nan')):.4f} | "
                f"{metrics.get('prototype_sens_at_80_spe', float('nan')):.4f} | "
                f"{metrics.get('core_auc', float('nan')):.4f} | "
                f"{metrics.get('core_auc_thresholded', float('nan')):.4f} | "
                f"{metrics.get('thresholded_involvement_correlation_pearson', float('nan')):.4f} | "
                f"{metrics.get('prototype_core_auc_patient_max', float('nan')):.4f} |"
            )
    lines.extend(
        [
            "",
            f"Projection drop ({tracked}): **{drop:.4f}**",
            "",
            "Prototype images are under `prototype_visualizations/` after the Slurm script finishes.",
        ]
    )
    (output_dir / "RESULTS.md").write_text("\n".join(lines) + "\n")


def _resume_after_projection(
    model: ProtoProstate,
    loaders,
    device: torch.device,
    cfg: DictConfig,
    output_dir: Path,
    run,
) -> None:
    """Resume only convex last-layer fitting from a projected checkpoint.

    Prototype training and the exhaustive nearest-training-token projection are
    intentionally skipped. This path is used when those expensive stages
    completed successfully but final classifier fitting or report generation
    was interrupted.
    """
    resume_path = Path(str(cfg.resume_post_projection_checkpoint))
    if not resume_path.exists():
        raise FileNotFoundError(
            f"Missing post-projection resume checkpoint: {resume_path}"
        )
    checkpoint = _restore_model(resume_path, model)
    projection_info = checkpoint.get("projection_info", [])
    post_metrics = checkpoint.get("metrics", {})
    if not post_metrics:
        post_metrics, _ = evaluate(model, loaders["val"], device, cfg)

    pre_path = Path(
        str(
            cfg.get(
                "resume_pre_projection_checkpoint",
                output_dir / "pre_final_projection.pth",
            )
        )
    )
    if pre_path.exists():
        pre_checkpoint = torch.load(pre_path, map_location="cpu", weights_only=False)
        pre_metrics = pre_checkpoint.get("metrics", {})
    else:
        logging.warning(
            "Pre-projection checkpoint %s is missing; using post-projection "
            "metrics as the report baseline.",
            pre_path,
        )
        pre_metrics = dict(post_metrics)

    tracked = str(cfg.projection_tracked_metric)
    history: dict[str, Any] = {
        "resumed_from": str(resume_path),
        "pre_projection": pre_metrics,
        "post_projection": post_metrics,
        "projection_drop": {
            tracked: float(
                pre_metrics.get(tracked, float("nan"))
                - post_metrics.get(tracked, float("nan"))
            )
        },
    }
    logging.info(
        "Resumed projected model from %s; skipping warm-up, joint, slots, "
        "pruning, and projection.",
        resume_path,
    )
    _wandb_log(run, "resume/pre_projection", pre_metrics, 0)
    _wandb_log(run, "resume/post_projection", post_metrics, 0)

    global_epoch = train_convex_last_layers(
        model,
        loaders["train"],
        loaders["val"],
        device,
        cfg,
        output_dir,
        history,
        run,
        0,
        projection_info,
    )
    final_metrics, final_table = evaluate(model, loaders["val"], device, cfg)
    history["final"] = final_metrics
    _save(
        output_dir / "best_interpretable.pth",
        model,
        cfg,
        final_metrics,
        projection_info,
    )
    final_table.to_csv(output_dir / "val_predictions.csv", index=False)
    (output_dir / "metrics.json").write_text(
        json.dumps(json_safe_metrics(final_metrics), indent=2)
    )
    (output_dir / "metrics_pre_projection.json").write_text(
        json.dumps(json_safe_metrics(pre_metrics), indent=2)
    )
    (output_dir / "metrics_post_projection.json").write_text(
        json.dumps(json_safe_metrics(post_metrics), indent=2)
    )
    (output_dir / "training_report.json").write_text(
        json.dumps(history, indent=2)
    )
    _write_results_summary(
        output_dir, pre_metrics, post_metrics, final_metrics, tracked
    )
    _wandb_log(run, "final", final_metrics, global_epoch)
    if run is not None:
        run.summary.update(final_metrics)
        run.summary[f"projection_drop_{tracked}"] = history["projection_drop"][
            tracked
        ]
        if os.environ.get("PROTOPROSTATE_NO_MARKDOWN") != "1":
            wandb.save(str(output_dir / "RESULTS.md"), base_path=str(output_dir), policy="now")
        wandb.save(
            str(output_dir / "training_report.json"),
            base_path=str(output_dir),
            policy="now",
        )


def main(cfg: DictConfig) -> None:
    set_global_seed(int(cfg.seed))
    fold = int(cfg.data.fold)
    cfg.name = str(cfg.name).format(fold=fold)
    if cfg.get("supervised_encoder_checkpoint"):
        cfg.supervised_encoder_checkpoint = os.environ.get(
            "PROST_RL_SUPERVISED_CHECKPOINT",
            str(cfg.supervised_encoder_checkpoint).format(fold=fold),
        )
    else:
        cfg.supervised_encoder_checkpoint = None
    if cfg.get("supervised_endpoint_checkpoint"):
        cfg.supervised_endpoint_checkpoint = os.environ.get(
            "PROST_RL_SUPERVISED_ENDPOINT_CHECKPOINT",
            str(cfg.supervised_endpoint_checkpoint).format(fold=fold),
        )
    else:
        cfg.supervised_endpoint_checkpoint = None
    cfg.checkpoint_dir = str(cfg.checkpoint_dir).format(fold=fold)
    if cfg.get("initial_model_checkpoint"):
        cfg.initial_model_checkpoint = str(cfg.initial_model_checkpoint).format(
            fold=fold
        )
    if cfg.get("wandb_group"):
        cfg.wandb_group = str(cfg.wandb_group).format(fold=fold)
    for resume_key in (
        "resume_post_projection_checkpoint",
        "resume_pre_projection_checkpoint",
    ):
        if cfg.get(resume_key):
            cfg[resume_key] = str(cfg[resume_key]).format(fold=fold)
    output_dir = Path(cfg.checkpoint_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        force=True,
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(output_dir / "training.log", mode="a"),
        ],
    )
    (output_dir / "resolved_config.yaml").write_text(OmegaConf.to_yaml(cfg, resolve=True))
    device = _device(cfg)

    run = None
    if bool(cfg.get("wandb_enabled", True)):
        wandb_dir = output_dir / "wandb"
        wandb_dir.mkdir(parents=True, exist_ok=True)
        wandb_kwargs = dict(OmegaConf.to_container(cfg.get("wandb", {}), resolve=True))
        wandb_kwargs = {key: value for key, value in wandb_kwargs.items() if value is not None}
        run = wandb.init(
            project=str(cfg.get("project", "protoprostate")),
            group=str(cfg.get("wandb_group", "medsam-prototypes")),
            name=cfg.name,
            dir=str(wandb_dir),
            config=OmegaConf.to_container(cfg, resolve=True),
            **wandb_kwargs,
        )
        wandb.define_metric("global_epoch")
        wandb.define_metric("*", step_metric="global_epoch")
        cfg.wandb_url = run.url
        (output_dir / "wandb_url.txt").write_text(f"{run.url}\n")
        (output_dir / "resolved_config.yaml").write_text(
            OmegaConf.to_yaml(cfg, resolve=True)
        )
        logging.info("W&B run: %s", run.url)

    try:
        model = create_model(cfg.model, **cfg.model_kw)
        if not isinstance(model, ProtoProstate):
            raise TypeError(f"Expected ProtoProstate, got {type(model)!r}")
        if cfg.get("supervised_encoder_checkpoint"):
            init_checkpoint = Path(str(cfg.supervised_encoder_checkpoint))
            if not init_checkpoint.exists():
                raise FileNotFoundError(
                    f"Missing supervised Prost-RL checkpoint: {init_checkpoint}"
                )
            model.load_supervised_encoder_checkpoint(str(init_checkpoint), strict=True)
        else:
            logging.info(
                "Using the backbone-native checkpoint declared in model_kw; "
                "no Prost-RL/MedSAM encoder checkpoint will be loaded."
            )
        if cfg.get("supervised_endpoint_checkpoint"):
            endpoint_checkpoint = Path(str(cfg.supervised_endpoint_checkpoint))
            if not endpoint_checkpoint.exists():
                raise FileNotFoundError(
                    "Missing supervised Prost-RL endpoint checkpoint: "
                    f"{endpoint_checkpoint}"
                )
            if cfg.get("supervised_encoder_checkpoint") and (
                endpoint_checkpoint.resolve() == init_checkpoint.resolve()
            ):
                logging.info(
                    "Supervised endpoint components were already loaded with the "
                    "MedSAM encoder checkpoint."
                )
            else:
                model.load_supervised_endpoint_checkpoint(
                    str(endpoint_checkpoint)
                )
        if cfg.get("initial_model_checkpoint"):
            warm_start_path = Path(str(cfg.initial_model_checkpoint))
            if not warm_start_path.exists():
                raise FileNotFoundError(
                    f"Missing full-model warm start: {warm_start_path}"
                )
            warm_start = torch.load(
                warm_start_path, map_location="cpu", weights_only=False
            )
            warm_state = model._checkpoint_state(warm_start)
            incompatible = model.load_state_dict(warm_state, strict=False)
            unexpected = list(incompatible.unexpected_keys)
            non_hyper_missing = [
                key
                for key in incompatible.missing_keys
                if not key.startswith("hyperbolic_needle_encoder.")
                and not key.startswith("pathology_geometry_projector.")
                and not key.startswith("semantic_anchor_core_projector.")
                and not key.startswith("semantic_anchor_prototype_projector.")
                and key != "semantic_anchor_log_temperature"
                and key != "semantic_anchor_fusion_logit"
                and key != "semantic_anchor_bank"
                and key != "semantic_anchor_states"
            ]
            if unexpected or non_hyper_missing:
                raise RuntimeError(
                    "V8 warm start is architecturally incompatible: "
                    f"missing={non_hyper_missing}, unexpected={unexpected}"
                )
            logging.info(
                "Warm-started ProtoProstate model from %s; initialized %d new "
                "parameters belonging to newly enabled modules.",
                warm_start_path,
                len(incompatible.missing_keys),
            )
        model.to(device)
        logging.info(
            "The configured image encoder uses a dynamic dense feature grid; "
            "%dx%d inputs are validated at runtime against model_kw.feature_dim.",
            int(cfg.data.image_size),
            int(cfg.data.image_size),
        )

        loaders = get_dataloaders(cfg.data)
        projection_data_cfg = OmegaConf.create(
            OmegaConf.to_container(cfg.data, resolve=True)
        )
        projection_data_cfg.augmentations = "none"
        # Projection is an exhaustive nearest-token search. Training may use
        # a weighted sampler, but projecting from a with-replacement sample
        # would omit real cores and make the result seed-dependent.
        projection_data_cfg.center_balanced_sampling = False
        projection_data_cfg.grade_patient_balanced_sampling = False
        projection_loaders = get_dataloaders(projection_data_cfg, mode="train")
        _set_fold_grade_weights(cfg, loaders["train"], model)
        _set_ordinal_frequency_weights(cfg, loaders["train"], model)
        (output_dir / "resolved_config.yaml").write_text(
            OmegaConf.to_yaml(cfg, resolve=True)
        )
        if run is not None:
            run.config.update(
                {
                    "grade_pos_weights": list(cfg.get("grade_pos_weights", [])),
                    "grade_threshold_positive_counts": dict(
                        cfg.get("grade_threshold_positive_counts", {})
                    ),
                },
                allow_val_change=True,
            )
        if cfg.get("resume_post_projection_checkpoint"):
            _resume_after_projection(
                model, loaders, device, cfg, output_dir, run
            )
            return
        criterion = build_loss(cfg)
        history: dict[str, Any] = {}
        global_epoch = 0

        # A transfer-only head can preserve an already projected warm start
        # byte-for-byte. Every historical run keeps the default projection.
        if bool(cfg.get("skip_prototype_projection", False)):
            projection_info = []
            logging.info(
                "Skipping prototype projection as explicitly configured; the "
                "warm-start prototype banks remain unchanged."
            )
        else:
            # Start on the real-token manifold instead of random directions.
            projection_info = _project(
                model,
                projection_loaders["train"],
                device,
                output_dir,
                "initial",
            )
        _wandb_log(run, "initial_projection", _projection_stats(projection_info), 0)
        _save(output_dir / "initial_projection.pth", model, cfg, projection_info=projection_info)

        for stage, epochs in (
            ("warmup", int(cfg.warmup_epochs)),
            ("joint", int(cfg.joint_epochs)),
            ("slots", int(cfg.slot_epochs)),
        ):
            global_epoch, projection_info = _train_prototype_stage(
                stage,
                epochs,
                model,
                loaders["train"],
                loaders["val"],
                projection_loaders["train"],
                criterion,
                device,
                cfg,
                output_dir,
                history,
                run,
                global_epoch,
                projection_info,
            )

        model.prototype_layer.prune_slots(
            threshold=float(cfg.slot_prune_threshold),
            min_slots=int(cfg.min_slots_per_prototype),
        )
        pre_metrics, _ = evaluate(model, loaders["val"], device, cfg)
        history["pre_projection"] = pre_metrics
        _save(output_dir / "pre_final_projection.pth", model, cfg, pre_metrics, projection_info)

        if not bool(cfg.get("skip_prototype_projection", False)):
            projection_info = _project(
                model,
                projection_loaders["train"],
                device,
                output_dir,
                "final",
            )
        (output_dir / "projection_info.json").write_text(
            json.dumps(projection_info, indent=2)
        )
        post_metrics, _ = evaluate(model, loaders["val"], device, cfg)
        post_metrics.update(_projection_stats(projection_info))
        history["post_projection"] = post_metrics
        tracked = str(cfg.projection_tracked_metric)
        history["projection_drop"] = {
            tracked: float(
                pre_metrics.get(tracked, float("nan"))
                - post_metrics.get(tracked, float("nan"))
            )
        }
        _save(output_dir / "post_projection.pth", model, cfg, post_metrics, projection_info)
        _wandb_log(run, "final/pre_projection", pre_metrics, global_epoch)
        _wandb_log(run, "final/post_projection", post_metrics, global_epoch)

        global_epoch = train_convex_last_layers(
            model,
            loaders["train"],
            loaders["val"],
            device,
            cfg,
            output_dir,
            history,
            run,
            global_epoch,
            projection_info,
        )
        final_metrics, final_table = evaluate(model, loaders["val"], device, cfg)
        history["final"] = final_metrics
        _save(
            output_dir / "best_interpretable.pth",
            model,
            cfg,
            final_metrics,
            projection_info,
        )
        final_table.to_csv(output_dir / "val_predictions.csv", index=False)
        (output_dir / "metrics.json").write_text(
            json.dumps(json_safe_metrics(final_metrics), indent=2)
        )
        (output_dir / "metrics_pre_projection.json").write_text(
            json.dumps(json_safe_metrics(pre_metrics), indent=2)
        )
        (output_dir / "metrics_post_projection.json").write_text(
            json.dumps(json_safe_metrics(post_metrics), indent=2)
        )
        (output_dir / "training_report.json").write_text(json.dumps(history, indent=2))
        _write_results_summary(output_dir, pre_metrics, post_metrics, final_metrics, tracked)
        _wandb_log(run, "final", final_metrics, global_epoch)
        if run is not None:
            run.summary.update(final_metrics)
            run.summary[f"projection_drop_{tracked}"] = history["projection_drop"][tracked]
            if os.environ.get("PROTOPROSTATE_NO_MARKDOWN") != "1":
                wandb.save(str(output_dir / "RESULTS.md"), base_path=str(output_dir), policy="now")
            wandb.save(
                str(output_dir / "training_report.json"),
                base_path=str(output_dir),
                policy="now",
            )
    finally:
        if run is not None:
            run.finish()


def _load_config_with_defaults(config_path: Path, seen: set[Path] | None = None):
    """Recursively resolve the repo's lightweight ``defaults_from`` configs."""
    config_path = config_path.resolve()
    seen = set() if seen is None else seen
    if config_path in seen:
        raise ValueError(f"Cyclic defaults_from chain at {config_path}")
    seen.add(config_path)
    config = OmegaConf.load(config_path)
    if "defaults_from" not in config:
        return config
    base_path = config_path.parent / str(config.defaults_from)
    base = _load_config_with_defaults(base_path, seen)
    overrides = OmegaConf.create(OmegaConf.to_container(config, resolve=False))
    del overrides["defaults_from"]
    return OmegaConf.merge(base, overrides)


def cli():
    """research_detector.train --config configs/dinov3.yaml --manifest data/cohort_a.csv --fold 0 --out runs/research/dinov3/fold0
    --backbone-weights runs/adapt/dinov3/fold0/encoder.pth --prostrl-supervised <Prost-RL supervised checkpoint of the fold>
    [--anchor-bank random|<path to a 60x256 .npz bank>] [--set key=value ...]"""
    from .anchors import resolve_bank

    parser = argparse.ArgumentParser(description=cli.__doc__)
    parser.add_argument("--config", "-c", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--fold", type=int, required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--backbone-weights", required=True, help="adapted DINOv3 / V-JEPA 2.1 backbone (protocouncil.adapt)")
    parser.add_argument("--prostrl-supervised", required=True, help="Prost-RL supervised checkpoint of the same fold")
    parser.add_argument("--anchor-bank", default="random", help="'random' (matched random bank) or a bank .npz file")
    parser.add_argument("--set", nargs="*", default=[], help="OmegaConf overrides, e.g. joint_epochs=1")
    args = parser.parse_args()
    config = _load_config_with_defaults(Path(args.config))
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    bank = resolve_bank(args.anchor_bank, out)
    config.data.manifest = str(Path(args.manifest).resolve())
    config.data.fold = args.fold
    config.checkpoint_dir = str(out.resolve())
    config.model_kw.backbone_kw.backbone_cfg.weights = str(Path(args.backbone_weights).resolve())
    config.supervised_endpoint_checkpoint = str(Path(args.prostrl_supervised).resolve())
    config.semantic_anchor_bank_path = config.model_kw.semantic_anchor_bank_path = str(bank)
    config = OmegaConf.merge(config, OmegaConf.from_dotlist(args.set))
    main(config)


if __name__ == "__main__":
    cli()
