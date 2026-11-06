"""Projector-free transfer of six-grade PANDA representation geometry."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import torch
from omegaconf import DictConfig
from torch.nn import functional as F


_TEACHER_CACHE: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}


def load_pathology_manifold_teacher(
    path: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    resolved = str(Path(path).resolve())
    if resolved not in _TEACHER_CACHE:
        archive = np.load(resolved, allow_pickle=False)
        grades = torch.as_tensor(archive["grades"], dtype=torch.long)
        relation = torch.as_tensor(
            archive["cosine_similarity"], dtype=torch.float
        )
        if relation.shape != (len(grades), len(grades)):
            raise ValueError("Pathology grade relation matrix has invalid shape.")
        if not torch.allclose(relation, relation.T, atol=1e-5):
            raise ValueError("Pathology grade relation matrix must be symmetric.")
        if not torch.allclose(
            relation.diag(), torch.ones(len(grades)), atol=1e-4
        ):
            raise ValueError("Pathology grade relation matrix must have unit diagonal.")
        _TEACHER_CACHE[resolved] = grades, relation
        logging.info(
            "Loaded projector-free pathology manifold %s with grades=%s.",
            resolved,
            grades.tolist(),
        )
    return _TEACHER_CACHE[resolved]


def _indices_for_grades(
    labels: torch.Tensor, teacher_grades: torch.Tensor
) -> torch.Tensor:
    matches = labels[:, None] == teacher_grades[None]
    if not bool(matches.any(1).all()):
        missing = labels[~matches.any(1)].unique().tolist()
        raise ValueError(f"Pathology manifold has no target for grades {missing}.")
    return matches.float().argmax(1).long()


def _center_normalize(values: torch.Tensor) -> torch.Tensor:
    values = values.float()
    return F.normalize(values - values.mean(1, keepdim=True), dim=1)


def _balanced_soft_kl(
    logits: torch.Tensor, targets: torch.Tensor, labels: torch.Tensor
) -> torch.Tensor:
    per_sample = F.kl_div(
        F.log_softmax(logits, dim=1), targets, reduction="none"
    ).sum(1)
    unique, counts = labels.unique(return_counts=True)
    weights = torch.ones_like(per_sample)
    for grade, count in zip(unique, counts):
        weights[labels == grade] = len(labels) / (len(unique) * count.float())
    return (per_sample * weights).sum() / weights.sum().clamp_min(1.0)


def _grade_balanced_relation_loss(
    values: torch.Tensor,
    grade_indices: torch.Tensor,
    target_relation: torch.Tensor,
) -> torch.Tensor:
    values = _center_normalize(values)
    similarities = values @ values.T
    unique = grade_indices.unique(sorted=True)
    terms: list[torch.Tensor] = []
    for left_index, left_grade in enumerate(unique):
        left = grade_indices == left_grade
        for right_grade in unique[left_index:]:
            right = grade_indices == right_grade
            mask = left[:, None] & right[None, :]
            if int(left_grade) == int(right_grade):
                mask = mask & ~torch.eye(
                    len(values), dtype=torch.bool, device=values.device
                )
            if bool(mask.any()):
                target = target_relation[left_grade, right_grade]
                terms.append(
                    F.smooth_l1_loss(
                        similarities[mask], target.expand_as(similarities[mask])
                    )
                )
    return torch.stack(terms).mean() if terms else values.new_zeros(())


def _centroid_relation_loss(
    values: torch.Tensor,
    grade_indices: torch.Tensor,
    target_relation: torch.Tensor,
) -> torch.Tensor:
    unique = grade_indices.unique(sorted=True)
    if len(unique) < 2:
        return values.new_zeros(())
    centered = _center_normalize(values)
    centroids = F.normalize(
        torch.stack([centered[grade_indices == grade].mean(0) for grade in unique]),
        dim=1,
    )
    prediction = centroids @ centroids.T
    target = target_relation[unique][:, unique]
    off_diagonal = ~torch.eye(
        len(unique), dtype=torch.bool, device=values.device
    )
    return F.smooth_l1_loss(prediction[off_diagonal], target[off_diagonal])


def _grade_logits_from_prototypes(
    activations: torch.Tensor,
    prototype_grade_indices: torch.Tensor,
    num_grades: int,
    temperature: float,
) -> torch.Tensor:
    logits = []
    scaled = activations.float() / temperature
    for grade in range(num_grades):
        keep = prototype_grade_indices == grade
        if not bool(keep.any()):
            raise ValueError(f"No ultrasound occurrence prototype targets grade {grade}.")
        logits.append(
            torch.logsumexp(scaled[:, keep], dim=1)
            - torch.log(keep.sum().to(scaled.dtype))
        )
    return torch.stack(logits, dim=1)


def pathology_manifold_regularization(
    model,
    data: dict[str, Any],
    cfg: DictConfig,
    stage: str,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Transfer grade relations directly; no trainable alignment head exists."""
    weight = float(cfg.get("pathology_manifold_loss_weight", 0.0))
    if weight <= 0 or stage != "joint":
        zero = data["prototype_activations"].new_zeros(())
        return zero, {}
    if getattr(model, "classification_target", None) != "three_way":
        raise ValueError("Pathology manifold transfer requires three-way classification.")

    teacher_grades, teacher_relation = load_pathology_manifold_teacher(
        str(cfg.pathology_manifold_teacher_path)
    )
    device = data["prototype_activations"].device
    teacher_grades = teacher_grades.to(device)
    teacher_relation = teacher_relation.to(device)
    active_grades = torch.as_tensor(
        list(cfg.pathology_manifold_active_grades), dtype=torch.long, device=device
    )
    active_teacher_indices = _indices_for_grades(active_grades, teacher_grades)
    relation = teacher_relation[active_teacher_indices][:, active_teacher_indices]

    core_grades = data["grade_group"].long().to(device)
    core_grade_indices = _indices_for_grades(core_grades, active_grades)
    prototype_grades = model.prototype_layer.prototype_grade_targets.to(device)
    if bool((prototype_grades < 0).any()):
        raise ValueError(
            "Every occurrence prototype needs a grade target for manifold transfer."
        )
    prototype_grade_indices = _indices_for_grades(prototype_grades, active_grades)

    activations = data["prototype_activations"].float()
    student_temperature = float(
        cfg.get("pathology_manifold_student_temperature", 0.20)
    )
    teacher_temperature = float(
        cfg.get("pathology_manifold_teacher_temperature", 0.15)
    )
    grade_logits = _grade_logits_from_prototypes(
        activations,
        prototype_grade_indices,
        len(active_grades),
        student_temperature,
    )
    teacher_targets = F.softmax(
        relation[core_grade_indices] / teacher_temperature, dim=1
    )
    usage_kl = _balanced_soft_kl(grade_logits, teacher_targets, core_grade_indices)
    usage_accuracy = (
        grade_logits.argmax(1) == core_grade_indices
    ).float().mean()

    occurrence_relation = _grade_balanced_relation_loss(
        activations, core_grade_indices, relation
    )
    direction_relation = _grade_balanced_relation_loss(
        data["hyperbolic_direction"], core_grade_indices, relation
    )
    occurrence_centroids = _centroid_relation_loss(
        activations, core_grade_indices, relation
    )
    direction_centroids = _centroid_relation_loss(
        data["hyperbolic_direction"], core_grade_indices, relation
    )
    prototype_vectors = model.prototype_layer.prototype_vectors.permute(0, 2, 1)
    prototype_vectors = prototype_vectors.mean(1)
    prototype_relation = _grade_balanced_relation_loss(
        prototype_vectors, prototype_grade_indices, relation
    )

    total = weight * (
        float(cfg.get("pathology_manifold_usage_weight", 1.0)) * usage_kl
        + float(cfg.get("pathology_manifold_occurrence_relation_weight", 0.25))
        * occurrence_relation
        + float(cfg.get("pathology_manifold_direction_relation_weight", 0.25))
        * direction_relation
        + float(cfg.get("pathology_manifold_centroid_weight", 0.5))
        * 0.5
        * (occurrence_centroids + direction_centroids)
        + float(cfg.get("pathology_manifold_prototype_relation_weight", 0.5))
        * prototype_relation
    )
    return total, {
        "pathology_manifold": total,
        "pathology_manifold_usage_kl": usage_kl,
        "pathology_manifold_usage_accuracy": usage_accuracy,
        "pathology_manifold_occurrence_relation": occurrence_relation,
        "pathology_manifold_direction_relation": direction_relation,
        "pathology_manifold_occurrence_centroids": occurrence_centroids,
        "pathology_manifold_direction_centroids": direction_centroids,
        "pathology_manifold_prototype_relation": prototype_relation,
    }
