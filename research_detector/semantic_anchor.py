"""Three-state pathology semantic-anchor contrastive supervision.

The anchors are fixed training-only pathology representations.  Ultrasound
whole-core embeddings and every localized prototype slot are projected into
the anchor space and optimized with a symmetric, class-aware InfoNCE loss.
The ordinary ProtoProstate classifier remains the inference head.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import torch
from omegaconf import DictConfig
from torch.nn import functional as F

from .prototype_utils import prototype_target_labels


_ANCHOR_CACHE: dict[tuple[str, str], tuple[torch.Tensor, torch.Tensor]] = {}


def load_semantic_anchor_bank(
    path: str, label_mode: str = "clinical_state"
) -> tuple[torch.Tensor, torch.Tensor]:
    resolved = str(Path(path).resolve())
    key = (resolved, str(label_mode))
    if key not in _ANCHOR_CACHE:
        archive = np.load(resolved, allow_pickle=False)
        anchors = F.normalize(
            torch.as_tensor(archive["anchors"], dtype=torch.float), dim=1
        )
        if label_mode == "grade_bands_01_23_45":
            if "grade_labels" not in archive:
                raise ValueError(
                    "Ordinal-band semantic anchors require grade_labels."
                )
            grades = torch.as_tensor(archive["grade_labels"], dtype=torch.long)
            states = (grades >= 2).long() + (grades >= 4).long()
        elif label_mode == "clinical_state":
            states = torch.as_tensor(archive["state_labels"], dtype=torch.long)
        else:
            raise ValueError(f"Unsupported semantic anchor label mode: {label_mode}")
        if anchors.ndim != 2 or len(anchors) != len(states):
            raise ValueError("Semantic anchor vectors and state labels do not match.")
        if set(states.tolist()) != {0, 1, 2}:
            raise ValueError("Semantic anchor bank must contain states 0, 1, and 2.")
        _ANCHOR_CACHE[key] = anchors, states
        logging.info(
            "Loaded semantic-anchor bank %s with label_mode=%s shape=%s "
            "and target_counts=%s.",
            resolved,
            label_mode,
            tuple(anchors.shape),
            torch.bincount(states, minlength=3).tolist(),
        )
    return _ANCHOR_CACHE[key]


def _state_logits(
    queries: torch.Tensor,
    candidates: torch.Tensor,
    candidate_states: torch.Tensor,
    temperature: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Aggregate cosine similarities into count-normalized three-state logits."""
    similarities = queries @ candidates.T / temperature
    logits: list[torch.Tensor] = []
    available: list[bool] = []
    for state in range(3):
        keep = candidate_states == state
        available.append(bool(keep.any()))
        if bool(keep.any()):
            logits.append(
                torch.logsumexp(similarities[:, keep], dim=1)
                - torch.log(keep.sum().to(similarities.dtype))
            )
        else:
            logits.append(
                torch.full(
                    (len(queries),),
                    -torch.inf,
                    dtype=similarities.dtype,
                    device=similarities.device,
                )
            )
    return torch.stack(logits, dim=1), torch.as_tensor(
        available, dtype=torch.bool, device=queries.device
    )


def _balanced_cross_entropy(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    counts = torch.bincount(labels, minlength=3).float()
    weights = labels.numel() / (3 * counts.clamp_min(1.0))
    weights[counts == 0] = 0
    return F.cross_entropy(logits, labels, weight=weights)


def _balanced_endpoint_cross_entropy(
    logits: torch.Tensor, labels: torch.Tensor
) -> torch.Tensor:
    counts = torch.bincount(labels, minlength=2).float()
    weights = labels.numel() / (2 * counts.clamp_min(1.0))
    weights[counts == 0] = 0
    return F.cross_entropy(logits, labels, weight=weights)


def _symmetric_anchor_loss(
    students: torch.Tensor,
    student_states: torch.Tensor,
    anchors: torch.Tensor,
    anchor_states: torch.Tensor,
    temperature: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    student_logits, _ = _state_logits(
        students, anchors, anchor_states, temperature
    )
    student_to_anchor = _balanced_cross_entropy(student_logits, student_states)
    present = torch.bincount(student_states, minlength=3) > 0
    reverse_keep = present[anchor_states]
    anchor_logits, _ = _state_logits(
        anchors[reverse_keep], students, student_states, temperature
    )
    anchor_to_student = _balanced_cross_entropy(
        anchor_logits, anchor_states[reverse_keep]
    )
    accuracy = (student_logits.argmax(1) == student_states).float().mean()
    return student_to_anchor, anchor_to_student, accuracy


def _balanced_transport_plan(
    logits: torch.Tensor, iterations: int
) -> torch.Tensor:
    """Return a row-normalized plan with balanced anchor-column usage."""
    rows, columns = logits.shape
    log_plan = logits.float()
    column_mass = log_plan.new_tensor(float(rows) / float(columns)).log()
    for _ in range(iterations):
        log_plan = log_plan - torch.logsumexp(log_plan, dim=1, keepdim=True)
        log_plan = (
            log_plan
            - torch.logsumexp(log_plan, dim=0, keepdim=True)
            + column_mass
        )
    log_plan = log_plan - torch.logsumexp(log_plan, dim=1, keepdim=True)
    return log_plan.exp()


def _margin_loss(
    values: torch.Tensor,
    states: torch.Tensor,
    anchors: torch.Tensor,
    anchor_states: torch.Tensor,
    margin: float,
) -> torch.Tensor:
    similarities = values @ anchors.T
    losses = []
    for state in range(3):
        keep = states == state
        if not bool(keep.any()):
            continue
        positive = similarities[keep][:, anchor_states == state].max(1).values
        negative = similarities[keep][:, anchor_states != state].max(1).values
        losses.append(F.relu(float(margin) - positive + negative).mean())
    return torch.stack(losses).mean() if losses else values.new_zeros(())


def _prototype_transport_terms(
    prototypes: torch.Tensor,
    prototype_states: torch.Tensor,
    anchors: torch.Tensor,
    anchor_states: torch.Tensor,
    temperature: torch.Tensor,
    iterations: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Balanced within-state mapping from ultrasound slots to real anchors.

    The old loss allowed every prototype to prefer one repeated pathology
    vector.  State-wise Sinkhorn gives every unique pathology exemplar equal
    soft mass, while the entropy and relation terms make the exported mapping
    specific and preserve within-state pathology geometry.
    """
    alignments = []
    relations = []
    entropies = []
    coverages = []
    for state in range(3):
        student = prototypes[prototype_states == state]
        teacher = anchors[anchor_states == state]
        similarity = student @ teacher.T
        plan = _balanced_transport_plan(
            similarity / temperature.float(), iterations
        )
        alignments.append(
            (plan * (1.0 - similarity)).sum() / float(len(student))
        )
        entropies.append(
            -(plan.clamp_min(1e-8) * plan.clamp_min(1e-8).log())
            .sum(1)
            .mean()
        )
        expected_column_mass = float(len(student)) / float(len(teacher))
        coverages.append(
            (plan.sum(0) - expected_column_mass).square().mean()
        )
        barycenters = F.normalize(plan @ teacher, dim=1)
        student_relation = student @ student.T
        teacher_relation = barycenters @ barycenters.T
        off_diagonal = ~torch.eye(
            len(student), dtype=torch.bool, device=student.device
        )
        relations.append(
            F.smooth_l1_loss(
                student_relation[off_diagonal],
                teacher_relation[off_diagonal],
            )
        )
    return (
        torch.stack(alignments).mean(),
        torch.stack(relations).mean(),
        torch.stack(entropies).mean(),
        torch.stack(coverages).mean(),
    )


def semantic_anchor_regularization(
    model,
    data: dict[str, Any],
    cfg: DictConfig,
    stage: str,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Bind ultrasound cores and prototype slots to fixed pathology anchors."""
    weight = float(cfg.get("semantic_anchor_loss_weight", 0.0))
    embedding = data.get("semantic_anchor_embedding")
    valid_stages = {"joint"}
    if bool(cfg.get("semantic_anchor_last_layer_alignment", False)):
        valid_stages.add("last_layer")
    if weight <= 0 or embedding is None or stage not in valid_stages:
        zero = data["prototype_activations"].new_zeros(())
        return zero, {}
    if getattr(model, "classification_target", None) != "three_way":
        raise ValueError("Semantic-anchor experiments require three-way classification.")
    label_mode = str(
        getattr(model, "semantic_anchor_label_mode", "clinical_state")
    )
    anchors, anchor_states = load_semantic_anchor_bank(
        str(cfg.semantic_anchor_bank_path), label_mode
    )
    anchors = anchors.to(embedding.device)
    anchor_states = anchor_states.to(embedding.device)
    if anchors.shape[1] != embedding.shape[1]:
        raise ValueError(
            f"Anchor dimension {anchors.shape[1]} does not match model "
            f"dimension {embedding.shape[1]}."
        )
    core_embeddings = F.normalize(embedding.float(), dim=1)
    clinical_states = prototype_target_labels(model, data).to(embedding.device)
    core_states = (
        model.hyperbolic_needle_encoder.target_indices_from_grades(
            data["grade_group"].long().to(embedding.device)
        )
        if label_mode == "grade_bands_01_23_45"
        else clinical_states
    )
    prototype_embeddings, prototype_states = (
        model.semantic_anchor_prototype_embeddings()
    )
    prototype_states = prototype_states.to(embedding.device)
    temperature = model.semantic_anchor_temperature()

    core_forward, core_reverse, core_accuracy = _symmetric_anchor_loss(
        core_embeddings, core_states, anchors, anchor_states, temperature
    )
    prototype_forward, prototype_reverse, prototype_accuracy = (
        _symmetric_anchor_loss(
            prototype_embeddings,
            prototype_states,
            anchors,
            anchor_states,
            temperature,
        )
    )
    bridges: list[torch.Tensor] = []
    for state in range(3):
        core_keep = core_states == state
        prototype_keep = prototype_states == state
        if bool(core_keep.any()) and bool(prototype_keep.any()):
            core_center = F.normalize(
                core_embeddings[core_keep].mean(0, keepdim=True), dim=1
            )
            prototype_center = F.normalize(
                prototype_embeddings[prototype_keep].mean(0, keepdim=True), dim=1
            )
            bridges.append(1.0 - (core_center * prototype_center).sum())
    bridge = (
        torch.stack(bridges).mean()
        if bridges
        else core_embeddings.new_zeros(())
    )
    reverse_weight = float(cfg.get("semantic_anchor_reverse_weight", 0.5))
    core_weight = float(cfg.get("semantic_anchor_core_weight", 1.0))
    prototype_weight = float(cfg.get("semantic_anchor_prototype_weight", 1.0))
    bridge_weight = float(cfg.get("semantic_anchor_bridge_weight", 0.02))
    endpoint_loss = core_embeddings.new_zeros(())
    consistency = core_embeddings.new_zeros(())
    direct_logits = data.get("semantic_anchor_class_logits")
    if direct_logits is not None:
        pca_logits = torch.stack(
            (direct_logits[:, 0], torch.logsumexp(direct_logits[:, 1:], dim=1)),
            dim=1,
        )
        cspca_logits = torch.stack(
            (torch.logsumexp(direct_logits[:, :2], dim=1), direct_logits[:, 2]),
            dim=1,
        )
        endpoint_loss = 0.5 * (
            _balanced_endpoint_cross_entropy(
                pca_logits, (clinical_states > 0).long()
            )
            + _balanced_endpoint_cross_entropy(
                cspca_logits, (clinical_states == 2).long()
            )
        )
        base_logits = data.get("prototype_class_logits_pre_semantic_anchor")
        if base_logits is not None:
            anchor_log_probs = F.log_softmax(direct_logits, dim=1)
            base_log_probs = F.log_softmax(base_logits, dim=1)
            consistency = 0.5 * (
                F.kl_div(
                    anchor_log_probs,
                    base_log_probs.exp(),
                    reduction="batchmean",
                )
                + F.kl_div(
                    base_log_probs,
                    anchor_log_probs.exp(),
                    reduction="batchmean",
                )
            )
    endpoint_weight = float(cfg.get("semantic_anchor_endpoint_weight", 0.5))
    consistency_weight = float(
        cfg.get("semantic_anchor_consistency_weight", 0.1)
    )
    transport_alignment = core_embeddings.new_zeros(())
    transport_relation = core_embeddings.new_zeros(())
    transport_entropy = core_embeddings.new_zeros(())
    transport_coverage = core_embeddings.new_zeros(())
    transport_weight = float(cfg.get("semantic_anchor_transport_weight", 0.0))
    if transport_weight > 0:
        (
            transport_alignment,
            transport_relation,
            transport_entropy,
            transport_coverage,
        ) = _prototype_transport_terms(
            prototype_embeddings,
            prototype_states,
            anchors,
            anchor_states,
            temperature,
            int(cfg.get("semantic_anchor_transport_iterations", 8)),
        )
    margin = float(cfg.get("semantic_anchor_margin", 0.0))
    core_margin = _margin_loss(
        core_embeddings,
        core_states,
        anchors,
        anchor_states,
        margin,
    )
    prototype_margin = _margin_loss(
        prototype_embeddings,
        prototype_states,
        anchors,
        anchor_states,
        margin,
    )
    margin_weight = float(cfg.get("semantic_anchor_margin_weight", 0.0))
    total = weight * (
        core_weight * (core_forward + reverse_weight * core_reverse)
        + prototype_weight
        * (prototype_forward + reverse_weight * prototype_reverse)
        + bridge_weight * bridge
        + endpoint_weight * endpoint_loss
        + consistency_weight * consistency
        + transport_weight
        * (
            transport_alignment
            + float(cfg.get("semantic_anchor_transport_relation_weight", 0.1))
            * transport_relation
            + float(cfg.get("semantic_anchor_transport_entropy_weight", 0.01))
            * transport_entropy
        )
        + margin_weight * (core_margin + prototype_margin)
    )
    return total, {
        "semantic_anchor": total,
        "semantic_anchor_core_forward": core_forward,
        "semantic_anchor_core_reverse": core_reverse,
        "semantic_anchor_prototype_forward": prototype_forward,
        "semantic_anchor_prototype_reverse": prototype_reverse,
        "semantic_anchor_bridge": bridge,
        "semantic_anchor_endpoint": endpoint_loss,
        "semantic_anchor_consistency": consistency,
        "semantic_anchor_transport_alignment": transport_alignment,
        "semantic_anchor_transport_relation": transport_relation,
        "semantic_anchor_transport_entropy": transport_entropy,
        "semantic_anchor_transport_coverage": transport_coverage,
        "semantic_anchor_core_margin": core_margin,
        "semantic_anchor_prototype_margin": prototype_margin,
        "semantic_anchor_core_accuracy": core_accuracy,
        "semantic_anchor_prototype_accuracy": prototype_accuracy,
        "semantic_anchor_temperature": temperature,
    }
