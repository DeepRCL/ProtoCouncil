"""Training, projection, and evaluation helpers for ProtoProstate."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment
from torch.nn import functional as F
from tqdm import tqdm

from .needle_axis import pool_logits_along_needle_axis


def valid_region_sample_mask(batch: dict[str, Any]) -> torch.Tensor:
    """Identify cores with at least one supervised needle/prostate pixel."""
    valid = (batch["prostate_mask"] > 0.5) & (batch["needle_mask"] > 0.5)
    return valid.flatten(1).any(dim=1)


def subset_batch(batch: dict[str, Any], keep: torch.Tensor) -> dict[str, Any]:
    """Select batch items while preserving the heterogeneous loader fields."""
    keep_cpu = keep.detach().cpu().bool()
    indices = keep_cpu.nonzero(as_tuple=False).flatten().tolist()
    batch_size = int(keep_cpu.numel())
    selected: dict[str, Any] = {}
    for key, value in batch.items():
        if torch.is_tensor(value) and value.ndim > 0 and len(value) == batch_size:
            selected[key] = value[keep_cpu]
        elif isinstance(value, np.ndarray) and value.ndim > 0 and len(value) == batch_size:
            selected[key] = value[keep_cpu.numpy()]
        elif isinstance(value, list) and len(value) == batch_size:
            selected[key] = [value[index] for index in indices]
        elif isinstance(value, tuple) and len(value) == batch_size:
            selected[key] = tuple(value[index] for index in indices)
        else:
            selected[key] = value
    return selected


def filter_invalid_region_samples(
    batch: dict[str, Any],
) -> tuple[dict[str, Any] | None, int]:
    """Strictly remove cores for which the weak label has no valid support."""
    keep = valid_region_sample_mask(batch)
    skipped = int((~keep).sum())
    if not bool(keep.any()):
        return None, skipped
    return subset_batch(batch, keep), skipped


def _dataset_core_index(loader) -> dict[str, int]:
    core_ids = getattr(loader.dataset, "cores", None)
    if core_ids is None:
        core_ids = getattr(loader.dataset, "core_ids", None)
    if core_ids is None:
        return {}
    return {str(core_id): index for index, core_id in enumerate(core_ids)}


def _assign_unique_hyperbolic_projection_records(
    candidates: list[list[dict[str, Any]]],
    targets: int,
    prototypes_per_target: int,
    candidate_limit: int,
) -> list[dict[str, Any] | None]:
    """Jointly project anchors so one training core cannot explain two anchors.

    V9 projected every anchor independently, which allowed the low/high
    severity anchors of a grade to collapse onto the exact same core.  This
    solves a small linear assignment problem per grade over the best real-core
    candidates.  The assignment remains class/grade compatible and retains a
    fallback only when the noisy q tolerance leaves too few feasible cores.
    """
    selected: list[dict[str, Any] | None] = [None] * len(candidates)
    invalid_cost = 1.0e6
    for target_index in range(targets):
        first = target_index * prototypes_per_target
        anchor_indices = list(range(first, first + prototypes_per_target))
        anchor_candidates: list[list[dict[str, Any]]] = []
        for anchor_index in anchor_indices:
            # Within-tolerance candidates always precede fallback candidates;
            # distance and q agreement then break ties.
            ranked = sorted(
                candidates[anchor_index],
                key=lambda record: (
                    bool(record["used_q_fallback"]),
                    float(record["projection_cost"]),
                ),
            )[:candidate_limit]
            anchor_candidates.append(ranked)
        core_ids = sorted(
            {
                str(record["core_id"])
                for records in anchor_candidates
                for record in records
            }
        )
        if not core_ids:
            continue
        core_column = {core_id: index for index, core_id in enumerate(core_ids)}
        cost = np.full(
            (prototypes_per_target, len(core_ids)), invalid_cost, dtype=np.float64
        )
        record_lookup: dict[tuple[int, int], dict[str, Any]] = {}
        for row, records in enumerate(anchor_candidates):
            for record in records:
                column = core_column[str(record["core_id"])]
                value = float(record["projection_cost"])
                if value < cost[row, column]:
                    cost[row, column] = value
                    record_lookup[(row, column)] = record
        rows, columns = linear_sum_assignment(cost)
        for row, column in zip(rows.tolist(), columns.tolist()):
            if cost[row, column] >= invalid_cost:
                continue
            record = dict(record_lookup[(row, column)])
            record["unique_projection"] = True
            selected[anchor_indices[row]] = record
    return selected


def _assign_unique_occurrence_projection_records(
    model, candidates: list[list[dict[str, Any]]]
) -> list[dict[str, Any] | None]:
    """Choose a different projected training core for every prototype in a bank.

    Projection used to maximize each prototype independently, so several
    prototypes of one class could become exact copies of evidence from the
    same core.  This class-compatible Hungarian assignment maximizes total
    activation while making the projected exemplar core unique within each
    prototype class (and grade target, when one is explicitly configured).
    """
    selected: list[dict[str, Any] | None] = [None] * len(candidates)
    groups: dict[tuple[int, int], list[int]] = {}
    layer = model.prototype_layer
    for prototype_index in range(layer.num_prototypes):
        key = (
            int(layer.prototype_class_identity[prototype_index].argmax()),
            int(layer.prototype_grade_targets[prototype_index]),
        )
        groups.setdefault(key, []).append(prototype_index)

    invalid_cost = 1.0e6
    limit = int(model.prototype_projection_candidate_limit)
    for prototype_indices in groups.values():
        ranked_candidates = [
            sorted(
                candidates[index],
                key=lambda record: float(record["activation"]),
                reverse=True,
            )[:limit]
            for index in prototype_indices
        ]
        core_ids = sorted(
            {
                str(record["core_id"])
                for records in ranked_candidates
                for record in records
            }
        )
        if not core_ids:
            continue
        core_column = {core_id: index for index, core_id in enumerate(core_ids)}
        cost = np.full(
            (len(prototype_indices), len(core_ids)), invalid_cost, dtype=np.float64
        )
        lookup: dict[tuple[int, int], dict[str, Any]] = {}
        for row, records in enumerate(ranked_candidates):
            for record in records:
                column = core_column[str(record["core_id"])]
                value = -float(record["activation"])
                if value < cost[row, column]:
                    cost[row, column] = value
                    lookup[(row, column)] = record
        rows, columns = linear_sum_assignment(cost)
        for row, column in zip(rows.tolist(), columns.tolist()):
            if cost[row, column] >= invalid_cost:
                continue
            record = dict(lookup[(row, column)])
            record["unique_projection"] = True
            selected[prototype_indices[row]] = record
    return selected


def _assign_center_balanced_needle_projection_records(
    candidates: list[list[dict[str, Any]]],
    num_classes: int,
    prototypes_per_class: int,
    candidate_limit: int,
) -> list[dict[str, Any] | None]:
    """Ground each class bank in unique real cores spread across source centers."""
    selected: list[dict[str, Any] | None] = [None] * len(candidates)
    for class_index in range(num_classes):
        first = class_index * prototypes_per_class
        prototype_indices = list(range(first, first + prototypes_per_class))
        centers = sorted(
            {
                str(record["center"])
                for index in prototype_indices
                for record in candidates[index]
            }
        )
        if not centers:
            continue
        center_usage = {center: 0 for center in centers}
        used_cores: set[str] = set()
        for local_index, prototype_index in enumerate(prototype_indices):
            ranked = sorted(
                candidates[prototype_index],
                key=lambda record: float(record["activation"]),
                reverse=True,
            )[:candidate_limit]
            available = [
                record
                for record in ranked
                if str(record["core_id"]) not in used_cores
            ]
            if not available:
                continue
            desired = centers[local_index % len(centers)]
            least_used = min(center_usage.values())
            preferred_centers = [
                desired,
                *[
                    center
                    for center in centers
                    if center != desired and center_usage[center] == least_used
                ],
            ]
            chosen = None
            for center in preferred_centers:
                matches = [
                    record for record in available
                    if str(record["center"]) == center
                ]
                if matches:
                    chosen = matches[0]
                    break
            if chosen is None:
                chosen = available[0]
            record = dict(chosen)
            record["unique_projection"] = True
            record["center_balanced_projection"] = True
            selected[prototype_index] = record
            used_cores.add(str(record["core_id"]))
            center_usage[str(record["center"])] += 1
    return selected


def clinical_prompt_tensor(model, batch: dict[str, Any], device: torch.device):
    """Stack normalized Prost-RL clinical fields in the model-declared order."""
    names = tuple(getattr(model, "clinical_prompt_names", ()))
    if not names:
        return None
    missing = [name for name in names if name not in batch]
    if missing:
        raise KeyError(f"Clinical prompts missing from batch: {missing}")
    batch_size = len(batch["bmode"])
    values = [
        torch.as_tensor(batch[name], device=device).float().reshape(batch_size, -1)[:, 0]
        for name in names
    ]
    return torch.stack(values, dim=1)


def prototype_target_labels(model, batch: dict[str, Any]) -> torch.Tensor:
    """Return the class represented by the model's prototype banks.

    NCT2013/Prost-RL define the csPCa endpoint as stored Grade Group > 2.
    Keeping this in one helper prevents projection, geometry losses, and the
    classification layer from silently using different targets.
    """
    target = getattr(model, "classification_target", "pca")
    if target == "isup":
        return batch["grade_group"].long()
    if target == "three_way":
        cancer = batch["label"].long()
        cspca = (batch["grade_group"] > 2).long()
        # 0=benign, 1=non-significant cancer, 2=csPCa.
        return torch.where(cspca.bool(), torch.full_like(cancer, 2), cancer)
    if target == "cspca":
        return (batch["grade_group"] > 2).long()
    return batch["label"].long()


def forward_batch(
    model, batch: dict[str, Any], device: torch.device
) -> dict[str, Any] | None:
    """Run ProtoProstate and attach the fields expected by Prost-RL losses/evaluator."""
    batch, skipped = filter_invalid_region_samples(batch)
    if batch is None:
        return None
    image = batch["bmode"].to(device)
    prostate_mask = batch["prostate_mask"].to(device)
    needle_mask = batch["needle_mask"].to(device)
    outputs = model(
        image,
        prostate_mask,
        needle_mask,
        clinical_prompt_tensor(model, batch, device),
    )

    data = dict(batch)
    cancer_logits = outputs["mask_logits"]
    data["cancer_logits"] = cancer_logits
    data["image_level_classification_outputs"] = outputs["cls_outputs"]
    if "cspca_cls_outputs" in outputs:
        data["image_level_cspca_outputs"] = outputs["cspca_cls_outputs"]
    data["prototype_class_logits"] = outputs["prototype_class_logits"]
    data["prototype_class_logits_image"] = outputs["prototype_class_logits_image"]
    for key in (
        "clinical_logit",
        "clinical_contributions",
        "occurrence_maps",
        "prototype_class_logits_conditioned",
        "prompt_class_logits",
        "prompt_spatial_gate",
        "prompt_film_gamma",
        "prompt_film_beta",
        "pca_class_logits",
        "cspca_class_logits",
        "prototype_class_logits_pre_hyperbolic",
        "occurrence_conditioned_logits_pre_hyperbolic",
        "occurrence_image_logits_pre_hyperbolic",
        "hyperbolic_direction",
        "hyperbolic_radius",
        "hyperbolic_predicted_involvement",
        "hyperbolic_prototype_directions",
        "hyperbolic_prototype_involvement",
        "hyperbolic_prototype_distances",
        "hyperbolic_grade_logits",
        "hyperbolic_three_way_logits",
        "hyperbolic_three_way_logits_image",
        "hyperbolic_grade_logits_image",
        "hyperbolic_fusion_scale",
        "hyperbolic_base_logit_scale",
        "branch_fusion_global_weight",
        "needle_axis_occupied_bins",
        "needle_selection_logits",
        "needle_selection_probabilities",
        "needle_prototype_logits",
        "needle_prototype_activations",
        "needle_prototype_slot_similarities",
        "needle_prototype_slot_attention",
        "needle_prototype_slot_features",
        "needle_prototype_slot_indices",
        "needle_prototype_class_weights",
        "needle_prototype_augmented_logits",
        "needle_prototype_augmented_activations",
        "needle_prototype_augmented_slot_attention",
        "needle_axis_local_sequence",
        "needle_axis_context_sequence",
        "pathology_transport_grade_logits",
        "pathology_transport_three_way_logits",
        "pathology_transport_three_way_logits_image",
        "pathology_transport_anchor_activations",
        "pathology_transport_anchor_attention",
        "pathology_transport_mapping",
        "pathology_transport_prototype_similarity",
        "pathology_transport_prototype_embeddings",
        "pathology_transport_barycenters",
        "pathology_transport_core_embedding",
        "pathology_transport_predicted_severity",
        "pathology_transport_lorentz_point",
        "pathology_transport_alignment",
        "pathology_transport_coverage",
        "pathology_transport_relation",
        "pathology_transport_diversity",
        "pathology_transport_assignment_entropy",
        "pathology_transport_fusion_scale",
        "prototype_class_logits_pre_pathology_transport",
        "prototype_class_logits_pre_dense_endpoint",
        "dense_endpoint_raw_logits",
        "dense_endpoint_three_way_logits",
        "dense_endpoint_final_logits",
        "dense_endpoint_three_way_logits_image",
        "dense_endpoint_final_logits_image",
        "dense_endpoint_residual_scale",
        "prototype_class_logits_pre_prost_rl_endpoint",
        "prost_rl_endpoint_pca_logits",
        "prost_rl_endpoint_conditional_cspca_logits",
        "prost_rl_endpoint_three_way_logits",
        "prost_rl_endpoint_final_logits",
        "prost_rl_endpoint_three_way_logits_image",
        "prost_rl_endpoint_final_logits_image",
        "prost_rl_endpoint_residual_scale",
        "pathology_geometry_embedding",
        "semantic_anchor_embedding",
        "semantic_anchor_target_logits",
        "semantic_anchor_class_logits",
        "semantic_anchor_fusion_scale",
        "prototype_class_logits_pre_semantic_anchor",
        "encoder_features",
    ):
        if key in outputs:
            data[key] = outputs[key]
    if "grade_group_logits" in outputs:
        data["grade_group_logits"] = outputs["grade_group_logits"]
        data["grade_group_probabilities"] = outputs["grade_group_probabilities"]
    if "uncertainty_logit" in outputs:
        data["uncertainty_logit"] = outputs["uncertainty_logit"]
        data["uncertainty_activations"] = outputs["uncertainty_activations"]
    data["prototype_activations"] = outputs["prototype_activations"]
    data["matched_values"] = outputs["matched_values"]
    data["matched_indices"] = outputs["matched_indices"]
    data["invalid_region_cores_filtered"] = skipped

    valid = (prostate_mask > 0.5) & (needle_mask > 0.5)
    prostate = prostate_mask > 0.5
    probabilities = cancer_logits.sigmoid()
    needle_scores = []
    prostate_scores = []
    thresholded = []
    for sample_idx in range(len(image)):
        valid_probs = probabilities[sample_idx][valid[sample_idx]]
        prostate_probs = probabilities[sample_idx][prostate[sample_idx]]
        if valid_probs.numel() == 0:
            valid_probs = probabilities[sample_idx].flatten()
        if prostate_probs.numel() == 0:
            prostate_probs = probabilities[sample_idx].flatten()
        axis_bins = int(getattr(model, "needle_axis_bins", 0))
        if axis_bins > 0:
            axis_logits = pool_logits_along_needle_axis(
                cancer_logits[sample_idx], valid[sample_idx], axis_bins
            )
            axis_probs = axis_logits.sigmoid()
            needle_scores.append(axis_probs.mean())
            thresholded.append((axis_probs > 0.5).float().mean())
        else:
            needle_scores.append(valid_probs.mean())
            thresholded.append((valid_probs > 0.5).float().mean())
        prostate_scores.append(prostate_probs.mean())
    data["average_needle_heatmap_value"] = torch.stack(needle_scores)
    data["average_prostate_heatmap_value"] = torch.stack(prostate_scores)
    data["thresholded_needle_involvement"] = torch.stack(thresholded)
    return data


def needle_axis_prototype_regularization(
    model,
    data: dict[str, Any],
    cfg,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Regularize the self-explaining ordered-needle prototype classifier."""
    encoder = getattr(model, "hyperbolic_needle_encoder", None)
    head = getattr(encoder, "needle_prototype_head", None)
    reference = data["matched_values"]
    zero = reference.new_zeros(())
    if head is None:
        return zero, {}

    logits = data["needle_prototype_logits"].float()
    activations = data["needle_prototype_activations"].float()
    attention = data["needle_prototype_slot_attention"].float()
    labels = prototype_target_labels(model, data).to(logits.device)
    classes = activations.shape[1]
    batch_indices = torch.arange(len(labels), device=labels.device)
    own = activations[batch_indices, labels]
    own_max = own.amax(dim=1)
    wrong_mask = F.one_hot(labels, num_classes=classes).bool()[:, :, None]
    wrong_max = activations.masked_fill(wrong_mask, -1.0).flatten(1).amax(dim=1)
    cluster = (1.0 - own_max).mean()
    margin = float(cfg.get("needle_prototype_separation_margin", 0.10))
    separation = F.relu(wrong_max - own_max + margin).mean()

    vectors = F.normalize(head.prototype_vectors.float(), dim=-1)
    flattened = vectors.flatten(1, 2)
    gram = torch.einsum("cmd,cnd->cmn", flattened, flattened)
    identity = torch.eye(gram.shape[-1], device=gram.device, dtype=gram.dtype)
    diversity = ((gram * (1.0 - identity[None])).square()).sum() / (
        classes * gram.shape[-1] * max(gram.shape[-1] - 1, 1)
    )

    normalized_attention = F.normalize(attention, p=2, dim=-1)
    overlap_matrix = torch.einsum(
        "bcksl,bcktl->bckst", normalized_attention, normalized_attention
    )
    slots = attention.shape[-2]
    slot_identity = torch.eye(
        slots, device=attention.device, dtype=attention.dtype
    )
    slot_overlap = (
        overlap_matrix * (1.0 - slot_identity[None, None, None])
    ).sum() / (
        len(attention)
        * attention.shape[1]
        * attention.shape[2]
        * slots
        * max(slots - 1, 1)
    )
    attention_entropy = -(
        attention.clamp_min(1e-8) * attention.clamp_min(1e-8).log()
    ).sum(-1).mean()
    classifier_l1 = head.class_weights.mean() + head.class_bias.abs().mean()

    counts = torch.bincount(labels, minlength=classes).float()
    class_weights = len(labels) / (classes * counts.clamp_min(1.0))
    main_ce = F.cross_entropy(logits, labels, weight=class_weights)
    augmented_ce = zero
    prediction_consistency = zero
    rationale_consistency = zero
    attention_consistency = zero
    augmented_logits = data.get("needle_prototype_augmented_logits")
    if augmented_logits is not None:
        augmented_logits = augmented_logits.float()
        augmented_ce = F.cross_entropy(
            augmented_logits, labels, weight=class_weights
        )
        prediction_consistency = F.kl_div(
            F.log_softmax(augmented_logits, dim=1),
            F.softmax(logits.detach(), dim=1),
            reduction="batchmean",
        )
        rationale_consistency = F.smooth_l1_loss(
            data["needle_prototype_augmented_activations"].float(),
            activations.detach(),
        )
        attention_consistency = F.smooth_l1_loss(
            data["needle_prototype_augmented_slot_attention"].float(),
            attention.detach(),
        )

    components = {
        "needle_prototype_main_ce": main_ce,
        "needle_prototype_cluster": cluster,
        "needle_prototype_separation": separation,
        "needle_prototype_diversity": diversity,
        "needle_prototype_slot_overlap": slot_overlap,
        "needle_prototype_attention_entropy": attention_entropy,
        "needle_prototype_classifier_l1": classifier_l1,
        "needle_prototype_augmented_ce": augmented_ce,
        "needle_prototype_prediction_consistency": prediction_consistency,
        "needle_prototype_rationale_consistency": rationale_consistency,
        "needle_prototype_attention_consistency": attention_consistency,
    }
    weights = cfg.get("needle_prototype_loss_weights", {})
    total = sum(
        float(weights.get(name.removeprefix("needle_prototype_"), 0.0)) * value
        for name, value in components.items()
    )
    components["needle_prototype_regularization"] = total
    return total, components


def pathology_prototype_transport_regularization(
    model,
    data: dict[str, Any],
    cfg,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Train the shared pathology component without treating cores as paired WSIs.

    State supervision is exact at core level. Grade supervision is deliberately
    soft within the compatible state because a PANDA slide-level prototype label
    is not an exact local histologic annotation for an ultrasound needle bin.
    """
    encoder = getattr(model, "hyperbolic_needle_encoder", None)
    head = getattr(encoder, "pathology_transport_head", None)
    reference = data["matched_values"]
    zero = reference.new_zeros(())
    if head is None:
        return zero, {}

    state_logits = data["pathology_transport_three_way_logits"].float()
    grade_logits = data["pathology_transport_grade_logits"].float()
    states = prototype_target_labels(model, data).long().to(state_logits.device)
    grades = data["grade_group"].long().to(grade_logits.device).clamp(0, 5)
    counts = torch.bincount(states, minlength=3).float()
    class_weights = len(states) / (3 * counts.clamp_min(1.0))
    state_ce = F.cross_entropy(state_logits, states, weight=class_weights)

    grade_values = torch.arange(6, device=grades.device, dtype=torch.float32)
    distance = (grade_values[None] - grades.float()[:, None]).abs()
    sigma = float(cfg.get("pathology_transport_grade_smoothing", 0.75))
    soft_targets = torch.exp(-distance / max(sigma, 1e-3))
    compatible = torch.where(
        states[:, None] == 0,
        grade_values[None] == 0,
        torch.where(
            states[:, None] == 1,
            (grade_values[None] >= 1) & (grade_values[None] <= 2),
            grade_values[None] >= 3,
        ),
    )
    soft_targets = soft_targets * compatible.float()
    soft_targets = soft_targets / soft_targets.sum(1, keepdim=True).clamp_min(1e-8)
    grade_log_probabilities = F.log_softmax(grade_logits, dim=1)
    grade_probabilities = grade_log_probabilities.exp()
    grade_ce = -(soft_targets * grade_log_probabilities).sum(1).mean()
    grade_emd = (
        grade_probabilities.cumsum(1) - soft_targets.cumsum(1)
    ).square().mean()

    alignment = data["pathology_transport_alignment"].float()
    coverage = data["pathology_transport_coverage"].float()
    relation = data["pathology_transport_relation"].float()
    diversity = data["pathology_transport_diversity"].float()
    entropy = data["pathology_transport_assignment_entropy"].float()
    minimum_entropy = state_logits.new_tensor(
        float(np.log(float(cfg.get("pathology_transport_min_effective_anchors", 2.0))))
    )
    entropy_floor = F.relu(minimum_entropy - entropy).square()

    components = {
        "pathology_transport_state_ce": state_ce,
        "pathology_transport_grade_ce": grade_ce,
        "pathology_transport_grade_emd": grade_emd,
        "pathology_transport_alignment_loss": alignment,
        "pathology_transport_coverage_loss": coverage,
        "pathology_transport_relation_loss": relation,
        "pathology_transport_diversity_loss": diversity,
        "pathology_transport_entropy_floor": entropy_floor,
        "pathology_transport_assignment_entropy": entropy,
        "pathology_transport_fusion_scale_value": data[
            "pathology_transport_fusion_scale"
        ].float(),
    }
    weights = cfg.get("pathology_transport_loss_weights", {})
    objective_names = {
        "state_ce": state_ce,
        "grade_ce": grade_ce,
        "grade_emd": grade_emd,
        "alignment": alignment,
        "coverage": coverage,
        "relation": relation,
        "diversity": diversity,
        "entropy_floor": entropy_floor,
    }
    total = sum(
        float(weights.get(name, 0.0)) * value
        for name, value in objective_names.items()
    )
    components["pathology_transport_regularization"] = total
    return total, components


def hyperbolic_needle_regularization(
    model,
    data: dict[str, Any],
    cfg,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Grade-direction and noisy-involvement-radius objectives.

    The five-way geodesic CE supplies a contrast against benign and every
    other observed grade. A second angular CE prevents involvement radius from
    doing all of the work. Pairwise ranking ignores cancer pairs whose q values
    differ by less than the configured noise tolerance.
    """
    if getattr(model, "hyperbolic_needle_encoder", None) is None:
        zero = data["prototype_activations"].new_zeros(())
        return zero, {}
    if str(cfg.get("hyperbolic_objective", "full")) == "simplified":
        return _simplified_hyperbolic_regularization(model, data, cfg)
    grades = data["grade_group"].long().to(model.device)
    involvement = data["involvement"].float().to(model.device).clamp(0, 1)
    involvement_valid = data.get("involvement_valid")
    involvement_valid = (
        involvement_valid.bool().to(model.device)
        if involvement_valid is not None
        else torch.ones_like(involvement, dtype=torch.bool)
    )
    encoder = model.hyperbolic_needle_encoder
    target_values = tuple(encoder.target_values)
    grade_values = torch.as_tensor(target_values, device=model.device)
    if encoder.target_mode == "three_way":
        targets = prototype_target_labels(model, data).long().to(model.device)
    elif encoder.target_mode == "grade_bands_01_23_45":
        targets = encoder.target_indices_from_grades(grades)
    else:
        targets = encoder.target_indices_from_grades(grades)
    counts = torch.bincount(targets, minlength=len(grade_values)).float()
    class_weights = targets.numel() / (len(grade_values) * counts.clamp_min(1))
    class_weights[counts == 0] = 0

    grade_ce = F.cross_entropy(
        data["hyperbolic_grade_logits"], targets, weight=class_weights
    )
    clinical_targets = prototype_target_labels(model, data).long().to(model.device)
    clinical_counts = torch.bincount(clinical_targets, minlength=3).float()
    clinical_weights = clinical_targets.numel() / (
        3 * clinical_counts.clamp_min(1)
    )
    clinical_weights[clinical_counts == 0] = 0
    endpoint_ce = F.cross_entropy(
        data["hyperbolic_three_way_logits"].float(),
        clinical_targets,
        weight=clinical_weights,
    )
    sample_directions = F.normalize(data["hyperbolic_direction"].float(), dim=-1)
    prototype_directions = F.normalize(
        data["hyperbolic_prototype_directions"].float(), dim=-1
    )
    angular_logits = torch.einsum(
        "bd,gpd->bgp", sample_directions, prototype_directions
    ) / float(cfg.get("hyperbolic_angular_temperature", 0.1))
    angular_grade_logits = torch.logsumexp(angular_logits, dim=2)
    cancer = grades > 0
    angular_margin = float(cfg.get("hyperbolic_angular_margin", 0.0))
    if angular_margin > 0 and bool(cancer.any()):
        # CosFace-style grade margin.  Applying the margin after aggregating a
        # grade's anchors makes every correct cancer direction move farther
        # inside its own cone without imposing an arbitrary ordering of cone
        # angles.
        angular_grade_logits = angular_grade_logits.clone()
        cancer_rows = torch.nonzero(cancer).flatten()
        angular_grade_logits[cancer_rows, targets[cancer]] -= (
            angular_margin
            / float(cfg.get("hyperbolic_angular_temperature", 0.1))
        )
    angular_ce = (
        F.cross_entropy(
            angular_grade_logits[cancer],
            targets[cancer],
            weight=class_weights,
        )
        if bool(cancer.any())
        else grade_ce.new_zeros(())
    )

    # Multiple anchors within a Grade Group should specialize rather than all
    # receiving the same cores.  Per-sample assignments are encouraged to be
    # sharp, while their aggregate usage remains balanced across anchors.
    usage_terms = []
    usage_sharpness = float(
        cfg.get("hyperbolic_anchor_usage_sharpness", 0.1)
    )
    num_anchors = prototype_directions.shape[1]
    if num_anchors > 1:
        log_num_anchors = torch.log(
            angular_logits.new_tensor(float(num_anchors))
        )
        for grade_index in range(len(grade_values)):
            keep = targets == grade_index
            if not bool(keep.any()):
                continue
            assignment = torch.softmax(
                angular_logits[keep, grade_index], dim=1
            ).clamp_min(1e-8)
            mean_usage = assignment.mean(0)
            balance = (
                mean_usage * (mean_usage.log() + log_num_anchors)
            ).sum()
            specialization = -(
                assignment * assignment.log()
            ).sum(1).mean() / log_num_anchors
            usage_terms.append(balance + usage_sharpness * specialization)
    anchor_usage = (
        torch.stack(usage_terms).mean()
        if usage_terms
        else grade_ce.new_zeros(())
    )

    predicted = data["hyperbolic_predicted_involvement"].float()
    tolerance = float(cfg.get("hyperbolic_involvement_tolerance", 0.05))
    residual = F.relu((predicted - involvement).abs() - tolerance)
    radial = (
        F.smooth_l1_loss(
            residual[involvement_valid],
            torch.zeros_like(residual[involvement_valid]),
            beta=0.1,
        )
        if bool(involvement_valid.any())
        else residual.new_zeros(())
    )
    benign_root = (
        predicted[~cancer].square().mean()
        if bool((~cancer).any())
        else radial.new_zeros(())
    )

    gap = involvement[:, None] - involvement[None, :]
    predicted_gap = predicted[:, None] - predicted[None, :]
    # Rank both benign--cancer and cancer--cancer pairs. Benign--benign pairs
    # carry no ordinal information because both targets are at the cone root.
    pair_mask = (
        (cancer[:, None] | cancer[None, :])
        & involvement_valid[:, None]
        & involvement_valid[None, :]
        & (gap.abs() >= float(cfg.get("hyperbolic_rank_min_gap", 0.1)))
    )
    pair_mask = pair_mask & torch.triu(
        torch.ones_like(pair_mask, dtype=torch.bool), diagonal=1
    )
    rank_margin = float(cfg.get("hyperbolic_rank_margin", 0.05))
    ordinal_rank = (
        F.relu(
            rank_margin
            - gap[pair_mask].sign() * predicted_gap[pair_mask]
        ).mean()
        if bool(pair_mask.any())
        else radial.new_zeros(())
    )

    # HyperPAS: prototypes of the same grade share a cone direction, while
    # different-grade prototype directions repel one another.
    flat_directions = prototype_directions.flatten(0, 1)
    proto_grades = torch.arange(
        len(grade_values), device=model.device
    ).repeat_interleave(prototype_directions.shape[1])
    cosine = flat_directions @ flat_directions.T
    eye = torch.eye(len(flat_directions), dtype=torch.bool, device=model.device)
    same = (proto_grades[:, None] == proto_grades[None]) & ~eye
    logits = cosine / float(cfg.get("hyperpas_temperature", 0.1))
    logits = logits.masked_fill(eye, -torch.inf)
    hyperpas = -(
        torch.logsumexp(logits.masked_fill(~same, -torch.inf), dim=1)
        - torch.logsumexp(logits, dim=1)
    ).mean()

    prototype_q = data["hyperbolic_prototype_involvement"].float()
    minimum_q_gap = float(cfg.get("hyperbolic_prototype_min_q_gap", 0.05))
    ordered_q = (
        prototype_q
        if encoder.target_mode == "grade_bands_01_23_45"
        else prototype_q[1:]
    )
    prototype_order = F.relu(
        ordered_q[:, :-1] - ordered_q[:, 1:] + minimum_q_gap
    ).mean()
    selection = grade_ce.new_zeros(())
    selection_sharpness = grade_ce.new_zeros(())
    selection_cohesion = grade_ce.new_zeros(())
    if model.hyperbolic_needle_encoder.structured_needle_selection:
        selections = data["needle_selection_probabilities"].float()
        occupied = data["needle_axis_occupied_bins"].bool()
        selected = selections[occupied].clamp(1e-6, 1 - 1e-6)
        selection_sharpness = -(
            selected * selected.log() + (1 - selected) * (1 - selected).log()
        ).mean()
        adjacent = occupied[:, :-1] & occupied[:, 1:]
        selection_cohesion = (
            (selections[:, 1:] - selections[:, :-1]).abs()[adjacent].mean()
            if bool(adjacent.any())
            else selection_sharpness.new_zeros(())
        )
        selection = selection_sharpness + float(
            cfg.get("hyperbolic_selection_cohesion", 0.5)
        ) * selection_cohesion
    weights = cfg.get("hyperbolic_loss_weights", {})
    components = {
        "hyperbolic_grade_ce": grade_ce,
        "hyperbolic_endpoint_ce": endpoint_ce,
        "hyperbolic_angular_ce": angular_ce,
        "hyperbolic_radial": radial,
        "hyperbolic_benign_root": benign_root,
        "hyperbolic_ordinal_rank": ordinal_rank,
        "hyperbolic_hyperpas": hyperpas,
        "hyperbolic_anchor_usage": anchor_usage,
        "hyperbolic_prototype_order": prototype_order,
        "hyperbolic_selection": selection,
        "hyperbolic_selection_sharpness": selection_sharpness,
        "hyperbolic_selection_cohesion": selection_cohesion,
        "hyperbolic_rank_pairs": pair_mask.sum().float(),
        "hyperbolic_predicted_involvement_mean": predicted.mean(),
        "hyperbolic_fusion_scale": data["hyperbolic_fusion_scale"],
    }
    total = sum(
        float(weights.get(name.removeprefix("hyperbolic_"), 0.0)) * value
        for name, value in components.items()
        if name not in {
            "hyperbolic_rank_pairs",
            "hyperbolic_predicted_involvement_mean",
            "hyperbolic_fusion_scale",
        }
    )
    return total, components


def _simplified_hyperbolic_regularization(
    model,
    data: dict[str, Any],
    cfg,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Minimal V8-compatible hyperbolic objective.

    It has four concepts only: three-way geodesic classification, tolerant
    involvement calibration, robust radial ordinality, and one structured
    selection penalty combining sharp gates with longitudinal cohesion.
    Geometry that can be guaranteed by construction is not re-imposed through
    extra alignment, origin, or prototype-order losses.
    """
    targets = prototype_target_labels(model, data).long().to(model.device)
    involvement = data["involvement"].float().to(model.device).clamp(0, 1)
    involvement_valid = data.get("involvement_valid")
    involvement_valid = (
        involvement_valid.bool().to(model.device)
        if involvement_valid is not None
        else torch.ones_like(involvement, dtype=torch.bool)
    )
    logits = data["hyperbolic_three_way_logits"].float()
    counts = torch.bincount(targets, minlength=3).float()
    class_weights = targets.numel() / (3 * counts.clamp_min(1))
    class_weights[counts == 0] = 0
    class_ce = F.cross_entropy(logits, targets, weight=class_weights)

    predicted = data["hyperbolic_predicted_involvement"].float()
    positive_tolerance = float(
        cfg.get("hyperbolic_involvement_tolerance", 0.075)
    )
    benign_tolerance = float(
        cfg.get("hyperbolic_benign_involvement_tolerance", 0.02)
    )
    tolerance = torch.where(
        targets == 0,
        torch.full_like(involvement, benign_tolerance),
        torch.full_like(involvement, positive_tolerance),
    )
    outside_tolerance = F.relu((predicted - involvement).abs() - tolerance)
    involvement_loss = (
        F.smooth_l1_loss(
            outside_tolerance[involvement_valid],
            torch.zeros_like(outside_tolerance[involvement_valid]),
            beta=float(cfg.get("hyperbolic_involvement_huber_beta", 0.1)),
        )
        if bool(involvement_valid.any())
        else outside_tolerance.new_zeros(())
    )

    true_gap = involvement[:, None] - involvement[None, :]
    predicted_gap = predicted[:, None] - predicted[None, :]
    ordinal_mask = (
        involvement_valid[:, None]
        & involvement_valid[None, :]
        & (true_gap.abs() >= float(cfg.get("hyperbolic_rank_min_gap", 0.2)))
    ) & torch.triu(
        torch.ones_like(true_gap, dtype=torch.bool), diagonal=1
    )
    ordinal = (
        F.relu(
            float(cfg.get("hyperbolic_rank_margin", 0.05))
            - true_gap[ordinal_mask].sign() * predicted_gap[ordinal_mask]
        ).mean()
        if bool(ordinal_mask.any())
        else involvement_loss.new_zeros(())
    )

    selections = data["needle_selection_probabilities"].float()
    occupied = data["needle_axis_occupied_bins"].bool()
    selected = selections[occupied].clamp(1e-6, 1 - 1e-6)
    sharpness = -(
        selected * selected.log() + (1 - selected) * (1 - selected).log()
    ).mean()
    adjacent = occupied[:, :-1] & occupied[:, 1:]
    cohesion = (
        (selections[:, 1:] - selections[:, :-1]).abs()[adjacent].mean()
        if bool(adjacent.any())
        else sharpness.new_zeros(())
    )
    selection = sharpness + float(
        cfg.get("hyperbolic_selection_cohesion", 0.5)
    ) * cohesion

    weights = cfg.get("hyperbolic_loss_weights", {})
    total = (
        float(weights.get("class_ce", 0.5)) * class_ce
        + float(weights.get("involvement", 0.5)) * involvement_loss
        + float(weights.get("ordinal", 0.1)) * ordinal
        + float(weights.get("selection", 0.02)) * selection
    )
    return total, {
        "hyperbolic_class_ce": class_ce,
        "hyperbolic_involvement": involvement_loss,
        "hyperbolic_ordinal": ordinal,
        "hyperbolic_selection": selection,
        "hyperbolic_selection_sharpness": sharpness,
        "hyperbolic_selection_cohesion": cohesion,
        "hyperbolic_rank_pairs": ordinal_mask.sum().float(),
        "hyperbolic_predicted_involvement_mean": predicted.mean(),
        "hyperbolic_fusion_scale": data["hyperbolic_fusion_scale"],
    }


def prototype_regularization(
    model,
    data: dict[str, Any],
    coefficients: dict[str, float],
    prototype_ce_weight: float = 1.0,
    involvement_weight_classification: bool = True,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    labels = prototype_target_labels(model, data).to(model.device)
    involvement = data["involvement"].float().to(model.device)
    involvement_valid = data.get("involvement_valid")
    involvement_valid = (
        involvement_valid.bool().to(model.device)
        if involvement_valid is not None
        else torch.ones_like(involvement, dtype=torch.bool)
    )
    components = model.prototype_losses(
        labels,
        data["matched_values"],
        involvement,
        data.get("grade_group"),
        data.get("occurrence_maps"),
        involvement_valid,
    )
    counts = torch.bincount(labels, minlength=model.num_classes).float()
    class_weights = labels.numel() / (
        model.num_classes * counts.clamp_min(1.0)
    )
    class_weights[counts == 0] = 0
    prototype_ce = F.cross_entropy(
        data["prototype_class_logits"].float(),
        labels,
        weight=class_weights,
        reduction="none",
    )
    geometry_weight = torch.where(
        labels == 0,
        torch.ones_like(involvement),
        torch.where(
            involvement_valid,
            involvement.clamp(min=0.1, max=1.0),
            torch.ones_like(involvement),
        ),
    )
    if involvement_weight_classification:
        # Match PyTorch's weighted-mean convention: the denominator is the
        # applied sample weight, not merely the sample count/geometry weight.
        # The former implementation made the CE scale depend on minibatch
        # composition and disagreed with every endpoint loss below.
        effective_weight = class_weights[labels] * geometry_weight
        components["prototype_ce"] = (
            prototype_ce * geometry_weight
        ).sum() / effective_weight.sum().clamp_min(1e-8)
    else:
        components["prototype_ce"] = F.cross_entropy(
            data["prototype_class_logits"].float(),
            labels,
            weight=class_weights,
        )
    total = components["prototype_ce"] * prototype_ce_weight
    if "occurrence_conditioned_logits_pre_hyperbolic" in data:
        components["occurrence_conditioned_ce"] = F.cross_entropy(
            data["occurrence_conditioned_logits_pre_hyperbolic"].float(),
            labels,
            weight=class_weights,
        )
        components["occurrence_image_ce"] = F.cross_entropy(
            data["occurrence_image_logits_pre_hyperbolic"].float(),
            labels,
            weight=class_weights,
        )
        total = total + coefficients.get(
            "occurrence_conditioned_ce", 0.0
        ) * components["occurrence_conditioned_ce"]
        total = total + coefficients.get(
            "occurrence_image_ce", 0.0
        ) * components["occurrence_image_ce"]
    else:
        zero = data["matched_values"].new_zeros(())
        components["occurrence_conditioned_ce"] = zero
        components["occurrence_image_ce"] = zero
    if getattr(model, "classification_target", "pca") in {"three_way", "isup"}:
        logits = data["prototype_class_logits"]
        pca_labels = data["label"].long().to(logits.device)
        cspca_labels = (data["grade_group"] > 2).long().to(logits.device)

        def balanced_binary_ce(binary_logits, binary_labels):
            counts = torch.bincount(binary_labels, minlength=2).float()
            weights = binary_labels.numel() / (2 * counts.clamp_min(1.0))
            weights[counts == 0] = 0
            return F.cross_entropy(binary_logits, binary_labels, weight=weights)

        pca_logits = torch.stack(
            (logits[:, 0], torch.logsumexp(logits[:, 1:], dim=1)), dim=1
        )
        cspca_logits = (
            torch.stack(
                (
                    torch.logsumexp(logits[:, :3], dim=1),
                    torch.logsumexp(logits[:, 3:], dim=1),
                ),
                dim=1,
            )
            if model.classification_target == "isup"
            else torch.stack(
                (torch.logsumexp(logits[:, :2], dim=1), logits[:, 2]), dim=1
            )
        )
        components["pca_endpoint_ce"] = balanced_binary_ce(pca_logits, pca_labels)
        components["cspca_endpoint_ce"] = balanced_binary_ce(
            cspca_logits, cspca_labels
        )
        involvement = data["involvement"].float().to(logits.device)
        high_keep = (pca_labels == 0) | (
            involvement_valid & (involvement >= 0.4)
        )
        components["high_involvement_endpoint_ce"] = (
            0.5
            * (
                balanced_binary_ce(pca_logits[high_keep], pca_labels[high_keep])
                + balanced_binary_ce(
                    cspca_logits[high_keep], cspca_labels[high_keep]
                )
            )
            if bool(high_keep.any())
            else logits.new_zeros(())
        )
        total = total + coefficients.get("pca_endpoint_ce", 0.0) * components[
            "pca_endpoint_ce"
        ]
        total = total + coefficients.get("cspca_endpoint_ce", 0.0) * components[
            "cspca_endpoint_ce"
        ]
        total = total + coefficients.get(
            "high_involvement_endpoint_ce", 0.0
        ) * components["high_involvement_endpoint_ce"]
    else:
        zero = data["matched_values"].new_zeros(())
        components["pca_endpoint_ce"] = zero
        components["cspca_endpoint_ce"] = zero
        components["high_involvement_endpoint_ce"] = zero
    if "clinical_logit" in data or "prompt_class_logits" in data:
        image_ce = F.cross_entropy(
            data["prototype_class_logits_image"].float(),
            labels,
            weight=class_weights,
            reduction="none",
        )
        components["image_only_ce"] = (
            (image_ce * geometry_weight).sum()
            / geometry_weight.sum().clamp_min(1e-8)
            if involvement_weight_classification
            else image_ce.mean()
        )
        # Preserve a independently discriminative visual explanation instead
        # of allowing clinical metadata to become a shortcut.
        total = total + coefficients.get("image_only_ce", 0.0) * components[
            "image_only_ce"
        ]
        if "prompt_class_logits" in data:
            conditioned_ce = F.cross_entropy(
                data["prototype_class_logits_conditioned"].float(),
                labels,
                weight=class_weights,
            )
            prompt_ce = F.cross_entropy(
                data["prompt_class_logits"].float(), labels, weight=class_weights
            )
            components["conditioned_image_ce"] = conditioned_ce
            components["prompt_only_ce"] = prompt_ce
            total = total + coefficients.get(
                "conditioned_image_ce", 0.0
            ) * conditioned_ce
            total = total + coefficients.get("prompt_only_ce", 0.0) * prompt_ce
        else:
            components["conditioned_image_ce"] = data["matched_values"].new_zeros(())
            components["prompt_only_ce"] = data["matched_values"].new_zeros(())
    else:
        components["image_only_ce"] = data["matched_values"].new_zeros(())
        components["conditioned_image_ce"] = data["matched_values"].new_zeros(())
        components["prompt_only_ce"] = data["matched_values"].new_zeros(())
    for name in (
        "cluster",
        "separation",
        "grade_cluster",
        "coherence",
        "orthogonality",
        "coverage",
        "anchor",
        "slot_sparsity",
        "slot_binarization",
        "l1",
        "grade_l1",
        "occurrence_overlap",
        "occurrence_entropy",
        "occurrence_region_area",
        "occurrence_region_compactness",
        "occurrence_region_cohesion",
        "occurrence_region_tv",
    ):
        total = total + coefficients.get(name, 0.0) * components[name]
    return total, components


@torch.no_grad()
def project_prototypes(
    model,
    loader,
    device: torch.device,
    output_json: str | Path | None = None,
    min_cancer_involvement: float = 0.4,
) -> list[dict[str, Any]]:
    """Project every prototype onto class-specific valid training tokens."""
    model.eval()
    layer = model.prototype_layer
    num_prototypes = layer.num_prototypes
    best_activation = torch.full((num_prototypes,), -torch.inf)
    projected = layer.prototype_vectors.detach().cpu().clone()
    occurrence_mode = getattr(model, "occurrence_matcher", None) is not None
    num_uncertainty = int(getattr(model, "num_uncertainty_prototypes", 0))
    uncertainty_best = torch.full((num_uncertainty,), -torch.inf)
    uncertainty_projected = (
        model.occurrence_matcher.uncertainty_vectors.detach().cpu().clone()
        if num_uncertainty
        else None
    )
    uncertainty_records: list[dict[str, Any] | None] = [None] * num_uncertainty
    records: list[dict[str, Any] | None] = [None] * num_prototypes
    occurrence_projection_candidates: list[list[dict[str, Any]]] = [
        [] for _ in range(num_prototypes)
    ]
    hyper_encoder = getattr(model, "hyperbolic_needle_encoder", None)
    hyper_records: list[dict[str, Any] | None] = []
    hyper_best: torch.Tensor | None = None
    hyper_fallback_records: list[dict[str, Any] | None] = []
    hyper_fallback_best: torch.Tensor | None = None
    hyper_projected_directions: torch.Tensor | None = None
    hyper_projected_q: torch.Tensor | None = None
    hyper_candidates: list[list[dict[str, Any]]] = []
    if hyper_encoder is not None:
        hyper_shape = (
            len(hyper_encoder.target_values),
            hyper_encoder.prototypes_per_grade,
        )
        hyper_best = torch.full(hyper_shape, torch.inf)
        hyper_fallback_best = torch.full(hyper_shape, torch.inf)
        hyper_records = [None] * (hyper_shape[0] * hyper_shape[1])
        hyper_fallback_records = [None] * len(hyper_records)
        hyper_candidates = [[] for _ in hyper_records]
        hyper_projected_directions = (
            hyper_encoder.prototype_directions.detach().cpu().clone()
        )
        _, _, initial_q = hyper_encoder.prototype_points()
        hyper_projected_q = initial_q.detach().cpu().clone()
    needle_head = (
        getattr(hyper_encoder, "needle_prototype_head", None)
        if hyper_encoder is not None
        else None
    )
    needle_candidates: list[list[dict[str, Any]]] = []
    needle_records: list[dict[str, Any] | None] = []
    if needle_head is not None:
        total = needle_head.num_classes * needle_head.prototypes_per_class
        needle_candidates = [[] for _ in range(total)]
        needle_records = [None] * total
    active_slots = layer.slot_weights().detach().cpu().squeeze(0) >= 0.5
    sample_offset = 0
    core_to_index = _dataset_core_index(loader)

    for batch in tqdm(loader, desc="Project prototypes"):
        original_batch_size = len(batch["bmode"])
        batch, skipped = filter_invalid_region_samples(batch)
        if batch is None:
            sample_offset += original_batch_size
            continue
        image = batch["bmode"].to(device)
        prostate_mask = batch["prostate_mask"].to(device)
        needle_mask = batch["needle_mask"].to(device)
        labels = prototype_target_labels(model, batch).cpu()
        involvement = batch["involvement"].float()
        involvement_valid = batch.get("involvement_valid")
        involvement_valid = (
            involvement_valid.bool().cpu()
            if involvement_valid is not None
            else torch.ones_like(involvement, dtype=torch.bool)
        )
        grade_groups = batch["grade_group"].long()
        outputs = model(
            image,
            prostate_mask,
            needle_mask,
            clinical_prompt_tensor(model, batch, device),
        )
        activations = outputs["prototype_activations"].detach().cpu()
        indices = outputs["matched_indices"].detach().cpu()
        features = outputs["encoder_features"].detach().cpu().flatten(2)
        pooled_features = outputs.get("occurrence_pooled_features")
        if pooled_features is not None:
            pooled_features = pooled_features.detach().cpu()
        grid_h, grid_w = outputs["encoder_features"].shape[-2:]

        core_ids = batch.get("core_id", [str(sample_offset + i) for i in range(len(image))])
        if needle_head is not None:
            needle_activations = outputs[
                "needle_prototype_activations"
            ].detach().cpu()
            needle_slot_features = outputs[
                "needle_prototype_slot_features"
            ].detach().cpu()
            needle_slot_indices = outputs[
                "needle_prototype_slot_indices"
            ].detach().cpu()
            raw_centers = batch.get("center")
            centers = (
                [str(value) for value in raw_centers]
                if raw_centers is not None
                else [str(core_id).split("-")[0] for core_id in core_ids]
            )
            for class_index in range(needle_head.num_classes):
                class_candidates = torch.nonzero(labels == class_index).flatten()
                if class_candidates.numel() == 0:
                    continue
                for prototype_index in range(needle_head.prototypes_per_class):
                    flat_index = (
                        class_index * needle_head.prototypes_per_class
                        + prototype_index
                    )
                    for batch_index in class_candidates.tolist():
                        core_id = str(core_ids[batch_index])
                        slot_indices = needle_slot_indices[
                            batch_index, class_index, prototype_index
                        ]
                        needle_candidates[flat_index].append(
                            {
                                "needle_prototype_index": flat_index,
                                "prototype_class": class_index,
                                "class_prototype_index": prototype_index,
                                "core_id": core_id,
                                "dataset_index": core_to_index.get(
                                    core_id, sample_offset + batch_index
                                ),
                                "center": centers[batch_index],
                                "activation": float(
                                    needle_activations[
                                        batch_index, class_index, prototype_index
                                    ]
                                ),
                                "involvement": float(involvement[batch_index]),
                                "grade_group": int(grade_groups[batch_index]),
                                "axis_bins": int(hyper_encoder.axis_bins),
                                "slot_bin_indices": [
                                    int(value) for value in slot_indices.tolist()
                                ],
                                "_projected_vectors": needle_slot_features[
                                    batch_index, class_index, prototype_index
                                ],
                            }
                        )
        if hyper_encoder is not None:
            hyper_distances = outputs["hyperbolic_prototype_distances"].detach().cpu()
            hyper_directions = outputs["hyperbolic_direction"].detach().cpu()
            hyper_predicted_q = outputs[
                "hyperbolic_predicted_involvement"
            ].detach().cpu()
            _, _, prototype_q_device = hyper_encoder.prototype_points()
            prototype_q = prototype_q_device.detach().cpu()
            if hyper_encoder.target_mode == "three_way":
                hyper_targets = labels
            elif hyper_encoder.target_mode == "grade_bands_01_23_45":
                hyper_targets = hyper_encoder.target_indices_from_grades(
                    grade_groups
                ).cpu()
            else:
                hyper_targets = grade_groups
            for target_index, target_value in enumerate(hyper_encoder.target_values):
                target_candidates = torch.nonzero(
                    (hyper_targets == target_value) & involvement_valid
                ).flatten()
                if target_candidates.numel() == 0:
                    continue
                for severity_index in range(hyper_encoder.prototypes_per_grade):
                    flat_index = (
                        target_index * hyper_encoder.prototypes_per_grade
                        + severity_index
                    )
                    candidate_distances = hyper_distances[
                        target_candidates, target_index, severity_index
                    ]
                    fallback_local = int(candidate_distances.argmin())
                    fallback_batch = int(target_candidates[fallback_local])
                    fallback_distance = candidate_distances[fallback_local]
                    record_base = {
                        "hyperbolic_prototype_index": flat_index,
                        "target_index": target_index,
                        "prototype_target_mode": hyper_encoder.target_mode,
                        "prototype_target": int(target_value),
                        "severity_index": severity_index,
                        "prototype_involvement_target": float(
                            prototype_q[target_index, severity_index]
                        ),
                    }
                    if hyper_encoder.unique_projection:
                        for candidate_index in target_candidates.tolist():
                            candidate_distance = float(
                                hyper_distances[
                                    candidate_index, target_index, severity_index
                                ]
                            )
                            q_difference_value = abs(
                                float(involvement[candidate_index])
                                - float(prototype_q[target_index, severity_index])
                            )
                            predicted_q_difference = abs(
                                float(hyper_predicted_q[candidate_index])
                                - float(prototype_q[target_index, severity_index])
                            )
                            used_q_fallback = (
                                q_difference_value
                                > hyper_encoder.projection_q_tolerance
                            )
                            core_id = str(core_ids[candidate_index])
                            # A large deterministic fallback penalty preserves
                            # the original q-tolerance semantics whenever a
                            # feasible unique assignment exists.
                            projection_cost = (
                                candidate_distance
                                + hyper_encoder.projection_q_weight
                                * (q_difference_value + predicted_q_difference)
                                + (10.0 if used_q_fallback else 0.0)
                            )
                            hyper_candidates[flat_index].append(
                                {
                                    **record_base,
                                    "core_id": core_id,
                                    "dataset_index": core_to_index.get(
                                        core_id, sample_offset + candidate_index
                                    ),
                                    "distance": candidate_distance,
                                    "involvement": float(
                                        involvement[candidate_index]
                                    ),
                                    "predicted_involvement": float(
                                        hyper_predicted_q[candidate_index]
                                    ),
                                    "grade_group": int(
                                        grade_groups[candidate_index]
                                    ),
                                    "q_difference": q_difference_value,
                                    "predicted_q_difference": predicted_q_difference,
                                    "projection_cost": projection_cost,
                                    "used_q_fallback": used_q_fallback,
                                    "direction": hyper_directions[
                                        candidate_index
                                    ].tolist(),
                                }
                            )
                    if fallback_distance < hyper_fallback_best[target_index, severity_index]:
                        hyper_fallback_best[target_index, severity_index] = fallback_distance
                        core_id = str(core_ids[fallback_batch])
                        hyper_fallback_records[flat_index] = {
                            **record_base,
                            "core_id": core_id,
                            "dataset_index": core_to_index.get(
                                core_id, sample_offset + fallback_batch
                            ),
                            "distance": float(fallback_distance),
                            "involvement": float(involvement[fallback_batch]),
                            "predicted_involvement": float(
                                hyper_predicted_q[fallback_batch]
                            ),
                            "grade_group": int(grade_groups[fallback_batch]),
                            "used_q_fallback": True,
                            "direction": hyper_directions[fallback_batch].tolist(),
                        }
                    q_difference = (
                        involvement[target_candidates]
                        - prototype_q[target_index, severity_index]
                    ).abs()
                    allowed = q_difference <= hyper_encoder.projection_q_tolerance
                    if not bool(allowed.any()):
                        continue
                    allowed_candidates = target_candidates[allowed]
                    allowed_distances = hyper_distances[
                        allowed_candidates, target_index, severity_index
                    ]
                    local = int(allowed_distances.argmin())
                    batch_idx_hyper = int(allowed_candidates[local])
                    distance = allowed_distances[local]
                    if distance >= hyper_best[target_index, severity_index]:
                        continue
                    hyper_best[target_index, severity_index] = distance
                    core_id = str(core_ids[batch_idx_hyper])
                    hyper_records[flat_index] = {
                        **record_base,
                        "core_id": core_id,
                        "dataset_index": core_to_index.get(
                            core_id, sample_offset + batch_idx_hyper
                        ),
                        "distance": float(distance),
                        "involvement": float(involvement[batch_idx_hyper]),
                        "predicted_involvement": float(
                            hyper_predicted_q[batch_idx_hyper]
                        ),
                        "grade_group": int(grade_groups[batch_idx_hyper]),
                        "used_q_fallback": False,
                        "direction": hyper_directions[batch_idx_hyper].tolist(),
                    }
        for prototype_idx in range(num_prototypes):
            target_class = int(layer.prototype_class_identity[prototype_idx].argmax())
            candidate_mask = labels == target_class
            target_grade = int(layer.prototype_grade_targets[prototype_idx])
            if target_grade >= 0:
                candidate_mask = candidate_mask & (grade_groups == target_grade)
            if target_class > 0:
                candidate_mask = candidate_mask & (
                    involvement_valid
                    & (involvement >= min_cancer_involvement)
                )
            candidates = torch.nonzero(candidate_mask).flatten()
            if candidates.numel() == 0:
                continue
            candidate_scores = activations[candidates, prototype_idx]
            if bool(getattr(model, "unique_prototype_projection", False)):
                for batch_idx_candidate, score_candidate in zip(
                    candidates.tolist(), candidate_scores.tolist()
                ):
                    token_indices_candidate = indices[
                        batch_idx_candidate, prototype_idx
                    ]
                    vectors = projected[prototype_idx].clone()
                    for slot_idx in range(layer.num_subprototypes):
                        if active_slots[prototype_idx, slot_idx]:
                            token_idx_candidate = int(
                                token_indices_candidate[slot_idx]
                            )
                            vectors[:, slot_idx] = (
                                pooled_features[
                                    batch_idx_candidate, prototype_idx, slot_idx
                                ]
                                if pooled_features is not None
                                else features[
                                    batch_idx_candidate, :, token_idx_candidate
                                ]
                            )
                    core_id_candidate = str(core_ids[batch_idx_candidate])
                    occurrence_projection_candidates[prototype_idx].append(
                        {
                            "prototype_index": prototype_idx,
                            "prototype_class": target_class,
                            "core_id": core_id_candidate,
                            "dataset_index": core_to_index.get(
                                core_id_candidate,
                                sample_offset + batch_idx_candidate,
                            ),
                            "activation": float(score_candidate),
                            "involvement": float(
                                involvement[batch_idx_candidate]
                            ),
                            "grade_group": int(
                                grade_groups[batch_idx_candidate]
                            ),
                            "prototype_grade_target": target_grade,
                            "grid_size": [int(grid_h), int(grid_w)],
                            "token_indices": [
                                int(value)
                                for value in token_indices_candidate.tolist()
                            ],
                            "token_rows": [
                                int(value) // int(grid_w)
                                for value in token_indices_candidate.tolist()
                            ],
                            "token_cols": [
                                int(value) % int(grid_w)
                                for value in token_indices_candidate.tolist()
                            ],
                            "active_slots": [
                                bool(value)
                                for value in active_slots[prototype_idx].tolist()
                            ],
                            "matching_mode": (
                                "occurrence" if occurrence_mode else "deformable"
                            ),
                            "_projected_vectors": vectors,
                        }
                    )
            local_best = int(candidate_scores.argmax())
            batch_idx = int(candidates[local_best])
            score = candidate_scores[local_best]
            if score <= best_activation[prototype_idx]:
                continue

            token_indices = indices[batch_idx, prototype_idx]
            for slot_idx in range(layer.num_subprototypes):
                if active_slots[prototype_idx, slot_idx]:
                    token_idx = int(token_indices[slot_idx])
                    projected[prototype_idx, :, slot_idx] = (
                        pooled_features[batch_idx, prototype_idx, slot_idx]
                        if pooled_features is not None
                        else features[batch_idx, :, token_idx]
                    )
            best_activation[prototype_idx] = score
            core_id = str(core_ids[batch_idx])
            records[prototype_idx] = {
                "prototype_index": prototype_idx,
                "prototype_class": target_class,
                "core_id": core_id,
                "dataset_index": core_to_index.get(
                    core_id, sample_offset + batch_idx
                ),
                "activation": float(score),
                "involvement": float(involvement[batch_idx]),
                "grade_group": int(grade_groups[batch_idx]),
                "prototype_grade_target": target_grade,
                "grid_size": [int(grid_h), int(grid_w)],
                "token_indices": [int(value) for value in token_indices.tolist()],
                "token_rows": [int(value) // int(grid_w) for value in token_indices.tolist()],
                "token_cols": [int(value) % int(grid_w) for value in token_indices.tolist()],
                "active_slots": [
                    bool(value) for value in active_slots[prototype_idx].tolist()
                ],
                "matching_mode": "occurrence" if occurrence_mode else "deformable",
            }

        if num_uncertainty:
            uncertainty_activations = outputs["uncertainty_activations"].detach().cpu()
            uncertainty_indices = outputs["uncertainty_matched_indices"].detach().cpu()
            uncertainty_pooled = outputs[
                "uncertainty_occurrence_pooled_features"
            ].detach().cpu()
            for uncertainty_idx in range(num_uncertainty):
                batch_idx = int(uncertainty_activations[:, uncertainty_idx].argmax())
                score = uncertainty_activations[batch_idx, uncertainty_idx]
                if score <= uncertainty_best[uncertainty_idx]:
                    continue
                uncertainty_projected[uncertainty_idx] = uncertainty_pooled[
                    batch_idx, uncertainty_idx
                ].transpose(0, 1)
                uncertainty_best[uncertainty_idx] = score
                token_indices = uncertainty_indices[batch_idx, uncertainty_idx]
                core_id = str(core_ids[batch_idx])
                uncertainty_records[uncertainty_idx] = {
                    "prototype_index": num_prototypes + uncertainty_idx,
                    "uncertainty_index": uncertainty_idx,
                    "prototype_class": -1,
                    "prototype_grade_target": -1,
                    "is_uncertainty": True,
                    "core_id": core_id,
                    "dataset_index": core_to_index.get(core_id, sample_offset + batch_idx),
                    "activation": float(score),
                    "involvement": float(involvement[batch_idx]),
                    "grade_group": int(grade_groups[batch_idx]),
                    "grid_size": [int(grid_h), int(grid_w)],
                    "token_indices": [int(value) for value in token_indices.tolist()],
                    "token_rows": [int(value) // int(grid_w) for value in token_indices.tolist()],
                    "token_cols": [int(value) % int(grid_w) for value in token_indices.tolist()],
                    "active_slots": [True] * layer.num_subprototypes,
                    "matching_mode": "occurrence",
                }
        sample_offset += original_batch_size

    if bool(getattr(model, "unique_prototype_projection", False)):
        uniquely_assigned = _assign_unique_occurrence_projection_records(
            model, occurrence_projection_candidates
        )
        for prototype_index, record in enumerate(uniquely_assigned):
            if record is None:
                logging.warning(
                    "Unique projection had no feasible assignment for prototype %d; "
                    "retaining its independent projection fallback.",
                    prototype_index,
                )
                continue
            projected[prototype_index] = record.pop("_projected_vectors")
            records[prototype_index] = record

    missing = [idx for idx, record in enumerate(records) if record is None]
    if missing:
        logging.warning("Could not project prototypes: %s", missing)
    layer.prototype_vectors.copy_(projected.to(layer.prototype_vectors.device))
    if needle_head is not None:
        needle_records = _assign_center_balanced_needle_projection_records(
            needle_candidates,
            needle_head.num_classes,
            needle_head.prototypes_per_class,
            hyper_encoder.projection_candidate_limit,
        )
        projected_needle = needle_head.prototype_vectors.detach().cpu().clone()
        for flat_index, record in enumerate(needle_records):
            if record is None:
                continue
            class_index = int(record["prototype_class"])
            prototype_index = int(record["class_prototype_index"])
            projected_needle[class_index, prototype_index] = record.pop(
                "_projected_vectors"
            )
        missing_needle = [
            index for index, record in enumerate(needle_records)
            if record is None
        ]
        if missing_needle:
            logging.warning(
                "Could not project needle-axis prototypes: %s", missing_needle
            )
        needle_head.prototype_vectors.copy_(
            projected_needle.to(needle_head.prototype_vectors.device)
        )
    if hyper_encoder is not None:
        if hyper_encoder.unique_projection:
            hyper_records = _assign_unique_hyperbolic_projection_records(
                hyper_candidates,
                len(hyper_encoder.target_values),
                hyper_encoder.prototypes_per_grade,
                hyper_encoder.projection_candidate_limit,
            )
        for flat_index, record in enumerate(hyper_records):
            if record is None:
                record = hyper_fallback_records[flat_index]
                hyper_records[flat_index] = record
            if record is None:
                continue
            target_index = int(record["target_index"])
            severity_index = int(record["severity_index"])
            if hyper_encoder.shared_class_directions:
                # Both severity anchors share one ray by construction. Use the
                # mean of their projected real-core directions below.
                pass
            else:
                hyper_projected_directions[target_index, severity_index] = (
                    torch.as_tensor(record["direction"])
                )
            hyper_projected_q[target_index, severity_index] = float(
                record["predicted_involvement"]
            )
        if hyper_encoder.shared_class_directions:
            for target_index in range(len(hyper_encoder.target_values)):
                directions_for_target = [
                    torch.as_tensor(record["direction"])
                    for record in hyper_records
                    if record is not None
                    and int(record["target_index"]) == target_index
                ]
                if directions_for_target:
                    hyper_projected_directions[target_index] = F.normalize(
                        torch.stack(directions_for_target).mean(0), dim=0
                    )
        hyper_encoder.prototype_directions.copy_(
            hyper_projected_directions.to(hyper_encoder.prototype_directions.device)
        )
        if hyper_encoder.low_band_prototype_radius_logits is not None:
            low_band_q = hyper_projected_q[:1].clamp(1e-4, 1 - 1e-4)
            hyper_encoder.low_band_prototype_radius_logits.copy_(
                torch.logit(low_band_q).to(
                    hyper_encoder.low_band_prototype_radius_logits.device
                )
            )
        cancer_q = hyper_projected_q[1:].clamp(1e-4, 1 - 1e-4)
        hyper_encoder.cancer_prototype_radius_logits.copy_(
            torch.logit(cancer_q).to(
                hyper_encoder.cancer_prototype_radius_logits.device
            )
        )
    if num_uncertainty:
        model.occurrence_matcher.uncertainty_vectors.copy_(
            uncertainty_projected.to(model.occurrence_matcher.uncertainty_vectors.device)
        )
    layer.set_projection_anchors()
    final_records = [
        record
        if record is not None
        else {
            "prototype_index": idx,
            "prototype_class": int(layer.prototype_class_identity[idx].argmax()),
            "prototype_grade_target": int(layer.prototype_grade_targets[idx]),
            "missing": True,
        }
        for idx, record in enumerate(records)
    ]
    final_records.extend(
        record
        if record is not None
        else {
            "prototype_index": num_prototypes + idx,
            "uncertainty_index": idx,
            "prototype_class": -1,
            "prototype_grade_target": -1,
            "is_uncertainty": True,
            "missing": True,
        }
        for idx, record in enumerate(uncertainty_records)
    )
    if output_json is not None:
        output_path = Path(output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(final_records, indent=2))
        if hyper_encoder is not None:
            hyper_path = output_path.with_name(
                f"{output_path.stem}_hyperbolic{output_path.suffix}"
            )
            serializable_hyper_records = []
            for index, record in enumerate(hyper_records):
                if record is None:
                    serializable_hyper_records.append(
                        {"hyperbolic_prototype_index": index, "missing": True}
                    )
                else:
                    clean = dict(record)
                    clean.pop("direction", None)
                    serializable_hyper_records.append(clean)
            hyper_path.write_text(json.dumps(serializable_hyper_records, indent=2))
        if needle_head is not None:
            needle_path = output_path.with_name(
                f"{output_path.stem}_needle_axis{output_path.suffix}"
            )
            serializable_needle_records = []
            for index, record in enumerate(needle_records):
                if record is None:
                    serializable_needle_records.append(
                        {"needle_prototype_index": index, "missing": True}
                    )
                else:
                    serializable_needle_records.append(dict(record))
            needle_path.write_text(
                json.dumps(serializable_needle_records, indent=2)
            )
    return final_records



def json_safe_metrics(metrics: dict[str, Any]) -> dict[str, Any]:
    result = {}
    for key, value in metrics.items():
        if isinstance(value, (float, int, str, bool)) or value is None:
            result[key] = value
        elif isinstance(value, np.generic):
            result[key] = value.item()
    return result
