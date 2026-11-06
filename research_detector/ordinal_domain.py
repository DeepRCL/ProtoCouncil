"""Dual-branch ordinal/domain-aware objectives for ProtoProstate V17.

The implementation follows the actual PANDA ``train_v3.py`` behavior: squared
CDF EMD, same-grade supervised contrastive positives, ordinally weighted
negatives, optional cross-center boosts for both negatives and same-grade
positives, hard-example weights, inverse-frequency weights, a classification-
only curriculum, and adaptive auxiliary weighting. Historical queue entries
are always detached; only current-batch anchors receive gradients.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch
from torch.nn import functional as F

from .prototype_utils import prototype_target_labels


@dataclass
class DetachedOrdinalQueue:
    capacity: int
    embeddings: torch.Tensor | None = None
    grades: torch.Tensor | None = None
    centers: torch.Tensor | None = None

    def add(
        self, embeddings: torch.Tensor, grades: torch.Tensor, centers: torch.Tensor
    ) -> None:
        z = F.normalize(embeddings.detach().float(), dim=1).cpu()
        y = grades.detach().long().cpu()
        d = centers.detach().long().cpu()
        self.embeddings = z if self.embeddings is None else torch.cat((self.embeddings, z))
        self.grades = y if self.grades is None else torch.cat((self.grades, y))
        self.centers = d if self.centers is None else torch.cat((self.centers, d))
        if len(self.grades) > self.capacity:
            keep = slice(-self.capacity, None)
            self.embeddings = self.embeddings[keep]
            self.grades = self.grades[keep]
            self.centers = self.centers[keep]


@dataclass
class OrdinalDomainState:
    capacity: int
    local_queue: DetachedOrdinalQueue = field(init=False)
    global_queue: DetachedOrdinalQueue = field(init=False)
    center_ids: dict[str, int] = field(default_factory=dict)
    ratio_ema: float | None = None


    def encode_centers(self, values: list[Any], device: torch.device) -> torch.Tensor:
        encoded = []
        for value in values:
            key = str(value)
            if key not in self.center_ids:
                self.center_ids[key] = len(self.center_ids)
            encoded.append(self.center_ids[key])
        return torch.as_tensor(encoded, device=device, dtype=torch.long)


def _state(model, capacity: int) -> OrdinalDomainState:
    state = getattr(model, "_ordinal_domain_state", None)
    if state is None or state.capacity != capacity:
        state = OrdinalDomainState(capacity)
        model._ordinal_domain_state = state
    return state


def _weighted_mean(values: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    return (values * weights).sum() / weights.sum().clamp_min(1e-8)


def _curriculum(
    count: int, epoch: int, start: int, decay: float, minimum: float, device
) -> torch.Tensor:
    if epoch < start:
        return torch.ones(count, device=device)
    fraction = torch.arange(count, device=device).float() / max(count - 1, 1)
    # Mildly downweight easy low states while preserving every classification
    # example. Contrastive weights deliberately do not use this curriculum.
    floor = 0.55 + 0.45 * fraction
    progress = 1.0 - torch.exp(
        torch.tensor(-decay * (epoch - start + 1), device=device)
    )
    return (1.0 - progress * (1.0 - floor)).clamp_min(minimum)


def _class_weights(cfg, key: str, count: int, device) -> torch.Tensor:
    configured = list(cfg.get(key, [1.0] * count))
    if len(configured) != count:
        raise ValueError(f"{key} has {len(configured)} values, expected {count}.")
    weights = torch.as_tensor(configured, device=device, dtype=torch.float32)
    return weights / weights.mean().clamp_min(1e-8)


def _emd_per_sample(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    probabilities = torch.softmax(logits.float(), dim=1)
    target_distribution = F.one_hot(targets, logits.shape[1]).float()
    return (
        probabilities.cumsum(1) - target_distribution.cumsum(1)
    ).square().sum(1)


def _supcon_with_memory(
    embeddings: torch.Tensor,
    grades: torch.Tensor,
    centers: torch.Tensor,
    queue: DetachedOrdinalQueue,
    sample_weights: torch.Tensor,
    temperature: float,
    negative_strength: float,
    domain_aware: bool,
    domain_weight: float,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    current = F.normalize(embeddings.float(), dim=1)
    memory_count = 0 if queue.grades is None else len(queue.grades)
    if memory_count:
        candidates = torch.cat((current, queue.embeddings.to(current.device)), dim=0)
        candidate_grades = torch.cat((grades, queue.grades.to(grades.device)))
        candidate_centers = torch.cat((centers, queue.centers.to(centers.device)))
    else:
        candidates = current
        candidate_grades = grades
        candidate_centers = centers

    similarities = current @ candidates.T / max(float(temperature), 1e-6)
    allowed = torch.ones_like(similarities, dtype=torch.bool)
    allowed[:, : len(current)].fill_diagonal_(False)
    same_grade = grades[:, None] == candidate_grades[None]
    positives = same_grade & allowed
    negatives = (~same_grade) & allowed
    distance = (grades[:, None].float() - candidate_grades[None].float()).abs()
    negative_weights = 1.0 + float(negative_strength) * (distance / 5.0).clamp(0, 1)
    negative_weights = negative_weights * negatives.float()

    cross_center = centers[:, None] != candidate_centers[None]
    if domain_aware:
        negative_weights = negative_weights * (
            1.0 + float(domain_weight) * cross_center.float()
        )
        positive_weights = positives.float() * (
            1.0 + float(domain_weight) * cross_center.float()
        )
    else:
        positive_weights = positives.float()

    # Match train_v3 exactly: domain weighting changes the positive aggregate,
    # while positives retain unit denominator weight.
    denominator_weights = positives.float() + negative_weights
    stable = similarities - similarities.max(dim=1, keepdim=True).values.detach()
    denominator = (
        stable.exp() * allowed.float() * denominator_weights
    ).sum(1).clamp_min(1e-8)
    log_probability = stable - denominator.log()[:, None]
    positive_count = positives.sum(1)
    valid = positive_count > 0
    if bool(valid.any()):
        positive_weight_sum = positive_weights.sum(1).clamp_min(1e-8)
        per_anchor = -(
            log_probability * positive_weights
        ).sum(1) / positive_weight_sum
        normalized_weights = sample_weights[valid]
        normalized_weights = normalized_weights / normalized_weights.mean().clamp_min(1e-8)
        loss = (per_anchor[valid] * normalized_weights).mean()
    else:
        loss = current.sum() * 0.0

    queue.add(current, grades, centers)
    return loss, {
        "valid_anchors": valid.sum().float(),
        "positive_pairs": positives.sum().float(),
        "cross_center_positive_pairs": (positives & cross_center).sum().float(),
        "memory_entries": current.new_tensor(float(memory_count)),
    }


def ordinal_domain_regularization(
    model,
    data: dict[str, Any],
    cfg,
    stage: str,
    epoch: int,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    weight = float(cfg.get("ordinal_domain_loss_weight", 0.0))
    active_stage = str(cfg.get("ordinal_domain_stage", "joint"))
    if weight <= 0 or stage != active_stage:
        zero = data["prototype_activations"].new_zeros(())
        return zero, {}
    required = (
        "prototype_class_logits_pre_hyperbolic",
        "hyperbolic_grade_logits",
        "hyperbolic_direction",
        "hyperbolic_radius",
    )
    missing = [key for key in required if key not in data]
    if missing:
        raise RuntimeError(f"Ordinal/domain objective misses model outputs: {missing}")

    local_logits = data["prototype_class_logits_pre_hyperbolic"].float()
    global_logits = data["hyperbolic_grade_logits"].float()
    states = prototype_target_labels(model, data).long().to(local_logits.device)
    grades = data["grade_group"].long().to(local_logits.device)
    target_values = torch.as_tensor(
        model.hyperbolic_needle_encoder.target_values,
        device=grades.device,
        dtype=grades.dtype,
    )
    matches = grades[:, None] == target_values[None]
    if not bool(matches.any(1).all()):
        raise ValueError(
            f"Unsupported V17 grades: {grades[~matches.any(1)].unique().tolist()}"
        )
    global_targets = matches.float().argmax(1).long()

    state = _state(model, int(cfg.get("ordinal_memory_size", 256)))
    raw_centers = data.get("center")
    if raw_centers is None:
        raw_centers = [str(value).split("-")[0] for value in data["core_id"]]
    centers = state.encode_centers(list(raw_centers), local_logits.device)

    local_inverse = _class_weights(
        cfg, "ordinal_local_inverse_frequency_weights", local_logits.shape[1], local_logits.device
    )
    global_inverse = _class_weights(
        cfg, "ordinal_grade_inverse_frequency_weights", global_logits.shape[1], local_logits.device
    )
    curriculum = bool(cfg.get("ordinal_use_curriculum", True))
    if curriculum:
        local_curriculum = _curriculum(
            local_logits.shape[1], epoch, int(cfg.get("ordinal_curriculum_start", 3)),
            float(cfg.get("ordinal_curriculum_decay", 0.1)),
            float(cfg.get("ordinal_curriculum_min", 0.3)), local_logits.device,
        )
        global_curriculum = _curriculum(
            global_logits.shape[1], epoch, int(cfg.get("ordinal_curriculum_start", 3)),
            float(cfg.get("ordinal_curriculum_decay", 0.1)),
            float(cfg.get("ordinal_curriculum_min", 0.3)), local_logits.device,
        )
    else:
        local_curriculum = torch.ones_like(local_inverse)
        global_curriculum = torch.ones_like(global_inverse)
    local_class_weight = local_inverse * local_curriculum
    global_class_weight = global_inverse * global_curriculum

    local_ce_vector = F.cross_entropy(local_logits, states, reduction="none")
    global_ce_vector = F.cross_entropy(global_logits, global_targets, reduction="none")
    local_emd_vector = _emd_per_sample(local_logits, states)
    global_emd_vector = _emd_per_sample(global_logits, global_targets)
    local_ce = _weighted_mean(local_ce_vector, local_class_weight[states])
    global_ce = _weighted_mean(global_ce_vector, global_class_weight[global_targets])
    local_emd = _weighted_mean(local_emd_vector, local_class_weight[states])
    global_emd = _weighted_mean(global_emd_vector, global_class_weight[global_targets])

    local_p_true = torch.softmax(local_logits, 1).gather(1, states[:, None]).squeeze(1)
    global_p_true = torch.softmax(global_logits, 1).gather(
        1, global_targets[:, None]
    ).squeeze(1)
    # Match train_v3: hard-example and inverse-frequency weights affect SupCon,
    # while benign contrastive pairs are never suppressed by the curriculum.
    local_sample_weight = (1.0 + (1.0 - local_p_true)) * global_inverse[global_targets]
    global_sample_weight = (1.0 + (1.0 - global_p_true)) * global_inverse[global_targets]
    domain_aware = bool(cfg.get("ordinal_domain_aware", False))
    supcon_args = dict(
        temperature=float(cfg.get("ordinal_temperature", 0.07)),
        negative_strength=float(cfg.get("ordinal_negative_strength", 1.0)),
        domain_aware=domain_aware,
        domain_weight=float(cfg.get("ordinal_domain_weight", 1.0)),
    )
    local_supcon, local_pairs = _supcon_with_memory(
        data["prototype_activations"], grades, centers, state.local_queue,
        local_sample_weight, **supcon_args,
    )
    tangent = data["hyperbolic_direction"].float() * data["hyperbolic_radius"].float()[:, None]
    global_supcon, global_pairs = _supcon_with_memory(
        tangent, grades, centers, state.global_queue, global_sample_weight, **supcon_args,
    )

    main = (
        float(cfg.get("ordinal_local_ce_weight", 0.5)) * local_ce
        + float(cfg.get("ordinal_global_ce_weight", 0.5)) * global_ce
        + float(cfg.get("ordinal_local_emd_weight", 0.5)) * local_emd
        + float(cfg.get("ordinal_global_emd_weight", 0.5)) * global_emd
    )
    auxiliary = (
        float(cfg.get("ordinal_local_supcon_weight", 1.0)) * local_supcon
        + float(cfg.get("ordinal_global_supcon_weight", 1.0)) * global_supcon
    )
    start = float(cfg.get("ordinal_aux_weight_start", 0.25))
    end = float(cfg.get("ordinal_aux_weight_end", 0.10))
    decay_end = max(int(cfg.get("ordinal_aux_weight_decay_end", 10)), 1)
    progress = min(max(epoch / decay_end, 0.0), 1.0)
    auxiliary_weight = start + (end - start) * progress
    if bool(cfg.get("ordinal_aux_ratio_adapt", True)):
        current_ratio = float(auxiliary.detach()) / max(
            float((main + auxiliary).detach()), 1e-8
        )
        beta = float(cfg.get("ordinal_aux_ratio_ema_beta", 0.9))
        state.ratio_ema = current_ratio if state.ratio_ema is None else (
            beta * state.ratio_ema + (1.0 - beta) * current_ratio
        )
        target_ratio = float(cfg.get("ordinal_aux_target_ratio", 0.10))
        auxiliary_weight *= target_ratio / max(state.ratio_ema, 1e-8)
        auxiliary_weight = min(max(auxiliary_weight, 0.0), 10.0)
    total = weight * (main + auxiliary_weight * auxiliary)
    components = {
        "ordinal_domain_total": total,
        "ordinal_local_ce": local_ce,
        "ordinal_global_ce": global_ce,
        "ordinal_local_emd": local_emd,
        "ordinal_global_emd": global_emd,
        "ordinal_local_supcon": local_supcon,
        "ordinal_global_supcon": global_supcon,
        "ordinal_aux_weight": total.new_tensor(auxiliary_weight),
        "ordinal_domain_aware": total.new_tensor(float(domain_aware)),
    }
    components.update({f"ordinal_local_{key}": value for key, value in local_pairs.items()})
    components.update({f"ordinal_global_{key}": value for key, value in global_pairs.items()})
    return total, components
