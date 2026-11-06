"""Prototype-based, weakly supervised prostate cancer model.

The prototype matching code is adapted from ProtoViT's ``PPNet``.  The
hard-coded 14x14 grid and CUDA allocations in the original implementation are
removed so that it can operate on MedSAM's 16x16 encoder grid and on CPU.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from protocouncil.models.sam import ClassDecoder
from protocouncil.models.sam import PromptEncoder
from protocouncil.models.sam import TwoWayTransformer
from .registry import create_model, register_model


class _GradientReversal(torch.autograd.Function):
    """Identity in the forward pass and sign-reversed gradient downstream."""


    @staticmethod
    def backward(ctx, gradients: torch.Tensor) -> tuple[torch.Tensor, None]:
        return -ctx.scale * gradients, None


class DeformablePrototypeLayer(nn.Module):
    """ProtoViT-style deformable prototypes over a 2-D token grid."""

    def __init__(
        self,
        feature_dim: int = 256,
        num_classes: int = 2,
        prototypes_per_class: int = 10,
        num_subprototypes: int = 4,
        adjacency_radius: int = 1,
        slot_temperature: float = 1.0,
        coverage_min_cancer_involvement: float = 0.4,
        prototype_grade_groups: list[int] | tuple[int, ...] | None = None,
        connected_greedy: bool = False,
        greedy_connectivity: int = 4,
        greedy_straight_through_temperature: float = 0.0,
    ) -> None:
        super().__init__()
        self.feature_dim = feature_dim
        self.num_classes = num_classes
        self.prototypes_per_class = prototypes_per_class
        self.num_prototypes = num_classes * prototypes_per_class
        self.num_subprototypes = num_subprototypes
        self.adjacency_radius = adjacency_radius
        self.coverage_min_cancer_involvement = coverage_min_cancer_involvement
        self.connected_greedy = bool(connected_greedy)
        if greedy_connectivity not in (4, 8):
            raise ValueError("greedy_connectivity must be 4 or 8.")
        self.greedy_connectivity = int(greedy_connectivity)
        if greedy_straight_through_temperature < 0:
            raise ValueError("greedy_straight_through_temperature must be non-negative.")
        self.greedy_straight_through_temperature = float(
            greedy_straight_through_temperature
        )
        self.register_buffer(
            "_slot_temperature", torch.tensor(float(slot_temperature))
        )
        self._warned_nonadjacent_fallback = False

        self.prototype_vectors = nn.Parameter(
            torch.empty(self.num_prototypes, feature_dim, num_subprototypes)
        )
        nn.init.trunc_normal_(self.prototype_vectors, std=0.02)

        # Start active but not saturated. ProtoViT's sigmoid(0.1 * 100) starts at
        # 0.99995, which produced effectively zero slot gradients on this task.
        # The training script anneals the temperature only during slot learning.
        self.slot_logits = nn.Parameter(
            torch.full((1, self.num_prototypes, num_subprototypes), 0.5)
        )
        self.register_buffer(
            "slot_mask", torch.ones(1, self.num_prototypes, num_subprototypes)
        )
        self.register_buffer("slots_pruned", torch.tensor(False))
        self.register_buffer(
            "projection_anchors",
            torch.zeros(self.num_prototypes, feature_dim, num_subprototypes),
        )
        self.register_buffer("has_projection_anchors", torch.tensor(False))

        class_identity = torch.zeros(self.num_prototypes, num_classes)
        for prototype_idx in range(self.num_prototypes):
            class_identity[
                prototype_idx, prototype_idx // prototypes_per_class
            ] = 1
        self.register_buffer("prototype_class_identity", class_identity)
        if prototype_grade_groups is None:
            grade_targets = torch.full((self.num_prototypes,), -1, dtype=torch.long)
        else:
            if len(prototype_grade_groups) != self.num_prototypes:
                raise ValueError(
                    "prototype_grade_groups must contain one entry per prototype "
                    f"({self.num_prototypes}), got {len(prototype_grade_groups)}"
                )
            grade_targets = torch.as_tensor(prototype_grade_groups, dtype=torch.long)
            valid_targets = (grade_targets >= -1) & (grade_targets <= 5)
            if not bool(valid_targets.all()):
                raise ValueError(
                    "Prototype grade targets must be -1 or a Grade Group in "
                    "{0, 1, 2, 3, 4, 5}."
                )
            prototype_classes = class_identity.argmax(dim=1)
            assigned = grade_targets >= 0
            if num_classes == 2:
                expected_classes = (grade_targets > 0).long()
            elif num_classes == 3:
                expected_classes = torch.where(
                    grade_targets == 0,
                    torch.zeros_like(grade_targets),
                    torch.where(
                        grade_targets <= 2,
                        torch.ones_like(grade_targets),
                        torch.full_like(grade_targets, 2),
                    ),
                )
            else:
                expected_classes = grade_targets.clamp_min(0)
            if bool(
                (prototype_classes[assigned] != expected_classes[assigned]).any()
            ):
                raise ValueError(
                    "Prototype grade targets conflict with their classifier classes."
                )
        # Reconstructed from the saved config. Keeping this non-persistent
        # preserves strict loading of pre-grade-aware v2 checkpoints.
        self.register_buffer(
            "prototype_grade_targets", grade_targets, persistent=False
        )


    def slot_weights(self) -> torch.Tensor:
        if bool(self.slots_pruned.item()):
            return self.slot_mask
        return torch.sigmoid(self.slot_logits * self._slot_temperature)

    @property
    def slot_temperature(self) -> float:
        return float(self._slot_temperature.item())

    def subprototype_similarities(self, features: torch.Tensor) -> torch.Tensor:
        """Return cosine similarities with shape ``[B, P, H*W, K]``."""
        tokens = F.normalize(features.flatten(2).transpose(1, 2), dim=-1)
        prototypes = F.normalize(self.prototype_vectors, dim=1)
        return torch.einsum("bnd,pdk->bpnk", tokens, prototypes)



    def regularization_losses(
        self,
        labels: torch.Tensor,
        matched_values: torch.Tensor,
        involvement: torch.Tensor | None = None,
        grade_groups: torch.Tensor | None = None,
        involvement_valid: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """ProtoViT cluster, separation, coherence, and orthogonality losses."""
        labels = labels.long().to(matched_values.device)
        identity = self.prototype_class_identity[:, labels].T.bool()
        slot_weights = self.slot_weights()
        normalized_match = (
            matched_values * slot_weights
        ).sum(-1) / slot_weights.sum(-1).clamp_min(1e-8)

        correct = normalized_match.masked_fill(~identity, -torch.inf)
        wrong = normalized_match.masked_fill(identity, -torch.inf)
        cluster_per_sample = 1.0 - correct.max(dim=1).values
        separation_per_sample = wrong.max(dim=1).values
        if involvement is not None:
            involvement = involvement.float().to(matched_values.device)
            q_valid = (
                involvement_valid.bool().to(matched_values.device)
                if involvement_valid is not None
                else torch.ones_like(involvement, dtype=torch.bool)
            )
            # A low-involvement positive bag contains mostly benign tokens. It
            # remains fully supervised by the q loss, but contributes less to
            # class-specific prototype geometry than a high-involvement core.
            geometry_weight = torch.where(
                labels == 0,
                torch.ones_like(involvement),
                torch.where(
                    q_valid,
                    involvement.clamp(min=0.1, max=1.0),
                    torch.ones_like(involvement),
                ),
            )
            cluster = (cluster_per_sample * geometry_weight).sum() / geometry_weight.sum().clamp_min(1e-8)
            separation = (separation_per_sample * geometry_weight).sum() / geometry_weight.sum().clamp_min(1e-8)
        else:
            cluster = cluster_per_sample.mean()
            separation = separation_per_sample.mean()

        # Grade-aware clustering is deliberately auxiliary: binary cancer
        # geometry remains the primary endpoint, while sufficiently involved
        # cancer cores are also pulled toward prototypes projected from their
        # own observed Grade Group. GG1 is absent from NCT2013 and therefore is
        # never fabricated as a trainable class.
        grade_cluster_terms = []
        if grade_groups is not None and bool((self.prototype_grade_targets >= 0).any()):
            grade_groups = grade_groups.long().to(matched_values.device)
            for sample_idx in range(len(labels)):
                if int(labels[sample_idx]) == 0:
                    continue
                if (
                    involvement is not None
                    and bool(q_valid[sample_idx])
                    and float(involvement[sample_idx])
                    < self.coverage_min_cancer_involvement
                ):
                    continue
                target_mask = self.prototype_grade_targets == grade_groups[sample_idx]
                if bool(target_mask.any()):
                    grade_cluster_terms.append(
                        1.0 - normalized_match[sample_idx, target_mask].max()
                    )
        grade_cluster = (
            torch.stack(grade_cluster_terms).mean()
            if grade_cluster_terms
            else matched_values.new_zeros(())
        )

        prototypes = F.normalize(self.prototype_vectors, dim=1)
        pairwise_cos = torch.einsum("pdk,pdl->pkl", prototypes, prototypes)
        pair_mask = 1.0 - torch.eye(
            self.num_subprototypes,
            device=pairwise_cos.device,
            dtype=pairwise_cos.dtype,
        )
        active_slots = self.slot_weights().squeeze(0)
        active_pairs = (
            active_slots.unsqueeze(-1) * active_slots.unsqueeze(-2) * pair_mask
        )
        coherence = (
            (1.0 - pairwise_cos) * active_pairs
        ).sum() / active_pairs.sum().clamp_min(1.0)

        flat = F.normalize(self.prototype_vectors.flatten(1), dim=1)
        by_class = flat.view(
            self.num_classes, self.prototypes_per_class, -1
        )
        gram = by_class @ by_class.transpose(1, 2)
        eye = torch.eye(
            self.prototypes_per_class, device=gram.device, dtype=gram.dtype
        )
        orthogonality = (gram - eye).square().mean()
        coverage_terms = []
        for prototype_idx in range(self.num_prototypes):
            prototype_class = int(
                self.prototype_class_identity[prototype_idx].argmax().item()
            )
            class_samples = labels == prototype_class
            grade_target = int(self.prototype_grade_targets[prototype_idx])
            if grade_groups is not None and grade_target >= 0:
                class_samples = class_samples & (grade_groups == grade_target)
            if prototype_class > 0 and involvement is not None:
                class_samples = class_samples & (
                    q_valid
                    & (involvement >= self.coverage_min_cancer_involvement)
                )
            if class_samples.any():
                coverage_terms.append(
                    1.0 - normalized_match[class_samples, prototype_idx].max()
                )
        coverage = (
            torch.stack(coverage_terms).mean()
            if coverage_terms
            else matched_values.new_zeros(())
        )

        if bool(self.has_projection_anchors.item()):
            current = F.normalize(self.prototype_vectors, dim=1)
            anchors = F.normalize(self.projection_anchors, dim=1)
            anchor_distance = 1.0 - (current * anchors).sum(dim=1)
            anchor = (
                anchor_distance * self.slot_weights().squeeze(0)
            ).sum() / self.slot_weights().sum().clamp_min(1.0)
        else:
            anchor = matched_values.new_zeros(())

        slot_weights_flat = self.slot_weights()
        return {
            "cluster": cluster,
            "separation": separation,
            "grade_cluster": grade_cluster,
            "coherence": coherence,
            "orthogonality": orthogonality,
            "coverage": coverage,
            "anchor": anchor,
            "slot_sparsity": slot_weights_flat.mean(),
            "slot_binarization": (
                slot_weights_flat * (1.0 - slot_weights_flat)
            ).mean(),
        }

    @torch.no_grad()
    def set_projection_anchors(self) -> None:
        """Remember the current real-token projection for manifold anchoring."""
        self.projection_anchors.copy_(self.prototype_vectors)
        self.has_projection_anchors.fill_(True)

    def set_slot_temperature(self, temperature: float) -> None:
        self._slot_temperature.fill_(float(temperature))

    @torch.no_grad()
    def prune_slots(self, threshold: float = 0.5, min_slots: int = 1) -> torch.Tensor:
        weights = torch.sigmoid(self.slot_logits * self._slot_temperature)
        mask = weights >= threshold
        for prototype_idx in range(self.num_prototypes):
            if int(mask[0, prototype_idx].sum()) < min_slots:
                keep = weights[0, prototype_idx].topk(min_slots).indices
                mask[0, prototype_idx, keep] = True
        self.slot_mask.copy_(mask.to(self.slot_mask.dtype))
        self.slots_pruned.fill_(True)
        self.slot_logits.requires_grad_(False)
        return self.slot_mask.clone()


class MaskedPrototypeOccurrence(nn.Module):
    """ProtoASNet-style dynamic regions constrained to the sampled core.

    A separate normalized occurrence map is learned for every sub-prototype.
    Each sub-prototype is compared with the convex pooled feature under its
    occurrence map.  Consequently projection can copy the *exact* pooled
    training feature used by the classifier, rather than an unrelated token.
    """

    def __init__(
        self,
        feature_dim: int,
        num_prototypes: int,
        num_subprototypes: int,
        hidden_dim: int = 128,
        num_uncertainty_prototypes: int = 0,
        occurrence_temperature: float = 1.0,
        learnable_temperature: bool = False,
        locality_sigma_tokens: float = 0.0,
        locality_strength: float = 1.0,
    ) -> None:
        super().__init__()
        self.num_prototypes = num_prototypes
        self.num_subprototypes = num_subprototypes
        self.num_uncertainty_prototypes = num_uncertainty_prototypes
        if locality_sigma_tokens < 0:
            raise ValueError("locality_sigma_tokens must be non-negative.")
        if locality_strength < 0:
            raise ValueError("locality_strength must be non-negative.")
        self.locality_sigma_tokens = float(locality_sigma_tokens)
        self.locality_strength = float(locality_strength)
        initial_scale = 1.0 / max(float(occurrence_temperature), 1e-3)
        if learnable_temperature:
            self.logit_scale = nn.Parameter(torch.tensor(initial_scale).log())
        else:
            self.register_buffer("logit_scale", torch.tensor(initial_scale).log())
        total = num_prototypes + num_uncertainty_prototypes
        self.occurrence_net = nn.Sequential(
            nn.Conv2d(feature_dim, hidden_dim, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(hidden_dim, total * num_subprototypes, kernel_size=1),
        )
        if num_uncertainty_prototypes:
            self.uncertainty_vectors = nn.Parameter(
                torch.empty(
                    num_uncertainty_prototypes, feature_dim, num_subprototypes
                )
            )
            nn.init.trunc_normal_(self.uncertainty_vectors, std=0.02)
        else:
            self.register_parameter("uncertainty_vectors", None)

    @staticmethod
    def _masked_softmax(
        logits: torch.Tensor, valid_token_mask: torch.Tensor
    ) -> torch.Tensor:
        if valid_token_mask.ndim == 2:
            mask = valid_token_mask[:, None, None, :].bool()
            nonempty = valid_token_mask.any(dim=-1)
        elif valid_token_mask.ndim == 3:
            mask = valid_token_mask[:, :, None, :].bool()
            nonempty = valid_token_mask.any(dim=-1)
        else:
            raise ValueError("valid_token_mask must have shape [B,N] or [B,P,N].")
        if bool((~nonempty).any()):
            raise ValueError("Occurrence matching received an empty valid region.")
        logits = logits.masked_fill(~mask, torch.finfo(logits.dtype).min)
        weights = torch.softmax(logits.float(), dim=-1).to(logits.dtype)
        return weights * mask.to(weights.dtype)

    @staticmethod
    def _match(
        features: torch.Tensor,
        vectors: torch.Tensor,
        occurrence_weights: torch.Tensor,
        slot_weights: torch.Tensor,
        valid_token_mask: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        tokens = features.flatten(2).transpose(1, 2)
        pooled = torch.einsum("bpkn,bnd->bpkd", occurrence_weights, tokens)
        normalized_pooled = F.normalize(pooled, dim=-1)
        normalized_vectors = F.normalize(vectors.permute(0, 2, 1), dim=-1)
        matched_values = (normalized_pooled * normalized_vectors[None]).sum(-1)
        slots = slot_weights.squeeze(0)
        active_count = slots.sum(-1).clamp_min(1e-8)
        activations = (matched_values * slots[None]).sum(-1) / active_count[None]

        # Spatial evidence retains both where the dynamic region is and how
        # strongly the local token resembles its sub-prototype. Scaling by the
        # valid token count keeps evidence magnitudes comparable across masks.
        local_similarity = torch.einsum(
            "bnd,pdk->bpkn", F.normalize(tokens, dim=-1), F.normalize(vectors, dim=1)
        )
        valid_count = valid_token_mask.sum(-1).clamp_min(1).to(features.dtype)
        if valid_count.ndim == 1:
            valid_count = valid_count[:, None]
        token_maps = (
            occurrence_weights
            * local_similarity
            * slots[None, :, :, None]
        ).sum(2) / active_count[None, :, None]
        token_maps = token_maps * valid_count[:, :, None]
        return {
            "prototype_activations": activations,
            "matched_values": matched_values,
            "matched_indices": occurrence_weights.argmax(-1),
            "prototype_token_maps": token_maps,
            "occurrence_maps": occurrence_weights,
            "occurrence_pooled_features": pooled,
            "subprototype_similarities": local_similarity.permute(0, 1, 3, 2),
        }

    def forward(
        self,
        features: torch.Tensor,
        prototype_vectors: torch.Tensor,
        slot_weights: torch.Tensor,
        valid_token_mask: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        batch, _, height, width = features.shape
        total = self.num_prototypes + self.num_uncertainty_prototypes
        logits = self.occurrence_net(features).reshape(
            batch, total, self.num_subprototypes, height * width
        )
        scale = self.logit_scale.exp().clamp(0.25, 20.0)
        scaled_logits = logits * scale
        if self.locality_sigma_tokens > 0 and self.locality_strength > 0:
            # All slots of one prototype should describe one tissue neighborhood.
            # Estimate a shared differentiable center from the unconstrained maps,
            # then re-normalize every slot through the same Gaussian locality gate.
            # Unlike a TV penalty, this construction cannot be minimized by a
            # uniform map and remains compatible with irregular prostate masks.
            initial = self._masked_softmax(scaled_logits, valid_token_mask)
            yy, xx = torch.meshgrid(
                torch.arange(height, device=features.device, dtype=torch.float32),
                torch.arange(width, device=features.device, dtype=torch.float32),
                indexing="ij",
            )
            coordinates = torch.stack((xx.flatten(), yy.flatten()), dim=1)
            shared = initial.float().mean(dim=2)
            centroids = torch.einsum("bpn,nd->bpd", shared, coordinates)
            squared_distance = (
                coordinates[None, None] - centroids[:, :, None]
            ).square().sum(-1)
            locality_bias = -(
                self.locality_strength
                * squared_distance
                / (2.0 * self.locality_sigma_tokens**2)
            )
            scaled_logits = scaled_logits + locality_bias[:, :, None].to(
                scaled_logits.dtype
            )
        weights = self._masked_softmax(scaled_logits, valid_token_mask)
        prototype_mask = (
            valid_token_mask[:, : self.num_prototypes]
            if valid_token_mask.ndim == 3
            else valid_token_mask
        )
        outputs = self._match(
            features,
            prototype_vectors,
            weights[:, : self.num_prototypes],
            slot_weights,
            prototype_mask,
        )
        if self.num_uncertainty_prototypes:
            uncertainty_slots = torch.ones(
                1,
                self.num_uncertainty_prototypes,
                self.num_subprototypes,
                device=features.device,
                dtype=features.dtype,
            )
            uncertainty = self._match(
                features,
                self.uncertainty_vectors,
                weights[:, self.num_prototypes :],
                uncertainty_slots,
                (
                    valid_token_mask[:, self.num_prototypes :]
                    if valid_token_mask.ndim == 3
                    else valid_token_mask
                ),
            )
            outputs.update(
                {
                    f"uncertainty_{key.removeprefix('prototype_')}": value
                    for key, value in uncertainty.items()
                    if key != "subprototype_similarities"
                }
            )
        return outputs


class ClinicalAdditiveHead(nn.Module):
    """Interpretable generalized-additive clinical risk branch.

    Age, PSA, and PSA density use monotone piecewise-linear functions. Core
    position uses unconstrained additive linear effects because risk need not be
    monotone from base to apex or medial to lateral.
    """

    prompt_names = (
        "age", "psa", "approx_psa_density",
        "base_apex_encoding", "mid_lateral_encoding",
    )

    def __init__(self, knots: tuple[float, ...] = (0.2, 0.4, 0.6, 0.8)) -> None:
        super().__init__()
        self.register_buffer("knots", torch.tensor(knots).float())
        # Begin as an almost exact image-only model. The clinical branch must
        # earn influence from training instead of adding a cancer offset.
        self.monotone_raw_weights = nn.Parameter(
            torch.full((3, 1 + len(knots)), -5.0)
        )
        self.position_weights = nn.Parameter(torch.zeros(2))
        self.bias = nn.Parameter(torch.zeros(()))



class ClinicalPromptConditioner(nn.Module):
    """Inject clinical prompts into spatial prototype evidence and class logits.

    ProstNFound injects prompt embeddings into its decoder.  ProtoProstate has
    no conventional mask decoder, so this module conditions the latent token
    grid immediately before prototype matching.  A prompt-derived FiLM term
    changes channel semantics, while a prompt-derived query supplies a bounded
    spatial gate.  A separate three-state residual supports information that is
    genuinely core-level.  All output layers start at zero, making the initial
    model exactly image-only and forcing prompts to earn their influence.
    """

    prompt_names = ClinicalAdditiveHead.prompt_names

    def __init__(
        self,
        feature_dim: int = 256,
        num_classes: int = 3,
        hidden_dim: int = 128,
        feature_scale: float = 0.25,
        spatial_scale: float = 1.0,
    ) -> None:
        super().__init__()
        self.feature_dim = int(feature_dim)
        self.feature_scale = float(feature_scale)
        self.spatial_scale = float(spatial_scale)
        self.register_buffer(
            "prompt_center", torch.tensor((0.5, 0.5, 0.5, 0.0, 0.0))
        )
        self.register_buffer(
            "prompt_scale", torch.tensor((0.5, 0.5, 0.5, 1.0, 1.0))
        )
        branch_dim = hidden_dim // 2
        self.continuous_encoder = nn.Sequential(
            nn.Linear(3, branch_dim), nn.GELU(), nn.LayerNorm(branch_dim)
        )
        self.position_encoder = nn.Sequential(
            nn.Linear(2, branch_dim), nn.GELU(), nn.LayerNorm(branch_dim)
        )
        self.fusion = nn.Sequential(
            nn.Linear(2 * branch_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
        )
        self.film = nn.Linear(hidden_dim, 2 * feature_dim)
        self.spatial_query = nn.Linear(hidden_dim, feature_dim)
        self.classifier = nn.Linear(hidden_dim, num_classes)
        nn.init.zeros_(self.film.weight)
        nn.init.zeros_(self.film.bias)
        nn.init.zeros_(self.spatial_query.weight)
        nn.init.zeros_(self.spatial_query.bias)
        nn.init.zeros_(self.classifier.weight)
        nn.init.zeros_(self.classifier.bias)

    def forward(
        self, features: torch.Tensor, prompts: torch.Tensor
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        if prompts.ndim != 2 or prompts.shape[1] != 5:
            raise ValueError(
                "Prompt conditioning expects [age, PSA, PSAD, base/apex, "
                f"medial/lateral], got {tuple(prompts.shape)}."
            )
        standardized = (prompts - self.prompt_center[None]) / self.prompt_scale[None]
        continuous = self.continuous_encoder(standardized[:, :3])
        position = self.position_encoder(standardized[:, 3:])
        embedding = self.fusion(torch.cat((continuous, position), dim=1))

        gamma, beta = self.film(embedding).chunk(2, dim=1)
        gamma = self.feature_scale * torch.tanh(gamma)
        beta = self.feature_scale * torch.tanh(beta)
        modulated = features * (1.0 + gamma[:, :, None, None]) + beta[:, :, None, None]

        query = self.spatial_query(embedding)
        normalized_features = F.normalize(modulated, dim=1)
        spatial_logits = torch.einsum(
            "bd,bdhw->bhw", query, normalized_features
        ) / self.feature_dim**0.5
        # 2*sigmoid(0)=1: zero initialization preserves the image-only path.
        spatial_gate = 2.0 * torch.sigmoid(self.spatial_scale * spatial_logits)
        conditioned = modulated * spatial_gate[:, None]
        return conditioned, {
            "prompt_embedding": embedding,
            "prompt_class_logits": self.classifier(embedding),
            "prompt_spatial_gate": spatial_gate[:, None],
            "prompt_film_gamma": gamma,
            "prompt_film_beta": beta,
        }


class NeedleAxisPrototypeHead(nn.Module):
    """Positive-reasoning multi-slot prototypes over ordered needle bins."""

    def __init__(
        self,
        embedding_dim: int,
        num_classes: int,
        prototypes_per_class: int,
        slots_per_prototype: int,
        attention_temperature: float,
    ) -> None:
        super().__init__()
        if num_classes < 2 or prototypes_per_class < 1 or slots_per_prototype < 1:
            raise ValueError("Needle-axis prototypes require positive bank sizes.")
        if attention_temperature <= 0:
            raise ValueError("Prototype attention temperature must be positive.")
        self.embedding_dim = int(embedding_dim)
        self.num_classes = int(num_classes)
        self.prototypes_per_class = int(prototypes_per_class)
        self.slots_per_prototype = int(slots_per_prototype)
        self.attention_temperature = float(attention_temperature)
        self.prototype_vectors = nn.Parameter(
            torch.empty(
                self.num_classes,
                self.prototypes_per_class,
                self.slots_per_prototype,
                self.embedding_dim,
            )
        )
        nn.init.normal_(self.prototype_vectors, std=0.02)
        # softplus(0.5413) is approximately one. Evidence weights remain
        # non-negative, so every displayed prototype is supportive.
        self.raw_class_weights = nn.Parameter(
            torch.full((self.num_classes, self.prototypes_per_class), 0.5413)
        )
        self.class_bias = nn.Parameter(torch.zeros(self.num_classes))
        self.register_buffer(
            "prototype_class_identity",
            torch.eye(self.num_classes).repeat_interleave(
                self.prototypes_per_class, dim=0
            ),
        )

    @property
    def class_weights(self) -> torch.Tensor:
        return F.softplus(self.raw_class_weights)



class PathologyPrototypeTransportHead(nn.Module):
    """Match ordered ultrasound evidence to fixed, real pathology prototypes.

    The modalities are deliberately not forced into a one-to-one pairing. A
    small ultrasound-only projector produces a shared component while the
    original Axis-Proto representation remains untouched as the modality-
    private decision path. Class-conditional Sinkhorn transport then maps each
    ultrasound decision prototype to several compatible PANDA prototypes.
    """

    def __init__(
        self,
        input_dim: int,
        bank_path: str,
        hidden_dim: int = 128,
        attention_temperature: float = 0.20,
        transport_temperature: float = 0.10,
        sinkhorn_iterations: int = 6,
        fusion_max_scale: float = 0.20,
        fusion_initial_scale: float = 0.05,
        fusion_fixed_scale: float | None = None,
    ) -> None:
        super().__init__()
        if attention_temperature <= 0 or transport_temperature <= 0:
            raise ValueError("Pathology transport temperatures must be positive.")
        if sinkhorn_iterations < 1:
            raise ValueError("Pathology transport needs at least one Sinkhorn step.")
        resolved = Path(bank_path).expanduser().resolve()
        if not resolved.exists():
            raise FileNotFoundError(f"Missing pathology prototype bank: {resolved}")
        archive = np.load(str(resolved), allow_pickle=False)
        required = {"anchors", "state_labels", "grade_labels"}
        missing = required.difference(archive.files)
        if missing:
            raise ValueError(
                f"Pathology prototype bank {resolved} is missing {sorted(missing)}."
            )
        anchors = F.normalize(
            torch.as_tensor(archive["anchors"], dtype=torch.float32), dim=1
        )
        states = torch.as_tensor(archive["state_labels"], dtype=torch.long)
        grades = torch.as_tensor(archive["grade_labels"], dtype=torch.long)
        if anchors.ndim != 2 or len(anchors) != len(states) or len(states) != len(grades):
            raise ValueError("Pathology anchors and labels must have aligned rows.")
        if set(states.tolist()) != {0, 1, 2}:
            raise ValueError("Pathology transport requires states 0, 1, and 2.")
        if set(grades.tolist()) != set(range(6)):
            raise ValueError("Pathology transport requires Grade Groups 0 through 5.")
        expected_states = torch.where(grades == 0, 0, torch.where(grades <= 2, 1, 2))
        if not torch.equal(states, expected_states):
            raise ValueError("Pathology state labels disagree with Grade Group labels.")
        self.bank_path = str(resolved)
        self.input_dim = int(input_dim)
        self.anchor_dim = int(anchors.shape[1])
        self.attention_temperature = float(attention_temperature)
        self.transport_temperature = float(transport_temperature)
        self.sinkhorn_iterations = int(sinkhorn_iterations)
        self.fusion_max_scale = float(fusion_max_scale)
        self.fusion_fixed_scale = (
            None if fusion_fixed_scale is None else float(fusion_fixed_scale)
        )
        if self.fusion_fixed_scale is not None and self.fusion_fixed_scale < 0:
            raise ValueError("Fixed pathology transport fusion must be non-negative.")
        if self.fusion_fixed_scale is None:
            if not 0 < fusion_initial_scale < self.fusion_max_scale:
                raise ValueError(
                    "Pathology transport fusion needs 0 < initial < maximum."
                )
            fraction = float(fusion_initial_scale) / self.fusion_max_scale
            fusion_logit = torch.logit(torch.tensor(fraction, dtype=torch.float32))
        else:
            fusion_logit = torch.zeros((), dtype=torch.float32)
        self.fusion_logit = nn.Parameter(fusion_logit)
        self.common_projector = nn.Sequential(
            nn.LayerNorm(self.input_dim),
            nn.Linear(self.input_dim, int(hidden_dim)),
            nn.GELU(),
            nn.Linear(int(hidden_dim), self.anchor_dim),
        )
        self.raw_anchor_weights = nn.Parameter(torch.full((len(anchors),), 0.5413))
        self.grade_bias = nn.Parameter(torch.zeros(6))
        self.register_buffer("anchors", anchors)
        self.register_buffer("anchor_states", states)
        self.register_buffer("anchor_grades", grades)

    @property
    def fusion_scale(self) -> torch.Tensor:
        if self.fusion_fixed_scale is not None:
            return self.fusion_logit.new_tensor(self.fusion_fixed_scale)
        return self.fusion_max_scale * torch.sigmoid(self.fusion_logit)

    def _project(self, values: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.common_projector(values.float()), dim=-1)




class HyperbolicNeedleEncoder(nn.Module):
    """Encode the complete sampled needle and place it in a Lorentz cone.

    The needle trace is first reduced to ordered bins along its principal
    axis.  This retains extent and ordering information that is destroyed by
    global average pooling.  Angular position represents Grade Group while
    radial position represents cancer involvement.  Unlike HypOProto's
    two-sided distance from a clinical cut-off, prostate involvement is
    one-sided: benign cores live near the origin and increasingly involved
    cancers move outwards.
    """

    observed_grades = (0, 2, 3, 4, 5)

    def __init__(
        self,
        feature_dim: int = 256,
        embedding_dim: int = 64,
        axis_bins: int = 16,
        transformer_layers: int = 1,
        transformer_heads: int = 4,
        prototypes_per_grade: int = 2,
        curvature: float = 1.0,
        radius_min: float = 0.05,
        radius_max: float = 2.5,
        distance_temperature: float = 0.5,
        fusion_min_scale: float = 0.0,
        fusion_max_scale: float = 0.5,
        fusion_initial_scale: float = 0.1,
        fusion_fixed_scale: float | None = None,
        projection_q_tolerance: float = 0.2,
        unique_projection: bool = False,
        projection_q_weight: float = 0.5,
        projection_candidate_limit: int = 256,
        target_mode: str = "grade5",
        shared_class_directions: bool = False,
        structured_needle_selection: bool = False,
        needle_prototype_classifier: bool = False,
        needle_prototype_slots: int = 4,
        needle_prototype_attention_temperature: float = 0.20,
        needle_prototype_style_mix_probability: float = 0.0,
        needle_prototype_style_mix_alpha: float = 0.30,
        pathology_transport_bank_path: str | None = None,
        pathology_transport_hidden_dim: int = 128,
        pathology_transport_attention_temperature: float = 0.20,
        pathology_transport_temperature: float = 0.10,
        pathology_transport_sinkhorn_iterations: int = 6,
        pathology_transport_fusion_max_scale: float = 0.20,
        pathology_transport_fusion_initial_scale: float = 0.05,
        pathology_transport_fusion_fixed_scale: float | None = None,
        geometry: str = "hyperbolic",
    ) -> None:
        super().__init__()
        if axis_bins < 2:
            raise ValueError("Hyperbolic needle encoder needs at least two axis bins.")
        if geometry not in {"hyperbolic", "euclidean"}:
            raise ValueError("geometry must be 'hyperbolic' or 'euclidean'.")
        if target_mode not in {
            "grade5",
            "grade6",
            "three_way",
            "grade_bands_01_23_45",
        }:
            raise ValueError(
                "target_mode must be 'grade5', 'grade6', 'three_way', or "
                "'grade_bands_01_23_45'."
            )
        if not 0 <= fusion_min_scale < fusion_max_scale:
            raise ValueError("fusion_min_scale must be in [0, fusion_max_scale).")
        if not fusion_min_scale < fusion_initial_scale < fusion_max_scale:
            raise ValueError(
                "fusion_initial_scale must be in (fusion_min_scale, fusion_max_scale)."
            )
        if fusion_fixed_scale is not None and fusion_fixed_scale < 0:
            raise ValueError("fusion_fixed_scale must be non-negative when supplied.")
        if projection_candidate_limit < prototypes_per_grade:
            raise ValueError(
                "projection_candidate_limit must be at least prototypes_per_grade."
            )
        self.feature_dim = int(feature_dim)
        self.embedding_dim = int(embedding_dim)
        self.axis_bins = int(axis_bins)
        self.prototypes_per_grade = int(prototypes_per_grade)
        self.curvature = float(curvature)
        self.radius_min = float(radius_min)
        self.radius_max = float(radius_max)
        self.distance_temperature = float(distance_temperature)
        self.fusion_min_scale = float(fusion_min_scale)
        self.fusion_max_scale = float(fusion_max_scale)
        self.fusion_fixed_scale = (
            None if fusion_fixed_scale is None else float(fusion_fixed_scale)
        )
        self.geometry = geometry
        self.projection_q_tolerance = float(projection_q_tolerance)
        self.unique_projection = bool(unique_projection)
        self.projection_q_weight = float(projection_q_weight)
        self.projection_candidate_limit = int(projection_candidate_limit)
        self.target_mode = target_mode
        self.target_values = (
            self.observed_grades
            if target_mode == "grade5"
            else (
                (0, 1, 2, 3, 4, 5)
                if target_mode == "grade6"
                else (0, 1, 2)
            )
        )
        self.target_names = (
            ("GG0-1", "GG2-3", "GG4-5")
            if target_mode == "grade_bands_01_23_45"
            else tuple(str(value) for value in self.target_values)
        )
        self.shared_class_directions = bool(shared_class_directions)
        self.structured_needle_selection = bool(structured_needle_selection)
        self.needle_prototype_style_mix_probability = float(
            needle_prototype_style_mix_probability
        )
        self.needle_prototype_style_mix_alpha = float(
            needle_prototype_style_mix_alpha
        )
        if not 0.0 <= self.needle_prototype_style_mix_probability <= 1.0:
            raise ValueError(
                "Needle prototype style-mix probability must be in [0, 1]."
            )
        if self.needle_prototype_style_mix_alpha <= 0:
            raise ValueError("Needle prototype style-mix alpha must be positive.")
        if needle_prototype_classifier and target_mode != "three_way":
            raise ValueError(
                "The needle-axis prototype classifier requires target_mode='three_way'."
            )

        self.input_projection = nn.Sequential(
            nn.Linear(feature_dim, embedding_dim),
            nn.GELU(),
            nn.LayerNorm(embedding_dim),
        )
        self.position_embedding = nn.Parameter(
            torch.empty(1, axis_bins, embedding_dim)
        )
        nn.init.trunc_normal_(self.position_embedding, std=0.02)
        layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim,
            nhead=transformer_heads,
            dim_feedforward=4 * embedding_dim,
            dropout=0.1,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.axis_encoder = nn.TransformerEncoder(
            layer, num_layers=transformer_layers, enable_nested_tensor=False
        )
        self.needle_prototype_head = (
            NeedleAxisPrototypeHead(
                embedding_dim=embedding_dim,
                num_classes=len(self.target_values),
                prototypes_per_class=prototypes_per_grade,
                slots_per_prototype=needle_prototype_slots,
                attention_temperature=needle_prototype_attention_temperature,
            )
            if needle_prototype_classifier
            else None
        )
        self.pathology_transport_head = (
            PathologyPrototypeTransportHead(
                input_dim=embedding_dim,
                bank_path=pathology_transport_bank_path,
                hidden_dim=pathology_transport_hidden_dim,
                attention_temperature=pathology_transport_attention_temperature,
                transport_temperature=pathology_transport_temperature,
                sinkhorn_iterations=pathology_transport_sinkhorn_iterations,
                fusion_max_scale=pathology_transport_fusion_max_scale,
                fusion_initial_scale=pathology_transport_fusion_initial_scale,
                fusion_fixed_scale=pathology_transport_fusion_fixed_scale,
            )
            if pathology_transport_bank_path
            else None
        )
        if self.pathology_transport_head is not None and self.needle_prototype_head is None:
            raise ValueError(
                "Pathology transport requires the needle-axis prototype classifier."
            )
        self.direction_head = nn.Linear(embedding_dim, embedding_dim)
        self.involvement_head = nn.Linear(embedding_dim, 1)
        self.needle_selection_head = (
            nn.Linear(embedding_dim, 1)
            if self.structured_needle_selection
            else None
        )
        if self.needle_selection_head is not None:
            # sigmoid(0)=0.5 gives useful gradients before noisy q supervision
            # decides which longitudinal bins are active.
            nn.init.zeros_(self.needle_selection_head.weight)
            nn.init.zeros_(self.needle_selection_head.bias)

        num_targets = len(self.target_values)
        direction_shape = (
            (num_targets, embedding_dim)
            if self.shared_class_directions
            else (num_targets, prototypes_per_grade, embedding_dim)
        )
        self.prototype_directions = nn.Parameter(torch.empty(*direction_shape))
        nn.init.orthogonal_(self.prototype_directions.flatten(0, -2))
        # Two initial severity anchors per cancer grade provide nearby and far
        # exemplars. Benign prototypes remain fixed at the cone origin.
        anchors = torch.tensor((0.25, 0.75)).float()
        if prototypes_per_grade != 2:
            anchors = torch.linspace(0.2, 0.8, prototypes_per_grade)
        anchor_logits = torch.logit(anchors.clamp(1e-4, 1 - 1e-4))
        self.cancer_prototype_radius_logits = nn.Parameter(
            anchor_logits.repeat(num_targets - 1, 1)
        )
        # In legacy modes target zero is genuinely benign and remains at the
        # cone root.  The first ordinal band also contains GG1, however, so it
        # needs its own learnable severity radii rather than being falsely
        # collapsed onto zero burden.  Initial values retain one near-root
        # exemplar while allowing the other anchors to cover low burden.
        if target_mode == "grade_bands_01_23_45":
            low_band_anchors = torch.linspace(
                0.0, 0.45, prototypes_per_grade
            ).clamp(1e-4, 1 - 1e-4)
            self.low_band_prototype_radius_logits = nn.Parameter(
                torch.logit(low_band_anchors)[None]
            )
        else:
            self.register_parameter("low_band_prototype_radius_logits", None)
        # The requested GG0-1/GG2-3/GG4-5 geometry is deliberately not used
        # directly as benign/non-csPCa/csPCa logits: GG3 is in the middle band
        # but is clinically significant.  This small readout consumes only
        # prototype-distance evidence plus predicted burden, so the global
        # branch remains independently auditable and prototype based.
        self.band_endpoint_readout = (
            nn.Linear(num_targets + 1, 3)
            if target_mode == "grade_bands_01_23_45"
            else None
        )
        if self.band_endpoint_readout is not None:
            nn.init.zeros_(self.band_endpoint_readout.weight)
            nn.init.zeros_(self.band_endpoint_readout.bias)
        initial_fraction = (fusion_initial_scale - self.fusion_min_scale) / (
            self.fusion_max_scale - self.fusion_min_scale
        )
        self.fusion_logit = nn.Parameter(
            torch.tensor(float(torch.logit(torch.tensor(initial_fraction))))
        )

    @property
    def fusion_scale(self) -> torch.Tensor:
        if self.fusion_fixed_scale is not None:
            return self.fusion_logit.new_tensor(self.fusion_fixed_scale)
        return self.fusion_min_scale + (
            self.fusion_max_scale - self.fusion_min_scale
        ) * torch.sigmoid(self.fusion_logit)

    @staticmethod
    def _principal_axis_bins(mask: torch.Tensor, bins: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Return token bin indices and non-empty-bin mask for one core."""
        height, width = mask.shape
        coordinates = torch.stack(
            torch.meshgrid(
                torch.arange(height, device=mask.device, dtype=torch.float32),
                torch.arange(width, device=mask.device, dtype=torch.float32),
                indexing="ij",
            ),
            dim=-1,
        ).reshape(-1, 2)
        selected = coordinates[mask.flatten()]
        if len(selected) < 2:
            axis = coordinates.new_tensor((0.0, 1.0))
            center = selected.mean(0) if len(selected) else coordinates.mean(0)
        else:
            center = selected.mean(0)
            covariance = (selected - center).T @ (selected - center)
            # This geometry depends only on a binary mask. Force float32 so AMP
            # cannot route the tiny eigendecomposition through unsupported fp16.
            _, eigenvectors = torch.linalg.eigh(covariance.float())
            axis = eigenvectors[:, -1]
            # Eigenvectors have arbitrary sign. Stable image-coordinate
            # orientation makes bin positions comparable across real cores.
            dominant = 1 if axis[1].abs() >= axis[0].abs() else 0
            if axis[dominant] < 0:
                axis = -axis
        projections = (coordinates - center) @ axis
        selected_projection = projections[mask.flatten()]
        low, high = selected_projection.min(), selected_projection.max()
        normalized = (projections - low) / (high - low).clamp_min(1e-6)
        indices = (normalized * bins).floor().long().clamp(0, bins - 1)
        valid_indices = indices[mask.flatten()]
        occupied = torch.zeros(bins, dtype=torch.bool, device=mask.device)
        occupied.scatter_(0, valid_indices, True)
        return indices, occupied

    def pool_needle_axis(
        self, features: torch.Tensor, needle_tokens: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch, channels, height, width = features.shape
        flat = features.flatten(2).transpose(1, 2)
        pooled_samples = []
        occupied_samples = []
        # Mask-derived bin assignments intentionally carry no gradient.
        with torch.autocast(device_type=features.device.type, enabled=False):
            for sample in range(batch):
                mask = needle_tokens[sample].reshape(height, width).bool()
                indices, occupied = self._principal_axis_bins(mask, self.axis_bins)
                values = flat[sample].float()
                pooled = values.new_zeros(self.axis_bins, channels)
                counts = values.new_zeros(self.axis_bins, 1)
                selected_indices = indices[mask.flatten()]
                pooled.index_add_(0, selected_indices, values[mask.flatten()])
                counts.index_add_(
                    0,
                    selected_indices,
                    torch.ones(len(selected_indices), 1, device=values.device),
                )
                pooled_samples.append(pooled / counts.clamp_min(1.0))
                occupied_samples.append(occupied)
        return (
            torch.stack(pooled_samples).to(features.dtype),
            torch.stack(occupied_samples),
        )

    def _radius(self, involvement: torch.Tensor) -> torch.Tensor:
        return self.radius_min + (self.radius_max - self.radius_min) * involvement

    def target_indices_from_grades(self, grades: torch.Tensor) -> torch.Tensor:
        """Map observed ISUP grades to this encoder's ordinal targets."""
        grades = grades.long()
        if self.target_mode == "grade_bands_01_23_45":
            invalid = (grades < 0) | (grades > 5)
            if bool(invalid.any()):
                unknown = grades[invalid].unique().tolist()
                raise ValueError(f"Unsupported ISUP Grade Groups: {unknown}")
            return (grades >= 2).long() + (grades >= 4).long()
        values = torch.as_tensor(
            self.target_values, device=grades.device, dtype=grades.dtype
        )
        matches = grades[:, None] == values[None]
        if not bool(matches.any(dim=1).all()):
            unknown = grades[~matches.any(dim=1)].unique().tolist()
            raise ValueError(f"Unsupported hyperbolic target grades: {unknown}")
        return matches.float().argmax(dim=1).long()

    def target_logits_to_three_way(
        self, target_logits: torch.Tensor, predicted_involvement: torch.Tensor
    ) -> torch.Tensor:
        """Convert geometry targets to coherent clinical-state evidence."""
        if self.target_mode == "three_way":
            return target_logits
        if self.target_mode == "grade6":
            return torch.stack(
                (
                    target_logits[:, 0],
                    torch.logsumexp(target_logits[:, 1:3], dim=1),
                    torch.logsumexp(target_logits[:, 3:], dim=1),
                ),
                dim=1,
            )
        if self.target_mode == "grade5":
            return torch.stack(
                (
                    target_logits[:, 0],
                    target_logits[:, 1],
                    torch.logsumexp(target_logits[:, 2:], dim=1),
                ),
                dim=1,
            )
        if self.band_endpoint_readout is None:
            raise RuntimeError("Ordinal-band endpoint readout is missing.")
        return self.band_endpoint_readout(
            torch.cat(
                (target_logits.float(), predicted_involvement.float()[:, None]),
                dim=1,
            )
        )

    def _mix_style(
        self, pooled: torch.Tensor, occupied: torch.Tensor
    ) -> torch.Tensor | None:
        """Synthesize source-domain feature statistics during training only."""
        probability = self.needle_prototype_style_mix_probability
        if (
            not self.training
            or probability <= 0
            or len(pooled) < 2
            or float(torch.rand((), device=pooled.device)) >= probability
        ):
            return None
        values = pooled.float()
        mask = occupied[..., None].float()
        count = mask.sum(1, keepdim=True).clamp_min(1.0)
        mean = (values * mask).sum(1, keepdim=True) / count
        variance = ((values - mean).square() * mask).sum(1, keepdim=True) / count
        std = variance.clamp_min(1e-6).sqrt()
        permutation = torch.randperm(len(values), device=values.device)
        identity = torch.arange(len(values), device=values.device)
        if bool(torch.equal(permutation, identity)):
            permutation = permutation.roll(1)
        beta = torch.distributions.Beta(
            self.needle_prototype_style_mix_alpha,
            self.needle_prototype_style_mix_alpha,
        )
        mixing = beta.sample((len(values),)).to(values.device)[:, None, None]
        mixed_mean = mixing * mean + (1.0 - mixing) * mean[permutation]
        mixed_std = mixing * std + (1.0 - mixing) * std[permutation]
        mixed = (values - mean) / std * mixed_std + mixed_mean
        return (mixed * mask).to(pooled.dtype)

    def _lorentz_lift(self, direction: torch.Tensor, radius: torch.Tensor) -> torch.Tensor:
        direction = F.normalize(direction.float(), dim=-1)
        sqrt_c = self.curvature**0.5
        space = torch.sinh(sqrt_c * radius[..., None]) * direction / sqrt_c
        time = torch.sqrt(1.0 / self.curvature + space.square().sum(-1, keepdim=True))
        return torch.cat((time, space), dim=-1)

    def _distance(self, first: torch.Tensor, second: torch.Tensor) -> torch.Tensor:
        if self.geometry == "euclidean":
            return (first[..., None, None, :] - second[None]).square().sum(-1).sqrt()
        time_product = first[..., None, None, 0] * second[None, ..., 0]
        space_product = (
            first[..., None, None, 1:] * second[None, ..., 1:]
        ).sum(-1)
        argument = self.curvature * (time_product - space_product)
        return torch.acosh(argument.clamp_min(1.0 + 1e-6)) / self.curvature**0.5

    def prototype_points(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        directions = F.normalize(self.prototype_directions.float(), dim=-1)
        if self.shared_class_directions:
            directions = directions[:, None, :].expand(
                -1, self.prototypes_per_grade, -1
            )
        cancer_q = torch.sigmoid(self.cancer_prototype_radius_logits.float())
        if self.low_band_prototype_radius_logits is None:
            first_target_q = cancer_q.new_zeros(1, self.prototypes_per_grade)
        else:
            first_target_q = torch.sigmoid(
                self.low_band_prototype_radius_logits.float()
            )
        involvement = torch.cat((first_target_q, cancer_q), dim=0)
        radius = self._radius(involvement)
        if self.geometry == "euclidean":
            points = directions * radius[..., None]
        else:
            points = self._lorentz_lift(directions, radius)
        return points, directions, involvement

    def forward(
        self, features: torch.Tensor, needle_tokens: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        pooled, occupied = self.pool_needle_axis(features, needle_tokens)
        local_sequence = self.input_projection(pooled) + self.position_embedding
        sequence = self.axis_encoder(
            local_sequence, src_key_padding_mask=~occupied
        )
        global_embedding = (
            sequence * occupied[..., None]
        ).sum(1) / occupied.sum(1, keepdim=True).clamp_min(1)
        if self.needle_selection_head is not None:
            selection_logits = self.needle_selection_head(sequence).squeeze(-1)
            selection_probabilities = torch.sigmoid(selection_logits) * occupied
            selected_embedding = (
                sequence * selection_probabilities[..., None]
            ).sum(1) / selection_probabilities.sum(1, keepdim=True).clamp_min(1e-4)
            # Global context stabilizes benign cores whose selected fraction is
            # intentionally near zero; focal selected evidence remains dominant.
            embedding = 0.25 * global_embedding + 0.75 * selected_embedding
            predicted_involvement = selection_probabilities.sum(1) / occupied.sum(
                1
            ).clamp_min(1)
        else:
            embedding = global_embedding
            selection_probabilities = occupied.to(sequence.dtype)
            selection_logits = torch.where(
                occupied,
                torch.zeros_like(selection_probabilities),
                torch.full_like(selection_probabilities, -torch.inf),
            )
            predicted_involvement = torch.sigmoid(
                self.involvement_head(embedding).squeeze(-1).float()
            )
        direction = F.normalize(self.direction_head(embedding).float(), dim=-1)
        radius = self._radius(predicted_involvement)
        if self.geometry == "euclidean":
            point = direction * radius[:, None]
        else:
            point = self._lorentz_lift(direction, radius)
        prototype_points, prototype_directions, prototype_involvement = (
            self.prototype_points()
        )
        distances = self._distance(point, prototype_points)
        prototype_logits = -distances / self.distance_temperature
        geometry_grade_logits = torch.logsumexp(prototype_logits, dim=2)
        needle_prototype_outputs: dict[str, torch.Tensor] = {}
        if self.needle_prototype_head is not None:
            matched = self.needle_prototype_head(
                local_sequence, sequence, occupied
            )
            grade_logits = matched["logits"]
            needle_prototype_outputs = {
                "needle_prototype_logits": matched["logits"],
                "needle_prototype_activations": matched["activations"],
                "needle_prototype_slot_similarities": matched["slot_similarities"],
                "needle_prototype_slot_attention": matched["slot_attention"],
                "needle_prototype_slot_features": matched["slot_features"],
                "needle_prototype_slot_indices": matched["slot_indices"],
                "needle_prototype_class_weights": (
                    self.needle_prototype_head.class_weights
                ),
                "needle_axis_local_sequence": local_sequence,
                "needle_axis_context_sequence": sequence,
            }
            mixed_pooled = self._mix_style(pooled, occupied)
            if mixed_pooled is not None:
                mixed_local = (
                    self.input_projection(mixed_pooled) + self.position_embedding
                )
                mixed_context = self.axis_encoder(
                    mixed_local, src_key_padding_mask=~occupied
                )
                augmented = self.needle_prototype_head(
                    mixed_local, mixed_context, occupied
                )
                needle_prototype_outputs.update(
                    {
                        "needle_prototype_augmented_logits": augmented["logits"],
                        "needle_prototype_augmented_activations": augmented[
                            "activations"
                        ],
                        "needle_prototype_augmented_slot_attention": augmented[
                            "slot_attention"
                        ],
                    }
                )
        pathology_transport_outputs: dict[str, torch.Tensor] = {}
        if self.pathology_transport_head is not None:
            # Preserve V18's anatomical decision geometry as the modality-
            # private path. Only the shared projector receives cross-modal
            # gradients; the Axis-Proto head continues to learn solely from
            # ultrasound objectives and real-core projection.
            pathology_transport_outputs = self.pathology_transport_head(
                local_sequence.detach(),
                sequence.detach(),
                occupied,
                self.needle_prototype_head.prototype_vectors.detach(),
            )
        else:
            grade_logits = geometry_grade_logits
        three_way_logits = self.target_logits_to_three_way(
            grade_logits, predicted_involvement
        )
        return {
            "needle_axis_embedding": embedding,
            "hyperbolic_direction": direction,
            "hyperbolic_radius": radius,
            "hyperbolic_point": point,
            "hyperbolic_predicted_involvement": predicted_involvement,
            "hyperbolic_prototype_points": prototype_points,
            "hyperbolic_prototype_directions": prototype_directions,
            "hyperbolic_prototype_involvement": prototype_involvement,
            "hyperbolic_prototype_distances": distances,
            "hyperbolic_prototype_logits": prototype_logits,
            "hyperbolic_geometry_grade_logits": geometry_grade_logits,
            "hyperbolic_grade_logits": grade_logits,
            "hyperbolic_three_way_logits": three_way_logits,
            "hyperbolic_fusion_scale": self.fusion_scale,
            "needle_axis_occupied_bins": occupied,
            "needle_selection_logits": selection_logits,
            "needle_selection_probabilities": selection_probabilities,
            **needle_prototype_outputs,
            **pathology_transport_outputs,
        }


class HierarchicalDenseEndpointHead(nn.Module):
    """Two-head cancer/severity classifier with coherent three-state outputs.

    The first logit predicts any cancer. The second predicts csPCa conditional
    on cancer. Their product guarantees P(csPCa) <= P(PCA) without forcing the
    two clinical endpoints to share a flat three-way softmax head.
    """

    def __init__(
        self,
        embedding_dim: int,
        hidden_dim: int,
        use_involvement: bool,
        residual_max_scale: float,
        residual_initial_scale: float,
        residual_fixed_scale: float | None,
    ) -> None:
        super().__init__()
        self.use_involvement = bool(use_involvement)
        input_dim = int(embedding_dim) + int(self.use_involvement)
        self.network = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, int(hidden_dim)),
            nn.GELU(),
            nn.Dropout(0.10),
            nn.Linear(int(hidden_dim), 2),
        )
        self.residual_max_scale = float(residual_max_scale)
        if self.residual_max_scale < 0:
            raise ValueError("dense residual max scale must be non-negative.")
        self.residual_fixed_scale = (
            None if residual_fixed_scale is None else float(residual_fixed_scale)
        )
        if self.residual_fixed_scale is not None:
            if not 0 <= self.residual_fixed_scale <= self.residual_max_scale:
                raise ValueError(
                    "dense residual fixed scale must lie in [0, max scale]."
                )
            self.register_parameter("residual_logit", None)
        else:
            if not 0 < float(residual_initial_scale) < self.residual_max_scale:
                raise ValueError(
                    "dense residual initial scale must lie strictly between zero "
                    "and max scale when it is learned."
                )
            fraction = float(residual_initial_scale) / self.residual_max_scale
            self.residual_logit = nn.Parameter(
                torch.logit(torch.tensor(fraction, dtype=torch.float32))
            )
        nn.init.zeros_(self.network[-1].weight)
        nn.init.zeros_(self.network[-1].bias)

    @property
    def residual_scale(self) -> torch.Tensor:
        if self.residual_fixed_scale is not None:
            return self.network[-1].weight.new_tensor(self.residual_fixed_scale)
        return self.residual_max_scale * torch.sigmoid(self.residual_logit)



class ProstRLAlignedEndpointHead(nn.Module):
    """Two Prost-RL class decoders with coherent PCA/csPCa probabilities.

    The original Prost-RL classifier is a 2-layer SAM two-way Transformer over
    the complete image-token grid, with five clinical prompt tokens.  Earlier
    V20 models called an 8.8k-parameter MLP over a 64-D needle summary a
    "dense" head, which was not a matched replacement.  This module restores
    the actual Prost-RL classifier capacity and prompt interface while keeping
    the occurrence/hyperbolic logits as a bounded explanatory residual.

    Two independent binary decoders are used so the pretrained csPCa decoder
    can initialize both clinical tasks without changing its token topology.
    The second output is interpreted as P(csPCa | PCA), which guarantees
    P(csPCa) <= P(PCA).  Predicted (never ground-truth) involvement is supplied
    as one additional sparse prompt token when enabled.
    """

    prompt_names = ClinicalPromptConditioner.prompt_names

    def __init__(
        self,
        feature_dim: int = 256,
        image_grid_size: int = 16,
        input_image_size: int = 256,
        use_involvement: bool = True,
        residual_max_scale: float = 0.35,
        residual_initial_scale: float = 0.05,
        residual_fixed_scale: float | None = None,
    ) -> None:
        super().__init__()
        self.input_feature_dim = int(feature_dim)
        # The pretrained Prost-RL prompt and two-way Transformer components
        # are intrinsically 256-D.  A backbone-specific 1x1 adapter permits a
        # clean DINOv3 substitution without widening/reinitializing those
        # supervised components.  MedSAM remains an exact identity path.
        self.feature_dim = 256
        self.image_feature_adapter = (
            nn.Identity()
            if self.input_feature_dim == self.feature_dim
            else nn.Conv2d(
                self.input_feature_dim, self.feature_dim, kernel_size=1, bias=True
            )
        )
        if isinstance(self.image_feature_adapter, nn.Conv2d):
            nn.init.orthogonal_(self.image_feature_adapter.weight)
            nn.init.zeros_(self.image_feature_adapter.bias)
        self.use_involvement = bool(use_involvement)
        self.prompt_encoder = PromptEncoder(
            embed_dim=self.feature_dim,
            image_embedding_size=(int(image_grid_size), int(image_grid_size)),
            input_image_size=(int(input_image_size), int(input_image_size)),
            mask_in_chans=16,
        )
        self.null_prompt = nn.Parameter(torch.zeros(1, self.feature_dim))
        self.floating_point_prompt_modules = nn.ModuleDict(
            {
                name: nn.Sequential(
                    nn.Linear(1, 128),
                    nn.ReLU(),
                    nn.Linear(128, self.feature_dim),
                )
                for name in self.prompt_names
            }
        )
        self.involvement_prompt_module = (
            nn.Sequential(
                nn.Linear(1, 128),
                nn.ReLU(),
                nn.Linear(128, self.feature_dim),
            )
            if self.use_involvement
            else None
        )
        if self.involvement_prompt_module is not None:
            # Preserve the pretrained Prost-RL function at initialization; the
            # predicted-q token must earn influence from NCT training.
            nn.init.zeros_(self.involvement_prompt_module[-1].weight)
            nn.init.zeros_(self.involvement_prompt_module[-1].bias)

        def class_decoder() -> ClassDecoder:
            return ClassDecoder(
                transformer_dim=self.feature_dim,
                transformer=TwoWayTransformer(
                    depth=2,
                    embedding_dim=self.feature_dim,
                    mlp_dim=2048,
                    num_heads=8,
                ),
                num_cls_tokens=1,
                cls_output_dims=[2],
            )

        self.pca_decoder = class_decoder()
        self.cspca_decoder = class_decoder()
        self.residual_max_scale = float(residual_max_scale)
        if self.residual_max_scale < 0:
            raise ValueError("Aligned-head residual max scale must be non-negative.")
        self.residual_fixed_scale = (
            None if residual_fixed_scale is None else float(residual_fixed_scale)
        )
        if self.residual_fixed_scale is not None:
            if not 0 <= self.residual_fixed_scale <= self.residual_max_scale:
                raise ValueError(
                    "Aligned-head fixed residual scale must lie in [0, max]."
                )
            self.register_parameter("residual_logit", None)
        else:
            if not 0 < float(residual_initial_scale) < self.residual_max_scale:
                raise ValueError(
                    "Aligned-head initial residual scale must lie strictly between "
                    "zero and max when learned."
                )
            fraction = float(residual_initial_scale) / self.residual_max_scale
            self.residual_logit = nn.Parameter(
                torch.logit(torch.tensor(fraction, dtype=torch.float32))
            )

    @property
    def residual_scale(self) -> torch.Tensor:
        reference = self.pca_decoder.cls_tokens.weight
        if self.residual_fixed_scale is not None:
            return reference.new_tensor(self.residual_fixed_scale)
        return self.residual_max_scale * torch.sigmoid(self.residual_logit)

    @staticmethod
    def _substate(
        state: Mapping[str, torch.Tensor], marker: str
    ) -> dict[str, torch.Tensor]:
        output: dict[str, torch.Tensor] = {}
        for key, value in state.items():
            clean = key.removeprefix("module.")
            if marker in clean:
                output[clean.split(marker, 1)[1]] = value
        return output

    def load_supervised_prost_rl_components(
        self, state: Mapping[str, torch.Tensor]
    ) -> dict[str, Any]:
        """Initialize both endpoint decoders and prompts from Prost-RL stage 1."""
        prompt_state = self._substate(
            state, "prostnfound.medsam_model.prompt_encoder."
        )
        decoder_state = self._substate(state, "prostnfound.class_decoder.")
        if not prompt_state or not decoder_state:
            raise KeyError(
                "The supervised checkpoint does not contain Prost-RL prompt/class "
                "decoder parameters."
            )
        prompt_message = self.prompt_encoder.load_state_dict(prompt_state, strict=True)
        pca_message = self.pca_decoder.load_state_dict(decoder_state, strict=True)
        cspca_message = self.cspca_decoder.load_state_dict(
            decoder_state, strict=True
        )

        prompt_module_state: dict[str, torch.Tensor] = {}
        marker = "prostnfound.floating_point_prompt_modules."
        for key, value in state.items():
            clean = key.removeprefix("module.")
            if marker in clean:
                prompt_module_state[clean.split(marker, 1)[1]] = value
        prompt_module_message = self.floating_point_prompt_modules.load_state_dict(
            prompt_module_state, strict=True
        )
        null_candidates = [
            value
            for key, value in state.items()
            if key.removeprefix("module.").endswith("prostnfound.null_prompt")
        ]
        if len(null_candidates) != 1:
            raise KeyError("Expected exactly one supervised Prost-RL null prompt.")
        self.null_prompt.data.copy_(null_candidates[0])
        return {
            "prompt_encoder": str(prompt_message),
            "pca_decoder": str(pca_message),
            "cspca_decoder": str(cspca_message),
            "clinical_prompts": str(prompt_module_message),
        }

    def _prompt_embeddings(
        self,
        batch_size: int,
        clinical_prompts: torch.Tensor | None,
        predicted_involvement: torch.Tensor,
    ) -> torch.Tensor:
        tokens = []
        if clinical_prompts is not None:
            if clinical_prompts.shape != (batch_size, len(self.prompt_names)):
                raise ValueError(
                    "Aligned Prost-RL prompts must have shape "
                    f"[{batch_size}, {len(self.prompt_names)}], got "
                    f"{tuple(clinical_prompts.shape)}."
                )
            for index, name in enumerate(self.prompt_names):
                tokens.append(
                    self.floating_point_prompt_modules[name](
                        clinical_prompts[:, index : index + 1]
                    )[:, None, :]
                )
        else:
            tokens.extend(
                self.null_prompt.expand(batch_size, -1)[:, None, :]
                for _ in self.prompt_names
            )
        if self.involvement_prompt_module is not None:
            # Add predicted burden to an existing clinical token so the sparse
            # token count remains exactly the pretrained Prost-RL topology.
            # With the zero-initialized output layer this is initially an exact
            # no-op, unlike appending a zero token (which still changes the
            # Transformer's attention normalization).
            tokens[-1] = tokens[-1] + self.involvement_prompt_module(
                predicted_involvement.float().reshape(batch_size, 1)
            )[:, None, :]
        return torch.cat(tokens, dim=1)

    def forward(
        self,
        image_features: torch.Tensor,
        clinical_prompts: torch.Tensor | None,
        predicted_involvement: torch.Tensor,
        residual_logits: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        image_features = self.image_feature_adapter(image_features.float())
        batch_size = len(image_features)
        sparse = self._prompt_embeddings(
            batch_size, clinical_prompts, predicted_involvement
        )
        _, dense = self.prompt_encoder(None, None, None)
        if dense.shape[-2:] != image_features.shape[-2:]:
            dense = F.interpolate(dense, size=image_features.shape[-2:])
        dense = dense.expand(batch_size, -1, -1, -1)
        positional = self.prompt_encoder.get_dense_pe()
        if positional.shape[-2:] != image_features.shape[-2:]:
            positional = F.interpolate(positional, size=image_features.shape[-2:])
        pca_logits = self.pca_decoder(
            image_features, positional, sparse, dense
        )[0]
        conditional_cspca_logits = self.cspca_decoder(
            image_features, positional, sparse, dense
        )[0]
        log_pca = F.log_softmax(pca_logits.float(), dim=1)
        log_conditional = F.log_softmax(conditional_cspca_logits.float(), dim=1)
        hierarchical_logits = torch.stack(
            (
                log_pca[:, 0],
                log_pca[:, 1] + log_conditional[:, 0],
                log_pca[:, 1] + log_conditional[:, 1],
            ),
            dim=1,
        )
        centered_residual = residual_logits.float() - residual_logits.float().mean(
            dim=1, keepdim=True
        )
        final_logits = (
            hierarchical_logits + self.residual_scale * centered_residual
        )
        return {
            "prost_rl_endpoint_pca_logits": pca_logits,
            "prost_rl_endpoint_conditional_cspca_logits": conditional_cspca_logits,
            "prost_rl_endpoint_three_way_logits": hierarchical_logits,
            "prost_rl_endpoint_final_logits": final_logits,
            "prost_rl_endpoint_residual_scale": self.residual_scale,
        }


class ProtoProstate(nn.Module):
    """Dense image encoder with prototype evidence and coherent endpoint heads."""

    def __init__(
        self,
        backbone: str = "medsam_adapter",
        backbone_kw: dict[str, Any] | None = None,
        feature_dim: int = 256,
        num_classes: int = 2,
        prototypes_per_class: int = 10,
        num_subprototypes: int = 4,
        adjacency_radius: int = 1,
        slot_temperature: float = 1.0,
        coverage_min_cancer_involvement: float = 0.4,
        prototype_grade_groups: list[int] | tuple[int, ...] | None = None,
        grade_thresholds: list[int] | tuple[int, ...] | None = None,
        matching_mode: str = "deformable",
        occurrence_hidden_dim: int = 128,
        num_uncertainty_prototypes: int = 0,
        occurrence_temperature: float = 1.0,
        learnable_occurrence_temperature: bool = False,
        occurrence_region_target_tokens: float = 0.0,
        occurrence_locality_sigma_tokens: float = 0.0,
        occurrence_locality_strength: float = 1.0,
        clinical_prompts: bool = False,
        prompt_conditioning: bool = False,
        prompt_hidden_dim: int = 128,
        prompt_feature_scale: float = 0.25,
        prompt_spatial_scale: float = 1.0,
        within_core_reference_subtraction: bool = False,
        prototype_search_region: str = "intersection",
        needle_constrained_prototype_fraction: float = 0.0,
        prototype_search_dilation: int = 0,
        prostate_token_min_fraction: float = 0.0,
        matching_pool_kernel: int = 1,
        classification_target: str = "pca",
        share_core_classifier: bool = False,
        connected_greedy: bool = False,
        greedy_connectivity: int = 4,
        greedy_straight_through_temperature: float = 0.0,
        needle_anchor_dilation: int = 0,
        needle_axis_bins: int = 0,
        hyperbolic_needle: bool = False,
        hyperbolic_geometry: str = "hyperbolic",
        hyperbolic_embedding_dim: int = 64,
        hyperbolic_axis_bins: int = 16,
        hyperbolic_transformer_layers: int = 1,
        hyperbolic_transformer_heads: int = 4,
        hyperbolic_prototypes_per_grade: int = 2,
        hyperbolic_curvature: float = 1.0,
        hyperbolic_radius_min: float = 0.05,
        hyperbolic_radius_max: float = 2.5,
        hyperbolic_distance_temperature: float = 0.5,
        hyperbolic_fusion_min_scale: float = 0.0,
        hyperbolic_fusion_max_scale: float = 0.5,
        hyperbolic_fusion_initial_scale: float = 0.1,
        hyperbolic_fusion_fixed_scale: float | None = None,
        hyperbolic_base_logit_scale: float = 1.0,
        branch_fusion_mode: str = "legacy_residual",
        branch_fusion_global_min: float = 0.35,
        branch_fusion_global_max: float = 0.75,
        branch_fusion_global_initial: float = 0.60,
        hyperbolic_projection_q_tolerance: float = 0.2,
        hyperbolic_unique_projection: bool = False,
        hyperbolic_projection_q_weight: float = 0.5,
        hyperbolic_projection_candidate_limit: int = 256,
        hyperbolic_target_mode: str = "grade5",
        hyperbolic_shared_class_directions: bool = False,
        hyperbolic_structured_needle_selection: bool = False,
        needle_prototype_classifier: bool = False,
        needle_prototype_slots: int = 4,
        needle_prototype_attention_temperature: float = 0.20,
        needle_prototype_style_mix_probability: float = 0.0,
        needle_prototype_style_mix_alpha: float = 0.30,
        pathology_transport_bank_path: str | None = None,
        pathology_transport_hidden_dim: int = 128,
        pathology_transport_attention_temperature: float = 0.20,
        pathology_transport_temperature: float = 0.10,
        pathology_transport_sinkhorn_iterations: int = 6,
        pathology_transport_fusion_max_scale: float = 0.20,
        pathology_transport_fusion_initial_scale: float = 0.05,
        pathology_transport_fusion_fixed_scale: float | None = None,
        dense_endpoint_head: bool = False,
        dense_endpoint_hidden_dim: int = 128,
        dense_endpoint_use_involvement: bool = False,
        dense_endpoint_residual_max_scale: float = 0.50,
        dense_endpoint_residual_initial_scale: float = 0.10,
        dense_endpoint_residual_fixed_scale: float | None = None,
        prost_rl_endpoint_head: bool = False,
        prost_rl_endpoint_use_involvement: bool = True,
        prost_rl_endpoint_residual_max_scale: float = 0.35,
        prost_rl_endpoint_residual_initial_scale: float = 0.05,
        prost_rl_endpoint_residual_fixed_scale: float | None = None,
        pathology_geometry_teacher_dim: int = 0,
        pathology_geometry_hidden_dim: int = 128,
        semantic_anchor_dim: int = 0,
        semantic_anchor_hidden_dim: int = 128,
        semantic_anchor_temperature_init: float = 0.07,
        semantic_anchor_temperature_min: float = 0.02,
        semantic_anchor_temperature_max: float = 0.50,
        semantic_anchor_bank_path: str | None = None,
        semantic_anchor_direct_head: bool = False,
        semantic_anchor_fusion_max_scale: float = 0.30,
        semantic_anchor_fusion_initial_scale: float = 0.05,
        semantic_anchor_label_mode: str = "clinical_state",
        center_adversarial_num_centers: int = 0,
        center_adversarial_num_states: int = 3,
        center_adversarial_hidden_dim: int = 64,
        unique_prototype_projection: bool = False,
        prototype_projection_candidate_limit: int = 256,
        lock_projected_prototypes: bool = False,
        freeze_encoder: bool = True,
    ) -> None:
        super().__init__()
        backbone_model = create_model(backbone, **(backbone_kw or {}))
        # SAM factories return a container with ``image_encoder`` whereas
        # dense-feature backbones such as DINOv3 are already encoders. Keeping
        # both forms supported preserves every historical MedSAM checkpoint.
        self.image_encoder = getattr(backbone_model, "image_encoder", backbone_model)
        self.feature_dim = feature_dim
        self.num_classes = num_classes
        self.matching_mode = matching_mode
        self.num_uncertainty_prototypes = int(num_uncertainty_prototypes)
        self.within_core_reference_subtraction = bool(
            within_core_reference_subtraction
        )
        if prototype_search_region not in {"intersection", "prostate", "image"}:
            raise ValueError(
                "prototype_search_region must be intersection, prostate, or image."
            )
        self.prototype_search_region = prototype_search_region
        if prototype_search_dilation < 0:
            raise ValueError("prototype_search_dilation must be non-negative.")
        self.prototype_search_dilation = int(prototype_search_dilation)
        if not 0.0 <= float(prostate_token_min_fraction) <= 1.0:
            raise ValueError("prostate_token_min_fraction must be in [0,1].")
        self.prostate_token_min_fraction = float(prostate_token_min_fraction)
        if matching_pool_kernel < 1 or matching_pool_kernel % 2 == 0:
            raise ValueError("matching_pool_kernel must be a positive odd integer.")
        self.matching_pool_kernel = int(matching_pool_kernel)
        if classification_target not in {"pca", "cspca", "three_way", "isup"}:
            raise ValueError(
                "classification_target must be 'pca', 'cspca', 'three_way', or 'isup'."
            )
        self.classification_target = classification_target
        self.share_core_classifier = bool(share_core_classifier)
        self.needle_axis_bins = int(needle_axis_bins)
        if self.needle_axis_bins not in (0,) and self.needle_axis_bins < 2:
            raise ValueError("needle_axis_bins must be zero or at least 2.")
        self.lock_projected_prototypes = bool(lock_projected_prototypes)
        if hyperbolic_base_logit_scale < 0:
            raise ValueError("hyperbolic_base_logit_scale must be non-negative.")
        self.hyperbolic_base_logit_scale = float(hyperbolic_base_logit_scale)
        if branch_fusion_mode not in {"legacy_residual", "normalized_convex"}:
            raise ValueError(
                "branch_fusion_mode must be legacy_residual or normalized_convex."
            )
        self.branch_fusion_mode = str(branch_fusion_mode)
        self.branch_fusion_global_min = float(branch_fusion_global_min)
        self.branch_fusion_global_max = float(branch_fusion_global_max)
        self.branch_fusion_logit: nn.Parameter | None = None
        if self.branch_fusion_mode == "normalized_convex":
            if not hyperbolic_needle:
                raise ValueError(
                    "Normalized convex fusion requires the whole-needle branch."
                )
            if not (
                0.0 <= self.branch_fusion_global_min
                < float(branch_fusion_global_initial)
                < self.branch_fusion_global_max
                <= 1.0
            ):
                raise ValueError(
                    "Normalized convex fusion requires 0 <= min < initial < max <= 1."
                )
            initial_fraction = (
                float(branch_fusion_global_initial) - self.branch_fusion_global_min
            ) / (self.branch_fusion_global_max - self.branch_fusion_global_min)
            self.branch_fusion_logit = nn.Parameter(
                torch.logit(torch.tensor(initial_fraction, dtype=torch.float32))
            )
        if hyperbolic_needle and classification_target not in {"three_way", "isup"}:
            raise ValueError(
                "The hyperbolic grade cone requires three_way or ISUP classification."
            )
        if pathology_transport_bank_path and classification_target != "three_way":
            raise ValueError(
                "Pathology prototype transport requires three-way classification."
            )
        if prototype_projection_candidate_limit < prototypes_per_class:
            raise ValueError(
                "prototype_projection_candidate_limit must be at least "
                "prototypes_per_class."
            )
        self.unique_prototype_projection = bool(unique_prototype_projection)
        self.prototype_projection_candidate_limit = int(
            prototype_projection_candidate_limit
        )
        if needle_anchor_dilation < 0:
            raise ValueError("needle_anchor_dilation must be non-negative.")
        self.needle_anchor_dilation = int(needle_anchor_dilation)
        if not 0.0 <= needle_constrained_prototype_fraction <= 1.0:
            raise ValueError("needle_constrained_prototype_fraction must be in [0,1].")
        self.prototype_layer = DeformablePrototypeLayer(
            feature_dim=feature_dim,
            num_classes=num_classes,
            prototypes_per_class=prototypes_per_class,
            num_subprototypes=num_subprototypes,
            adjacency_radius=adjacency_radius,
            slot_temperature=slot_temperature,
            coverage_min_cancer_involvement=coverage_min_cancer_involvement,
            prototype_grade_groups=prototype_grade_groups,
            connected_greedy=connected_greedy,
            greedy_connectivity=greedy_connectivity,
            greedy_straight_through_temperature=greedy_straight_through_temperature,
        )
        self.grade_thresholds = (
            tuple(int(value) for value in grade_thresholds)
            if grade_thresholds is not None
            else ()
        )
        if self.grade_thresholds and self.grade_thresholds != (3, 4, 5):
            raise ValueError(
                "Hierarchical NCT2013 grading expects conditional cancer "
                "thresholds (3, 4, 5); GG1 is absent."
            )
        if matching_mode not in {"deformable", "occurrence"}:
            raise ValueError("matching_mode must be 'deformable' or 'occurrence'.")
        if self.num_uncertainty_prototypes and matching_mode != "occurrence":
            raise ValueError("Uncertainty prototypes require occurrence matching.")
        self.occurrence_matcher = (
            MaskedPrototypeOccurrence(
                feature_dim=feature_dim,
                num_prototypes=self.prototype_layer.num_prototypes,
                num_subprototypes=num_subprototypes,
                hidden_dim=occurrence_hidden_dim,
                num_uncertainty_prototypes=self.num_uncertainty_prototypes,
                occurrence_temperature=occurrence_temperature,
                learnable_temperature=learnable_occurrence_temperature,
                locality_sigma_tokens=occurrence_locality_sigma_tokens,
                locality_strength=occurrence_locality_strength,
            )
            if matching_mode == "occurrence"
            else None
        )
        if float(occurrence_region_target_tokens) < 0:
            raise ValueError("occurrence_region_target_tokens must be non-negative.")
        self.occurrence_region_target_tokens = float(
            occurrence_region_target_tokens
        )
        self.center_adversarial_num_centers = int(center_adversarial_num_centers)
        self.center_adversarial_num_states = int(center_adversarial_num_states)
        self.center_discriminator = None
        if clinical_prompts and prompt_conditioning:
            raise ValueError(
                "Use either the legacy additive clinical head or prompt "
                "conditioning, not both."
            )
        self.clinical_head = ClinicalAdditiveHead() if clinical_prompts else None
        self.prompt_conditioner = (
            ClinicalPromptConditioner(
                feature_dim=feature_dim,
                num_classes=num_classes,
                hidden_dim=prompt_hidden_dim,
                feature_scale=prompt_feature_scale,
                spatial_scale=prompt_spatial_scale,
            )
            if prompt_conditioning
            else None
        )
        self.hyperbolic_needle_encoder = (
            HyperbolicNeedleEncoder(
                feature_dim=feature_dim,
                embedding_dim=hyperbolic_embedding_dim,
                axis_bins=hyperbolic_axis_bins,
                transformer_layers=hyperbolic_transformer_layers,
                transformer_heads=hyperbolic_transformer_heads,
                prototypes_per_grade=hyperbolic_prototypes_per_grade,
                curvature=hyperbolic_curvature,
                radius_min=hyperbolic_radius_min,
                radius_max=hyperbolic_radius_max,
                distance_temperature=hyperbolic_distance_temperature,
                fusion_min_scale=hyperbolic_fusion_min_scale,
                fusion_max_scale=hyperbolic_fusion_max_scale,
                fusion_initial_scale=hyperbolic_fusion_initial_scale,
                fusion_fixed_scale=hyperbolic_fusion_fixed_scale,
                projection_q_tolerance=hyperbolic_projection_q_tolerance,
                unique_projection=hyperbolic_unique_projection,
                projection_q_weight=hyperbolic_projection_q_weight,
                projection_candidate_limit=hyperbolic_projection_candidate_limit,
                target_mode=hyperbolic_target_mode,
                shared_class_directions=hyperbolic_shared_class_directions,
                structured_needle_selection=hyperbolic_structured_needle_selection,
                needle_prototype_classifier=needle_prototype_classifier,
                needle_prototype_slots=needle_prototype_slots,
                needle_prototype_attention_temperature=(
                    needle_prototype_attention_temperature
                ),
                needle_prototype_style_mix_probability=(
                    needle_prototype_style_mix_probability
                ),
                needle_prototype_style_mix_alpha=needle_prototype_style_mix_alpha,
                pathology_transport_bank_path=pathology_transport_bank_path,
                pathology_transport_hidden_dim=pathology_transport_hidden_dim,
                pathology_transport_attention_temperature=(
                    pathology_transport_attention_temperature
                ),
                pathology_transport_temperature=pathology_transport_temperature,
                pathology_transport_sinkhorn_iterations=(
                    pathology_transport_sinkhorn_iterations
                ),
                pathology_transport_fusion_max_scale=(
                    pathology_transport_fusion_max_scale
                ),
                pathology_transport_fusion_initial_scale=(
                    pathology_transport_fusion_initial_scale
                ),
                pathology_transport_fusion_fixed_scale=(
                    pathology_transport_fusion_fixed_scale
                ),
                geometry=hyperbolic_geometry,
            )
            if hyperbolic_needle
            else None
        )
        if dense_endpoint_head and classification_target != "three_way":
            raise ValueError(
                "The hierarchical dense endpoint head requires three-way targets."
            )
        if dense_endpoint_head and self.hyperbolic_needle_encoder is None:
            raise ValueError(
                "The dense endpoint head requires the ordered whole-needle encoder."
            )
        self.dense_endpoint_head = (
            HierarchicalDenseEndpointHead(
                embedding_dim=hyperbolic_embedding_dim,
                hidden_dim=dense_endpoint_hidden_dim,
                use_involvement=dense_endpoint_use_involvement,
                residual_max_scale=dense_endpoint_residual_max_scale,
                residual_initial_scale=dense_endpoint_residual_initial_scale,
                residual_fixed_scale=dense_endpoint_residual_fixed_scale,
            )
            if dense_endpoint_head
            else None
        )
        if prost_rl_endpoint_head and classification_target != "three_way":
            raise ValueError(
                "The Prost-RL aligned endpoint head requires three-way targets."
            )
        if prost_rl_endpoint_head and self.hyperbolic_needle_encoder is None:
            raise ValueError(
                "The Prost-RL aligned endpoint head requires the involvement-aware "
                "whole-needle branch."
            )
        if prost_rl_endpoint_head and dense_endpoint_head:
            raise ValueError(
                "Enable either the old needle-summary dense head or the Prost-RL "
                "aligned full-grid head, not both."
            )
        self.prost_rl_endpoint_head = (
            ProstRLAlignedEndpointHead(
                feature_dim=feature_dim,
                image_grid_size=16,
                input_image_size=256,
                use_involvement=prost_rl_endpoint_use_involvement,
                residual_max_scale=prost_rl_endpoint_residual_max_scale,
                residual_initial_scale=prost_rl_endpoint_residual_initial_scale,
                residual_fixed_scale=prost_rl_endpoint_residual_fixed_scale,
            )
            if prost_rl_endpoint_head
            else None
        )
        if self.center_adversarial_num_centers > 0:
            if self.hyperbolic_needle_encoder is None:
                raise ValueError(
                    "Center-adversarial training requires the global needle encoder."
                )
            if self.center_adversarial_num_states < 1:
                raise ValueError("center_adversarial_num_states must be positive.")
            center_input_dim = (
                self.prototype_layer.num_prototypes + int(hyperbolic_embedding_dim)
            )
            self.center_discriminator = nn.Sequential(
                nn.LayerNorm(center_input_dim),
                nn.Linear(center_input_dim, int(center_adversarial_hidden_dim)),
                nn.GELU(),
                nn.Linear(
                    int(center_adversarial_hidden_dim),
                    self.center_adversarial_num_states
                    * self.center_adversarial_num_centers,
                ),
            )
        self.pathology_geometry_projector = None
        if int(pathology_geometry_teacher_dim) > 0:
            if self.hyperbolic_needle_encoder is None:
                raise ValueError(
                    "Pathology geometry transfer requires the global needle encoder."
                )
            geometry_input_dim = (
                self.prototype_layer.num_prototypes + int(hyperbolic_embedding_dim)
            )
            self.pathology_geometry_projector = nn.Sequential(
                nn.LayerNorm(geometry_input_dim),
                nn.Linear(geometry_input_dim, int(pathology_geometry_hidden_dim)),
                nn.GELU(),
                nn.Linear(
                    int(pathology_geometry_hidden_dim),
                    int(pathology_geometry_teacher_dim),
                ),
            )
        self.semantic_anchor_core_projector = None
        self.semantic_anchor_prototype_projector = None
        self.semantic_anchor_log_temperature = None
        self.semantic_anchor_direct_head = bool(semantic_anchor_direct_head)
        self.semantic_anchor_fusion_max_scale = float(
            semantic_anchor_fusion_max_scale
        )
        self.semantic_anchor_fusion_logit = None
        self.semantic_anchor_temperature_min = float(semantic_anchor_temperature_min)
        self.semantic_anchor_temperature_max = float(semantic_anchor_temperature_max)
        if semantic_anchor_label_mode not in {
            "clinical_state",
            "grade_bands_01_23_45",
        }:
            raise ValueError(
                "semantic_anchor_label_mode must be clinical_state or "
                "grade_bands_01_23_45."
            )
        self.semantic_anchor_label_mode = str(semantic_anchor_label_mode)
        if (
            self.semantic_anchor_label_mode == "grade_bands_01_23_45"
            and (
                self.hyperbolic_needle_encoder is None
                or self.hyperbolic_needle_encoder.target_mode
                != "grade_bands_01_23_45"
            )
        ):
            raise ValueError(
                "Ordinal-band pathology anchoring requires the matching "
                "GG0-1/GG2-3/GG4-5 hyperbolic target mode."
            )
        if int(semantic_anchor_dim) > 0:
            if self.hyperbolic_needle_encoder is None:
                raise ValueError(
                    "Semantic-anchor supervision requires the global needle encoder."
                )
            if not (
                0 < self.semantic_anchor_temperature_min
                <= float(semantic_anchor_temperature_init)
                <= self.semantic_anchor_temperature_max
            ):
                raise ValueError(
                    "Semantic-anchor temperatures must satisfy "
                    "0 < min <= initial <= max."
                )
            semantic_core_dim = (
                self.prototype_layer.num_prototypes + int(hyperbolic_embedding_dim)
            )
            semantic_hidden = int(semantic_anchor_hidden_dim)
            semantic_output = int(semantic_anchor_dim)
            self.semantic_anchor_core_projector = nn.Sequential(
                nn.LayerNorm(semantic_core_dim),
                nn.Linear(semantic_core_dim, semantic_hidden),
                nn.GELU(),
                nn.Dropout(0.1),
                nn.Linear(semantic_hidden, semantic_output),
            )
            semantic_prototype_dim = (
                int(hyperbolic_embedding_dim)
                if self.semantic_anchor_label_mode == "grade_bands_01_23_45"
                else feature_dim
            )
            self.semantic_anchor_prototype_projector = nn.Sequential(
                nn.LayerNorm(semantic_prototype_dim),
                nn.Linear(semantic_prototype_dim, semantic_hidden),
                nn.GELU(),
                nn.Dropout(0.1),
                nn.Linear(semantic_hidden, semantic_output),
            )
            self.semantic_anchor_log_temperature = nn.Parameter(
                torch.tensor(float(semantic_anchor_temperature_init)).log()
            )
            if self.semantic_anchor_direct_head:
                if classification_target != "three_way":
                    raise ValueError(
                        "The direct semantic-anchor head requires three-way "
                        "benign/non-csPCa/csPCa classification."
                    )
                if not semantic_anchor_bank_path:
                    raise ValueError(
                        "semantic_anchor_bank_path is required for the direct head."
                    )
                archive = np.load(
                    str(Path(semantic_anchor_bank_path).expanduser().resolve()),
                    allow_pickle=False,
                )
                anchors = F.normalize(
                    torch.as_tensor(archive["anchors"], dtype=torch.float), dim=1
                )
                if self.semantic_anchor_label_mode == "grade_bands_01_23_45":
                    if "grade_labels" not in archive:
                        raise ValueError(
                            "Ordinal-band pathology anchoring requires grade_labels."
                        )
                    grade_labels = torch.as_tensor(
                        archive["grade_labels"], dtype=torch.long
                    )
                    anchor_states = (
                        (grade_labels >= 2).long() + (grade_labels >= 4).long()
                    )
                else:
                    anchor_states = torch.as_tensor(
                        archive["state_labels"], dtype=torch.long
                    )
                if anchors.ndim != 2 or anchors.shape[1] != semantic_output:
                    raise ValueError(
                        "Direct semantic-anchor bank shape does not match "
                        f"semantic_anchor_dim: {tuple(anchors.shape)} vs "
                        f"{semantic_output}."
                    )
                if len(anchors) != len(anchor_states) or set(
                    anchor_states.tolist()
                ) != {0, 1, 2}:
                    raise ValueError(
                        "Direct semantic-anchor bank must contain aligned states "
                        "0, 1, and 2."
                    )
                if not (
                    0.0 < float(semantic_anchor_fusion_initial_scale)
                    < self.semantic_anchor_fusion_max_scale
                ):
                    raise ValueError(
                        "semantic_anchor_fusion_initial_scale must be between "
                        "zero and semantic_anchor_fusion_max_scale."
                    )
                self.register_buffer("semantic_anchor_bank", anchors)
                self.register_buffer("semantic_anchor_states", anchor_states)
                initial_fraction = float(semantic_anchor_fusion_initial_scale) / (
                    self.semantic_anchor_fusion_max_scale
                )
                self.semantic_anchor_fusion_logit = nn.Parameter(
                    torch.logit(torch.tensor(initial_fraction))
                )
        self.clinical_prompt_names = (
            ClinicalAdditiveHead.prompt_names
            if clinical_prompts
            else (
                ClinicalPromptConditioner.prompt_names
                if prompt_conditioning
                else ()
            )
        )
        constrained = torch.zeros(self.prototype_layer.num_prototypes, dtype=torch.bool)
        constrained_per_class = round(
            prototypes_per_class * float(needle_constrained_prototype_fraction)
        )
        for class_idx in range(num_classes):
            start = class_idx * prototypes_per_class
            constrained[start : start + constrained_per_class] = True
        self.register_buffer(
            "needle_constrained_prototypes", constrained, persistent=False
        )
        # Keep weak proportion calibration separate from ProtoViT's convex
        # classification layer. Classification-only fine-tuning must not alter
        # the per-token evidence map trained against involvement q.
        self.evidence_layer = nn.Linear(
            self.prototype_layer.num_prototypes, 1, bias=True
        )
        self.prototype_classifier = nn.Linear(
            self.prototype_layer.num_prototypes, num_classes, bias=False
        )
        # Separate two-class csPCa head, mirroring Prost-RL's two-head design.
        self.core_classifier = (
            None
            if self.share_core_classifier
            else nn.Linear(self.prototype_layer.num_prototypes, 2, bias=True)
        )
        # Ordered Grade Group supervision is derived only from prototype
        # activations, so the auxiliary prediction remains inspectable through
        # prototype weights. Four cumulative logits represent GG>=2/3/4/5.
        self.grade_classifier = (
            nn.Linear(
                self.prototype_layer.num_prototypes,
                len(self.grade_thresholds),
                bias=True,
            )
            if self.grade_thresholds
            else None
        )
        self.uncertainty_head = (
            nn.Linear(self.num_uncertainty_prototypes, 1)
            if self.num_uncertainty_prototypes
            else None
        )
        self._initialize_last_layers()
        if freeze_encoder:
            self.freeze_encoder()

    @property
    def prototype_class_identity(self) -> torch.Tensor:
        return self.prototype_layer.prototype_class_identity

    @property
    def num_prototypes(self) -> int:
        return self.prototype_layer.num_prototypes

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    def _initialize_last_layers(self) -> None:
        correct = self.prototype_class_identity.T
        self.prototype_classifier.weight.data.copy_(
            correct - 0.5 * (1.0 - correct)
        )
        if self.num_classes >= 3:
            evidence_sign = (
                self.prototype_class_identity[:, 1:].sum(1)
                - self.prototype_class_identity[:, 0]
            )
        else:
            evidence_sign = (
                self.prototype_class_identity[:, 1]
                - self.prototype_class_identity[:, 0]
            )
        self.evidence_layer.weight.data.copy_(0.25 * evidence_sign.unsqueeze(0))
        nn.init.zeros_(self.evidence_layer.bias)
        if self.core_classifier is not None:
            if self.num_classes == 3:
                cspca_positive = self.prototype_class_identity[:, 2]
                cspca_correct = torch.stack((1.0 - cspca_positive, cspca_positive))
                self.core_classifier.weight.data.copy_(
                    cspca_correct - 0.5 * (1.0 - cspca_correct)
                )
            else:
                self.core_classifier.weight.data.copy_(correct - 0.5 * (1.0 - correct))
            nn.init.zeros_(self.core_classifier.bias)
        if self.grade_classifier is not None:
            cancer_identity = self.prototype_class_identity[:, 1].bool()
            for threshold_idx, threshold in enumerate(self.grade_thresholds):
                self.grade_classifier.weight.data[threshold_idx].copy_(
                    torch.where(
                        cancer_identity,
                        torch.full_like(cancer_identity, 0.05, dtype=torch.float),
                        torch.full_like(cancer_identity, -0.0125, dtype=torch.float),
                    )
                )
            nn.init.zeros_(self.grade_classifier.bias)
        if self.uncertainty_head is not None:
            nn.init.constant_(self.uncertainty_head.weight, 0.1)
            nn.init.constant_(self.uncertainty_head.bias, -2.0)

    def freeze_encoder(self) -> None:
        for parameter in self.image_encoder.parameters():
            parameter.requires_grad = False

    def unfreeze_adapters(self) -> int:
        """Unfreeze only adapter parameters; return their parameter count."""
        self.freeze_encoder()
        count = 0
        for name, parameter in self.image_encoder.named_parameters():
            if "adapter" in name.lower():
                parameter.requires_grad = True
                count += parameter.numel()
        return count

    def unfreeze_encoder(self) -> int:
        """Unfreeze the complete image encoder for domain adaptation."""
        count = 0
        for parameter in self.image_encoder.parameters():
            parameter.requires_grad = True
            count += parameter.numel()
        return count

    def unfreeze_lora(self) -> int:
        """Unfreeze only LoRA matrices in a parameter-efficient backbone."""
        self.freeze_encoder()
        count = 0
        for name, parameter in self.image_encoder.named_parameters():
            if "lora_" in name.lower():
                parameter.requires_grad = True
                count += parameter.numel()
        if count == 0:
            raise RuntimeError("LoRA mode selected but the image encoder has no LoRA parameters.")
        return count

    def unfreeze_registers(self) -> int:
        """Unfreeze only DINOv3 storage/register tokens."""
        self.freeze_encoder()
        count = 0
        for name, parameter in self.image_encoder.named_parameters():
            if name.endswith("storage_tokens"):
                parameter.requires_grad = True
                count += parameter.numel()
        if count == 0:
            raise RuntimeError(
                "Register mode selected but the image encoder has no storage_tokens parameter."
            )
        return count

    @staticmethod
    def _checkpoint_state(checkpoint: Mapping[str, Any]) -> Mapping[str, Any]:
        state: Mapping[str, Any] = checkpoint
        for key in ("model", "state_dict"):
            if key in state and isinstance(state[key], Mapping):
                state = state[key]
        return state

    def load_supervised_encoder_checkpoint(
        self, checkpoint_path: str, strict: bool = True
    ) -> torch.nn.modules.module._IncompatibleKeys:
        """Load only the MedSAM image encoder from a Prost-RL stage-1 checkpoint."""
        raw = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        state = self._checkpoint_state(raw)
        encoder_state: dict[str, torch.Tensor] = {}
        marker = "medsam_model.image_encoder."
        direct_marker = "image_encoder."
        for key, value in state.items():
            clean_key = key.removeprefix("module.")
            if marker in clean_key:
                encoder_state[clean_key.split(marker, 1)[1]] = value
            elif direct_marker in clean_key:
                encoder_state[clean_key.split(direct_marker, 1)[1]] = value
        if not encoder_state:
            raise KeyError(
                "No MedSAM image-encoder keys were found in the supervised "
                f"checkpoint: {checkpoint_path}"
            )
        message = self.image_encoder.load_state_dict(encoder_state, strict=strict)
        logging.info("Loaded supervised Prost-RL encoder: %s", message)
        if self.prost_rl_endpoint_head is not None:
            aligned_message = (
                self.prost_rl_endpoint_head.load_supervised_prost_rl_components(
                    state
                )
            )
            logging.info(
                "Loaded supervised Prost-RL prompt and class decoders: %s",
                aligned_message,
            )
        return message

    def load_supervised_endpoint_checkpoint(self, checkpoint_path: str) -> dict[str, Any]:
        """Load only Prost-RL prompt/classification components.

        This is used by non-MedSAM backbones.  Loading the task head separately
        avoids either trying to insert MedSAM weights into DINOv3 or discarding
        the supervised classifier initialization that made V20g competitive.
        """
        if self.prost_rl_endpoint_head is None:
            raise RuntimeError(
                "A supervised endpoint checkpoint requires prost_rl_endpoint_head."
            )
        raw = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        state = self._checkpoint_state(raw)
        message = self.prost_rl_endpoint_head.load_supervised_prost_rl_components(
            state
        )
        logging.info(
            "Loaded supervised Prost-RL prompt and class decoders only: %s",
            message,
        )
        return message

    @staticmethod
    def valid_token_mask(
        prostate_mask: torch.Tensor,
        needle_mask: torch.Tensor,
        grid_size: tuple[int, int],
    ) -> torch.Tensor:
        valid = (prostate_mask > 0.5) & (needle_mask > 0.5)
        # Max pooling preserves thin needle traces better than nearest-neighbor
        # resizing when moving from the 64x64 loss mask to the 16x16 token grid.
        valid = F.adaptive_max_pool2d(valid.float(), grid_size) > 0.5
        return valid.flatten(1)

    def encode(self, image: torch.Tensor) -> torch.Tensor:
        features = self.image_encoder(image)
        if features.ndim != 4 or features.shape[1] != self.feature_dim:
            raise RuntimeError(
                "Expected image encoder output [B, "
                f"{self.feature_dim}, H, W], got {tuple(features.shape)}"
            )
        return features

    def extract_prototype_features(
        self,
        image: torch.Tensor,
        prostate_mask: torch.Tensor,
        needle_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        features = self.encode(image)
        grid_size = features.shape[-2:]
        supervision_tokens = self.valid_token_mask(
            prostate_mask, needle_mask, grid_size
        )
        prostate_binary = (prostate_mask > 0.5).float()
        prostate_overlap_tokens = (
            F.adaptive_max_pool2d(prostate_binary, grid_size) > 0.5
        ).flatten(1)
        if self.prostate_token_min_fraction > 0:
            prostate_tokens = (
                F.adaptive_avg_pool2d(prostate_binary, grid_size)
                >= self.prostate_token_min_fraction
            ).flatten(1)
            empty = ~prostate_tokens.any(dim=1)
            prostate_tokens[empty] = prostate_overlap_tokens[empty]
        else:
            prostate_tokens = prostate_overlap_tokens
        search_prostate_tokens = prostate_tokens
        if self.prototype_search_dilation > 0:
            radius = self.prototype_search_dilation
            search_prostate_tokens = (
                F.max_pool2d(
                    prostate_tokens.reshape(len(features), 1, *grid_size).float(),
                    kernel_size=2 * radius + 1,
                    stride=1,
                    padding=radius,
                )
                > 0.5
            ).flatten(1)
        if self.within_core_reference_subtraction:
            reference_tokens = prostate_tokens & ~supervision_tokens
            flat = features.flatten(2)
            references = []
            for sample_idx in range(len(features)):
                mask = reference_tokens[sample_idx]
                if not bool(mask.any()):
                    mask = prostate_tokens[sample_idx]
                references.append(flat[sample_idx, :, mask].median(-1).values)
            reference = torch.stack(references)[:, :, None, None]
            features = features - reference
        if self.matching_pool_kernel > 1:
            # A prototype slot represents a real local latent region rather
            # than a single fine token. Projection therefore copies the same
            # pooled training feature used during classification.
            radius = self.matching_pool_kernel // 2
            features = F.avg_pool2d(
                features,
                kernel_size=self.matching_pool_kernel,
                stride=1,
                padding=radius,
            )
        if self.prototype_search_region == "intersection":
            search_tokens = supervision_tokens
        elif self.prototype_search_region == "prostate":
            search_tokens = search_prostate_tokens
        else:
            search_tokens = torch.ones_like(prostate_tokens)
        return features, search_tokens, supervision_tokens

    def classification_outputs(
        self,
        activations: torch.Tensor,
        clinical_prompts: torch.Tensor | None = None,
        image_only_activations: torch.Tensor | None = None,
        prompt_class_logits: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        conditioned_logits = self.prototype_classifier(activations)
        image_logits = self.prototype_classifier(
            activations if image_only_activations is None else image_only_activations
        )
        outputs = {
            "prototype_class_logits_image": image_logits,
            "prototype_class_logits_conditioned": conditioned_logits,
        }
        if self.prompt_conditioner is not None:
            if prompt_class_logits is None:
                raise ValueError("Prompt-conditioned classification needs prompt logits.")
            combined_logits = conditioned_logits + prompt_class_logits
            outputs["prompt_class_logits"] = prompt_class_logits
        elif self.clinical_head is None:
            combined_logits = conditioned_logits
        else:
            if self.classification_target == "three_way":
                raise ValueError(
                    "The legacy additive clinical head is not defined for "
                    "three-way classification; use prompt_conditioning."
                )
            if clinical_prompts is None:
                raise ValueError("Clinical prompts are enabled but were not supplied.")
            clinical_logit, contributions = self.clinical_head(clinical_prompts)
            combined_logit = (
                conditioned_logits[:, 1] - conditioned_logits[:, 0] + clinical_logit
            )
            combined_logits = torch.stack(
                (torch.zeros_like(combined_logit), combined_logit), dim=1
            )
            outputs.update(
                {
                    "clinical_logit": clinical_logit,
                    "clinical_contributions": contributions,
                }
            )
        outputs["prototype_class_logits"] = combined_logits
        if self.classification_target == "three_way":
            outputs.update(
                {
                    "pca_class_logits": torch.stack(
                        (
                            combined_logits[:, 0],
                            torch.logsumexp(combined_logits[:, 1:], dim=1),
                        ),
                        dim=1,
                    ),
                    "cspca_class_logits": torch.stack(
                        (
                            torch.logsumexp(combined_logits[:, :2], dim=1),
                            combined_logits[:, 2],
                        ),
                        dim=1,
                    ),
                }
            )
        elif self.classification_target == "isup":
            outputs.update(self._isup_endpoints(combined_logits))
        return outputs

    @staticmethod
    def _isup_endpoints(logits: torch.Tensor) -> dict[str, torch.Tensor]:
        """Collapse GG0--GG5 logits into coherent PCA and csPCa endpoints."""
        return {
            "pca_class_logits": torch.stack(
                (logits[:, 0], torch.logsumexp(logits[:, 1:], dim=1)), dim=1
            ),
            "cspca_class_logits": torch.stack(
                (
                    torch.logsumexp(logits[:, :3], dim=1),
                    torch.logsumexp(logits[:, 3:], dim=1),
                ),
                dim=1,
            ),
        }

    @staticmethod
    def _three_way_endpoints(logits: torch.Tensor) -> dict[str, torch.Tensor]:
        return {
            "pca_class_logits": torch.stack(
                (logits[:, 0], torch.logsumexp(logits[:, 1:], dim=1)), dim=1
            ),
            "cspca_class_logits": torch.stack(
                (torch.logsumexp(logits[:, :2], dim=1), logits[:, 2]), dim=1
            ),
        }

    def fuse_hyperbolic_logits(
        self, base_logits: torch.Tensor, hyperbolic_logits: torch.Tensor
    ) -> torch.Tensor:
        if self.hyperbolic_needle_encoder is None:
            return base_logits
        if self.branch_fusion_mode == "normalized_convex":
            def centered_rms(logits: torch.Tensor) -> torch.Tensor:
                centered_logits = logits - logits.mean(dim=1, keepdim=True)
                rms = centered_logits.square().mean(dim=1, keepdim=True).sqrt()
                return centered_logits / rms.clamp_min(1e-6)

            local = centered_rms(base_logits.float()).to(base_logits.dtype)
            global_logits = centered_rms(hyperbolic_logits.float()).to(
                hyperbolic_logits.dtype
            )
            global_weight = self.branch_fusion_global_weight.to(local.dtype)
            return (1.0 - global_weight) * local + global_weight * global_logits
        # Remove the arbitrary common offset of distance logits before fusion.
        centered = hyperbolic_logits - hyperbolic_logits.mean(dim=1, keepdim=True)
        return (
            self.hyperbolic_base_logit_scale * base_logits
            + self.hyperbolic_needle_encoder.fusion_scale * centered
        )

    @property
    def branch_fusion_global_weight(self) -> torch.Tensor:
        if self.branch_fusion_logit is None:
            if self.hyperbolic_needle_encoder is None:
                raise RuntimeError("The whole-needle branch is disabled.")
            # Legacy fusion is not convex; expose its residual scale only for
            # backward-compatible diagnostics.
            return self.hyperbolic_needle_encoder.fusion_scale
        return self.branch_fusion_global_min + (
            self.branch_fusion_global_max - self.branch_fusion_global_min
        ) * torch.sigmoid(self.branch_fusion_logit)

    @property
    def semantic_anchor_fusion_scale(self) -> torch.Tensor:
        if self.semantic_anchor_fusion_logit is None:
            raise RuntimeError("The direct semantic-anchor head is disabled.")
        return self.semantic_anchor_fusion_max_scale * torch.sigmoid(
            self.semantic_anchor_fusion_logit
        )

    def semantic_anchor_class_logits(
        self, embeddings: torch.Tensor
    ) -> torch.Tensor:
        """Count-normalized three-state logits against fixed pathology anchors."""
        if not self.semantic_anchor_direct_head:
            raise RuntimeError("The direct semantic-anchor head is disabled.")
        similarities = (
            F.normalize(embeddings.float(), dim=1)
            @ self.semantic_anchor_bank.T
            / self.semantic_anchor_temperature()
        )
        logits = []
        for state in range(3):
            keep = self.semantic_anchor_states == state
            logits.append(
                torch.logsumexp(similarities[:, keep], dim=1)
                - torch.log(keep.sum().to(similarities.dtype))
            )
        return torch.stack(logits, dim=1)

    def forward(
        self,
        image: torch.Tensor,
        prostate_mask: torch.Tensor,
        needle_mask: torch.Tensor,
        clinical_prompts: torch.Tensor | None = None,
        return_distances: bool = False,
    ) -> dict[str, torch.Tensor]:
        image_features, search_tokens, supervision_tokens = self.extract_prototype_features(
            image, prostate_mask, needle_mask
        )
        prompt_outputs: dict[str, torch.Tensor] = {}
        if self.prompt_conditioner is not None:
            if clinical_prompts is None:
                raise ValueError("Prompt conditioning is enabled but prompts are missing.")
            features, prompt_outputs = self.prompt_conditioner(
                image_features, clinical_prompts
            )
        else:
            features = image_features
        grid_size = features.shape[-2:]

        def match_feature_grid(feature_grid: torch.Tensor) -> dict[str, torch.Tensor]:
            prototype_search_tokens = search_tokens[:, None, :].expand(
                -1, self.num_prototypes, -1
            ).clone()
            prototype_anchor_tokens = prototype_search_tokens.clone()
            if bool(self.needle_constrained_prototypes.any()):
                supervision_grid = supervision_tokens.reshape(
                    len(feature_grid), 1, *grid_size
                ).float()
                radius = self.needle_anchor_dilation
                if radius > 0:
                    anchor_tokens = (
                        F.max_pool2d(
                            supervision_grid,
                            kernel_size=2 * radius + 1,
                            stride=1,
                            padding=radius,
                        )
                        > 0.5
                    ).flatten(1)
                else:
                    anchor_tokens = supervision_tokens
                anchor_tokens = anchor_tokens & search_tokens

                # A thin needle can miss every strict majority-prostate token
                # after augmentation. Expand only as far as needed to reach a
                # valid prostate token; never reintroduce outside-prostate cells.
                empty_anchor = ~anchor_tokens.any(1)
                for expanded_radius in (
                    radius + 1,
                    radius + 2,
                    max(grid_size),
                ):
                    if not bool(empty_anchor.any()):
                        break
                    candidate = (
                        F.max_pool2d(
                            supervision_grid,
                            kernel_size=2 * expanded_radius + 1,
                            stride=1,
                            padding=expanded_radius,
                        )
                        > 0.5
                    ).flatten(1) & search_tokens
                    anchor_tokens[empty_anchor] = candidate[empty_anchor]
                    empty_anchor = ~anchor_tokens.any(1)
                if bool(empty_anchor.any()):
                    anchor_tokens[empty_anchor] = search_tokens[empty_anchor]
                prototype_anchor_tokens[:, self.needle_constrained_prototypes] = (
                    anchor_tokens[:, None, :]
                )
            if self.occurrence_matcher is None:
                return self.prototype_layer(
                    feature_grid,
                    prototype_search_tokens,
                    prototype_anchor_tokens,
                )
            # Occurrence maps have no ordered anchor; retain the stricter
            # all-slots constraint for this controlled matcher ablation.
            prototype_search_tokens = prototype_anchor_tokens
            if self.num_uncertainty_prototypes:
                prototype_search_tokens = torch.cat(
                    (
                        prototype_search_tokens,
                        search_tokens[:, None, :].expand(
                            -1, self.num_uncertainty_prototypes, -1
                        ),
                    ),
                    dim=1,
                )
            return self.occurrence_matcher(
                feature_grid,
                self.prototype_layer.prototype_vectors,
                self.prototype_layer.slot_weights(),
                prototype_search_tokens,
            )

        prototype_outputs = match_feature_grid(features)
        image_only_prototype_outputs = (
            match_feature_grid(image_features)
            if self.prompt_conditioner is not None
            else prototype_outputs
        )
        token_maps = prototype_outputs["prototype_token_maps"]
        batch, _, num_tokens = token_maps.shape
        cancer_evidence = self.evidence_layer(
            token_maps.transpose(1, 2)
        ).transpose(1, 2).view(batch, 1, *grid_size)
        cancer_evidence = F.interpolate(
            cancer_evidence,
            size=prostate_mask.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
        # Weak involvement supervision remains needle ∩ prostate while the
        # classifier can discover context throughout the prostate. Dense
        # evidence itself is never allowed outside the exact pixel mask.
        cancer_evidence = cancer_evidence.masked_fill(
            ~(prostate_mask > 0.5), -12.0
        )
        activations = prototype_outputs["prototype_activations"]
        classification = self.classification_outputs(
            activations,
            clinical_prompts,
            image_only_prototype_outputs["prototype_activations"],
            prompt_outputs.get("prompt_class_logits"),
        )
        hyperbolic_outputs: dict[str, torch.Tensor] = {}
        if self.hyperbolic_needle_encoder is not None:
            hyperbolic_outputs = self.hyperbolic_needle_encoder(
                features, supervision_tokens
            )
            image_hyperbolic_outputs = (
                self.hyperbolic_needle_encoder(image_features, supervision_tokens)
                if self.prompt_conditioner is not None
                else hyperbolic_outputs
            )
            classification["prototype_class_logits_pre_hyperbolic"] = classification[
                "prototype_class_logits"
            ]
            # Preserve auditable local-prototype heads before any global,
            # Prost-RL, or pathology residual is applied.  The conditioned
            # version contains prompt FiLM/spatial-gate effects in the matched
            # image features but deliberately excludes the additive
            # prompt-only classifier, so its performance is attributable to
            # prompt-conditioned prototype evidence.
            classification[
                "occurrence_conditioned_logits_pre_hyperbolic"
            ] = classification["prototype_class_logits_conditioned"]
            classification[
                "occurrence_image_logits_pre_hyperbolic"
            ] = classification["prototype_class_logits_image"]
            hyperbolic_class_logits = (
                hyperbolic_outputs["hyperbolic_grade_logits"]
                if self.classification_target == "isup"
                else hyperbolic_outputs["hyperbolic_three_way_logits"]
            )
            image_hyperbolic_class_logits = (
                image_hyperbolic_outputs["hyperbolic_grade_logits"]
                if self.classification_target == "isup"
                else image_hyperbolic_outputs["hyperbolic_three_way_logits"]
            )
            classification["prototype_class_logits"] = self.fuse_hyperbolic_logits(
                classification["prototype_class_logits"],
                hyperbolic_class_logits,
            )
            classification["prototype_class_logits_image"] = self.fuse_hyperbolic_logits(
                classification["prototype_class_logits_image"],
                image_hyperbolic_class_logits,
            )
            if "prototype_class_logits_conditioned" in classification:
                classification["prototype_class_logits_conditioned"] = (
                    self.fuse_hyperbolic_logits(
                        classification["prototype_class_logits_conditioned"],
                        hyperbolic_class_logits,
                    )
                )
            classification.update(
                self._isup_endpoints(classification["prototype_class_logits"])
                if self.classification_target == "isup"
                else self._three_way_endpoints(
                    classification["prototype_class_logits"]
                )
            )
            hyperbolic_outputs["hyperbolic_three_way_logits_image"] = (
                image_hyperbolic_outputs["hyperbolic_three_way_logits"]
            )
            hyperbolic_outputs["hyperbolic_grade_logits_image"] = (
                image_hyperbolic_outputs["hyperbolic_grade_logits"]
            )
            hyperbolic_outputs["hyperbolic_base_logit_scale"] = (
                hyperbolic_outputs["hyperbolic_fusion_scale"].new_tensor(
                    self.hyperbolic_base_logit_scale
                )
            )
            hyperbolic_outputs["branch_fusion_global_weight"] = (
                self.branch_fusion_global_weight
            )
            transport_logits = hyperbolic_outputs.get(
                "pathology_transport_three_way_logits"
            )
            if transport_logits is not None:
                image_transport_logits = image_hyperbolic_outputs[
                    "pathology_transport_three_way_logits"
                ]
                transport_scale = hyperbolic_outputs[
                    "pathology_transport_fusion_scale"
                ]
                classification["prototype_class_logits_pre_pathology_transport"] = (
                    classification["prototype_class_logits"]
                )
                classification["prototype_class_logits"] = (
                    classification["prototype_class_logits"]
                    + transport_scale
                    * (transport_logits - transport_logits.mean(1, keepdim=True))
                )
                classification["prototype_class_logits_image"] = (
                    classification["prototype_class_logits_image"]
                    + transport_scale
                    * (
                        image_transport_logits
                        - image_transport_logits.mean(1, keepdim=True)
                    )
                )
                if "prototype_class_logits_conditioned" in classification:
                    classification["prototype_class_logits_conditioned"] = (
                        classification["prototype_class_logits_conditioned"]
                        + transport_scale
                        * (transport_logits - transport_logits.mean(1, keepdim=True))
                    )
                classification.update(
                    self._three_way_endpoints(
                        classification["prototype_class_logits"]
                    )
                )
                hyperbolic_outputs[
                    "pathology_transport_three_way_logits_image"
                ] = image_transport_logits
            if self.dense_endpoint_head is not None:
                prototype_residual = classification["prototype_class_logits"]
                image_prototype_residual = classification[
                    "prototype_class_logits_image"
                ]
                dense_outputs = self.dense_endpoint_head(
                    hyperbolic_outputs["needle_axis_embedding"],
                    hyperbolic_outputs["hyperbolic_predicted_involvement"],
                    prototype_residual,
                )
                image_dense_outputs = self.dense_endpoint_head(
                    image_hyperbolic_outputs["needle_axis_embedding"],
                    image_hyperbolic_outputs["hyperbolic_predicted_involvement"],
                    image_prototype_residual,
                )
                classification["prototype_class_logits_pre_dense_endpoint"] = (
                    prototype_residual
                )
                classification["prototype_class_logits"] = dense_outputs[
                    "dense_endpoint_final_logits"
                ]
                classification["prototype_class_logits_image"] = (
                    image_dense_outputs["dense_endpoint_final_logits"]
                )
                if "prototype_class_logits_conditioned" in classification:
                    classification["prototype_class_logits_conditioned"] = (
                        dense_outputs["dense_endpoint_final_logits"]
                    )
                classification.update(
                    self._three_way_endpoints(
                        classification["prototype_class_logits"]
                    )
                )
                hyperbolic_outputs.update(dense_outputs)
                hyperbolic_outputs["dense_endpoint_three_way_logits_image"] = (
                    image_dense_outputs["dense_endpoint_three_way_logits"]
                )
                hyperbolic_outputs["dense_endpoint_final_logits_image"] = (
                    image_dense_outputs["dense_endpoint_final_logits"]
                )
            if self.prost_rl_endpoint_head is not None:
                residual_logits = classification["prototype_class_logits"]
                image_residual_logits = classification[
                    "prototype_class_logits_image"
                ]
                aligned_outputs = self.prost_rl_endpoint_head(
                    image_features,
                    clinical_prompts,
                    hyperbolic_outputs["hyperbolic_predicted_involvement"],
                    residual_logits,
                )
                aligned_image_outputs = self.prost_rl_endpoint_head(
                    image_features,
                    None,
                    image_hyperbolic_outputs[
                        "hyperbolic_predicted_involvement"
                    ],
                    image_residual_logits,
                )
                classification[
                    "prototype_class_logits_pre_prost_rl_endpoint"
                ] = residual_logits
                classification["prototype_class_logits"] = aligned_outputs[
                    "prost_rl_endpoint_final_logits"
                ]
                classification["prototype_class_logits_image"] = (
                    aligned_image_outputs["prost_rl_endpoint_final_logits"]
                )
                classification.update(
                    self._three_way_endpoints(
                        classification["prototype_class_logits"]
                    )
                )
                hyperbolic_outputs.update(aligned_outputs)
                hyperbolic_outputs[
                    "prost_rl_endpoint_three_way_logits_image"
                ] = aligned_image_outputs["prost_rl_endpoint_three_way_logits"]
                hyperbolic_outputs[
                    "prost_rl_endpoint_final_logits_image"
                ] = aligned_image_outputs["prost_rl_endpoint_final_logits"]
        semantic_outputs: dict[str, torch.Tensor] = {}
        if self.semantic_anchor_core_projector is not None:
            semantic_source = torch.cat(
                (
                    activations.float(),
                    hyperbolic_outputs["hyperbolic_direction"].float(),
                ),
                dim=1,
            )
            semantic_embedding = F.normalize(
                self.semantic_anchor_core_projector(semantic_source).float(), dim=1
            )
            semantic_outputs["semantic_anchor_embedding"] = semantic_embedding
            if self.semantic_anchor_direct_head:
                anchor_target_logits = self.semantic_anchor_class_logits(
                    semantic_embedding
                )
                semantic_outputs["semantic_anchor_target_logits"] = (
                    anchor_target_logits
                )
                anchor_logits = (
                    self.hyperbolic_needle_encoder.target_logits_to_three_way(
                        anchor_target_logits,
                        hyperbolic_outputs["hyperbolic_predicted_involvement"],
                    )
                    if self.semantic_anchor_label_mode
                    == "grade_bands_01_23_45"
                    else anchor_target_logits
                )
                semantic_outputs["semantic_anchor_class_logits"] = anchor_logits
                semantic_outputs["prototype_class_logits_pre_semantic_anchor"] = (
                    classification["prototype_class_logits"]
                )
                centered_anchor_logits = anchor_logits - anchor_logits.mean(
                    dim=1, keepdim=True
                )
                classification["prototype_class_logits"] = (
                    classification["prototype_class_logits"]
                    + self.semantic_anchor_fusion_scale * centered_anchor_logits
                )
                classification.update(
                    self._three_way_endpoints(
                        classification["prototype_class_logits"]
                    )
                )
                semantic_outputs["semantic_anchor_fusion_scale"] = (
                    self.semantic_anchor_fusion_scale
                )

        outputs = {
            "mask_logits": cancer_evidence,
            "cls_outputs": [
                classification["prototype_class_logits"]
                if self.core_classifier is None
                else self.core_classifier(activations)
            ],
            **classification,
            "encoder_features": features,
            "encoder_features_image": image_features,
            "prototype_search_token_mask": search_tokens,
            "supervision_token_mask": supervision_tokens,
            "prototype_activations_image": image_only_prototype_outputs[
                "prototype_activations"
            ],
            **prototype_outputs,
            **prompt_outputs,
            **hyperbolic_outputs,
            **semantic_outputs,
        }
        if self.pathology_geometry_projector is not None:
            geometry_source = torch.cat(
                (
                    activations.float(),
                    hyperbolic_outputs["hyperbolic_direction"].float(),
                ),
                dim=1,
            )
            outputs["pathology_geometry_embedding"] = F.normalize(
                self.pathology_geometry_projector(geometry_source).float(), dim=1
            )
        if self.classification_target == "three_way":
            outputs["cls_outputs"] = [classification["pca_class_logits"]]
            outputs["cspca_cls_outputs"] = [classification["cspca_class_logits"]]
        elif self.classification_target == "isup":
            outputs["cls_outputs"] = [classification["pca_class_logits"]]
            outputs["cspca_cls_outputs"] = [classification["cspca_class_logits"]]
        if self.grade_classifier is not None:
            outputs["grade_group_logits"] = self.grade_classifier(activations)
            outputs["grade_group_probabilities"] = self.hierarchical_probabilities(
                outputs["prototype_class_logits"], outputs["grade_group_logits"]
            )
        if self.uncertainty_head is not None:
            uncertainty_activations = prototype_outputs["uncertainty_activations"]
            outputs["uncertainty_logit"] = self.uncertainty_head(
                uncertainty_activations
            ).squeeze(-1)
        if not return_distances:
            outputs.pop("subprototype_similarities")
        return outputs

    def center_adversarial_logits(
        self,
        occurrence_activations: torch.Tensor,
        hyperbolic_direction: torch.Tensor,
        states: torch.Tensor,
        reversal_scale: float,
    ) -> torch.Tensor:
        """Predict acquisition center conditionally within pathology state."""
        if self.center_discriminator is None:
            raise RuntimeError("Center-adversarial discriminator is disabled.")
        embedding = torch.cat(
            (occurrence_activations.float(), hyperbolic_direction.float()), dim=1
        )
        reversed_embedding = _GradientReversal.apply(embedding, reversal_scale)
        all_logits = self.center_discriminator(reversed_embedding).reshape(
            len(embedding),
            self.center_adversarial_num_states,
            self.center_adversarial_num_centers,
        )
        state_indices = states.long().clamp(
            min=0, max=self.center_adversarial_num_states - 1
        )
        return all_logits[
            torch.arange(len(embedding), device=embedding.device), state_indices
        ]

    @staticmethod
    def hierarchical_probabilities(
        binary_logits: torch.Tensor, conditional_logits: torch.Tensor
    ) -> torch.Tensor:
        """Return probabilities for observed labels [benign, GG2, GG3, GG4, GG5]."""
        cancer = torch.softmax(binary_logits, dim=-1)[:, 1]
        survival = torch.sigmoid(conditional_logits)
        # Enforce P(GG>=3) >= P(GG>=4) >= P(GG>=5) at inference.
        survival = torch.cummin(survival, dim=1).values
        conditional = torch.stack(
            (
                1.0 - survival[:, 0],
                survival[:, 0] - survival[:, 1],
                survival[:, 1] - survival[:, 2],
                survival[:, 2],
            ),
            dim=1,
        ).clamp_min(0.0)
        conditional = conditional / conditional.sum(1, keepdim=True).clamp_min(1e-8)
        return torch.cat(((1.0 - cancer)[:, None], cancer[:, None] * conditional), dim=1)

    def prototype_losses(
        self,
        labels: torch.Tensor,
        matched_values: torch.Tensor,
        involvement: torch.Tensor | None = None,
        grade_groups: torch.Tensor | None = None,
        occurrence_maps: torch.Tensor | None = None,
        involvement_valid: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        losses = self.prototype_layer.regularization_losses(
            labels,
            matched_values,
            involvement,
            grade_groups,
            involvement_valid,
        )
        wrong_class_mask = 1.0 - self.prototype_class_identity.T
        losses["l1"] = (
            self.prototype_classifier.weight * wrong_class_mask
        ).abs().sum()
        losses["grade_l1"] = (
            self.grade_classifier.weight.abs().sum()
            if self.grade_classifier is not None
            else matched_values.new_zeros(())
        )
        if occurrence_maps is not None:
            # Directly target v4B's K=4 sub-prototype spatial collapse.
            normalized = F.normalize(occurrence_maps.float(), p=2, dim=-1)
            overlap = torch.einsum("bpkn,bpln->bpkl", normalized, normalized)
            k = occurrence_maps.shape[2]
            off_diagonal = 1.0 - torch.eye(
                k, device=overlap.device, dtype=overlap.dtype
            )
            slots = self.prototype_layer.slot_weights().squeeze(0)
            active_pairs = (
                slots[None, :, :, None]
                * slots[None, :, None, :]
                * off_diagonal[None, None]
            )
            losses["occurrence_overlap"] = (
                overlap * active_pairs
            ).sum() / active_pairs.expand_as(overlap).sum().clamp_min(1.0)
            probabilities = occurrence_maps.float().clamp_min(1e-8)
            entropy = -(
                probabilities * probabilities.log()
            ).sum(-1)
            losses["occurrence_entropy"] = entropy.mean()
            losses["occurrence_effective_area"] = entropy.exp().mean()

            target_tokens = self.occurrence_region_target_tokens
            if target_tokens > 0:
                num_tokens = probabilities.shape[-1]
                side = int(round(num_tokens**0.5))
                if side * side != num_tokens:
                    raise ValueError(
                        "Anatomical occurrence-region losses require a square "
                        f"token grid, got {num_tokens} tokens."
                    )
                target = probabilities.new_tensor(target_tokens).clamp(
                    min=1.0, max=float(num_tokens)
                )
                # V15 normalized by log(num_tokens), leaving a requested
                # 9-token map at roughly 104 tokens. Log-area is already
                # scale stable, so enforce the residual directly.
                losses["occurrence_region_area"] = (
                    entropy - target.log()
                ).square().mean()

                axis = torch.linspace(
                    -1.0, 1.0, side, device=probabilities.device
                )
                yy, xx = torch.meshgrid(axis, axis, indexing="ij")
                coordinates = torch.stack((xx.flatten(), yy.flatten()), dim=1)
                centroids = torch.einsum(
                    "bpkn,nd->bpkd", probabilities, coordinates
                )
                displacement = (
                    coordinates[None, None, None] - centroids[..., None, :]
                )
                squared_radius = displacement.square().sum(-1)
                losses["occurrence_region_compactness"] = (
                    probabilities * squared_radius
                ).sum(-1).mean()

                centroid_distance = (
                    centroids[:, :, :, None] - centroids[:, :, None, :]
                ).square().sum(-1)
                centroid_pairs = active_pairs.expand_as(centroid_distance)
                losses["occurrence_region_cohesion"] = (
                    centroid_distance * centroid_pairs
                ).sum() / centroid_pairs.sum().clamp_min(1.0)

                spatial = probabilities.reshape(
                    *probabilities.shape[:-1], side, side
                )
                horizontal = (spatial[..., :, 1:] - spatial[..., :, :-1]).abs()
                vertical = (spatial[..., 1:, :] - spatial[..., :-1, :]).abs()
                losses["occurrence_region_tv"] = 0.5 * (
                    horizontal.mean() + vertical.mean()
                )
            else:
                zero = matched_values.new_zeros(())
                losses["occurrence_region_area"] = zero
                losses["occurrence_region_compactness"] = zero
                losses["occurrence_region_cohesion"] = zero
                losses["occurrence_region_tv"] = zero
        else:
            zero = matched_values.new_zeros(())
            losses["occurrence_overlap"] = zero
            losses["occurrence_entropy"] = zero
            losses["occurrence_effective_area"] = zero
            losses["occurrence_region_area"] = zero
            losses["occurrence_region_compactness"] = zero
            losses["occurrence_region_cohesion"] = zero
            losses["occurrence_region_tv"] = zero
        return losses

    def semantic_anchor_temperature(self) -> torch.Tensor:
        if self.semantic_anchor_log_temperature is None:
            raise RuntimeError("Semantic-anchor supervision is disabled.")
        return self.semantic_anchor_log_temperature.exp().clamp(
            min=self.semantic_anchor_temperature_min,
            max=self.semantic_anchor_temperature_max,
        )

    def semantic_anchor_prototype_embeddings(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Project the configured ultrasound prototypes into anchor space."""
        if self.semantic_anchor_prototype_projector is None:
            raise RuntimeError("Semantic-anchor supervision is disabled.")
        if self.semantic_anchor_label_mode == "grade_bands_01_23_45":
            if self.hyperbolic_needle_encoder is None:
                raise RuntimeError("Ordinal-band prototype projector needs geometry.")
            vectors = F.normalize(
                self.hyperbolic_needle_encoder.prototype_directions.float(), dim=-1
            )
            flat_vectors = vectors.reshape(-1, vectors.shape[-1])
            states = torch.arange(
                len(self.hyperbolic_needle_encoder.target_values),
                device=flat_vectors.device,
            ).repeat_interleave(
                self.hyperbolic_needle_encoder.prototypes_per_grade
            )
        else:
            vectors = self.prototype_layer.prototype_vectors.permute(0, 2, 1)
            flat_vectors = vectors.reshape(-1, vectors.shape[-1])
            prototype_states = self.prototype_layer.prototype_class_identity.argmax(1)
            states = prototype_states.repeat_interleave(
                self.prototype_layer.num_subprototypes
            )
        embeddings = F.normalize(
            self.semantic_anchor_prototype_projector(flat_vectors).float(), dim=1
        )
        return embeddings, states

    def get_params_groups(self) -> dict[str, list[nn.Parameter]]:
        core_parameters = (
            list(self.core_classifier.parameters())
            if self.core_classifier is not None
            else []
        )
        last_layers = (
            list(self.evidence_layer.parameters())
            + list(self.prototype_classifier.parameters())
            + core_parameters
        )
        classification_layers = (
            list(self.prototype_classifier.parameters()) + core_parameters
        )
        if self.grade_classifier is not None:
            last_layers += list(self.grade_classifier.parameters())
            classification_layers += list(self.grade_classifier.parameters())
        if self.occurrence_matcher is not None:
            last_layers += list(self.occurrence_matcher.occurrence_net.parameters())
            if isinstance(self.occurrence_matcher.logit_scale, nn.Parameter):
                last_layers += [self.occurrence_matcher.logit_scale]
        if self.clinical_head is not None:
            last_layers += list(self.clinical_head.parameters())
            classification_layers += list(self.clinical_head.parameters())
        if self.prompt_conditioner is not None:
            last_layers += list(self.prompt_conditioner.parameters())
            # Spatial FiLM/query parameters remain fixed in the final convex
            # stage; only the linear prompt residual is optimized there.
            classification_layers += list(
                self.prompt_conditioner.classifier.parameters()
            )
        if self.hyperbolic_needle_encoder is not None:
            last_layers += list(self.hyperbolic_needle_encoder.parameters())
            classification_layers += [self.hyperbolic_needle_encoder.fusion_logit]
            if self.hyperbolic_needle_encoder.band_endpoint_readout is not None:
                classification_layers += list(
                    self.hyperbolic_needle_encoder.band_endpoint_readout.parameters()
                )
            if self.branch_fusion_logit is not None:
                last_layers += [self.branch_fusion_logit]
                classification_layers += [self.branch_fusion_logit]
        if self.dense_endpoint_head is not None:
            last_layers += list(self.dense_endpoint_head.parameters())
            classification_layers += list(self.dense_endpoint_head.parameters())
        if self.prost_rl_endpoint_head is not None:
            last_layers += list(self.prost_rl_endpoint_head.parameters())
            classification_layers += list(self.prost_rl_endpoint_head.parameters())
        if self.pathology_geometry_projector is not None:
            last_layers += list(self.pathology_geometry_projector.parameters())
        if self.semantic_anchor_core_projector is not None:
            last_layers += list(self.semantic_anchor_core_projector.parameters())
            last_layers += list(self.semantic_anchor_prototype_projector.parameters())
            last_layers += [self.semantic_anchor_log_temperature]
            classification_layers += list(
                self.semantic_anchor_core_projector.parameters()
            )
            classification_layers += list(
                self.semantic_anchor_prototype_projector.parameters()
            )
            classification_layers += [self.semantic_anchor_log_temperature]
            if self.semantic_anchor_fusion_logit is not None:
                last_layers += [self.semantic_anchor_fusion_logit]
                classification_layers += [self.semantic_anchor_fusion_logit]
        if self.center_discriminator is not None:
            last_layers += list(self.center_discriminator.parameters())
        if self.uncertainty_head is not None:
            last_layers += list(self.uncertainty_head.parameters())
        return {
            "encoder": list(self.image_encoder.parameters()),
            "prototypes": [self.prototype_layer.prototype_vectors]
            + (
                [self.occurrence_matcher.uncertainty_vectors]
                if self.occurrence_matcher is not None
                and self.occurrence_matcher.uncertainty_vectors is not None
                else []
            ),
            "occurrence": (
                list(self.occurrence_matcher.occurrence_net.parameters())
                if self.occurrence_matcher is not None
                else []
            ),
            "slots": [self.prototype_layer.slot_logits],
            "hyperbolic": (
                list(self.hyperbolic_needle_encoder.parameters())
                if self.hyperbolic_needle_encoder is not None
                else []
            ),
            "last_layers": last_layers,
            "classification_layers": classification_layers,
        }


@register_model
def protoprostate_medsam(
    backbone: str = "medsam_adapter",
    backbone_kw: dict[str, Any] | None = None,
    **kwargs: Any,
) -> ProtoProstate:
    """Registry entry used by Prost-RL Hydra configs."""
    # Import triggers registration of the MedSAM backbone factories.
    pass

    return ProtoProstate(backbone=backbone, backbone_kw=backbone_kw, **kwargs)

