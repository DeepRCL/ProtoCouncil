import argparse
import json
import math
from typing import Callable
import torch
from torch import nn
import torch
import torch.nn as nn
from torch.nn import functional as F
from einops import repeat, rearrange

from .needle_axis import pool_logits_along_needle_axis


class CancerDetectionValidRegionLoss(nn.Module):
    def __init__(
        self,
        base_loss: Callable = F.binary_cross_entropy_with_logits,
        prostate_mask: bool = True,
        needle_mask: bool = True,
    ):
        super().__init__()
        self.base_loss = base_loss
        self.prostate_mask = prostate_mask
        self.needle_mask = needle_mask



class ProportionBCE(nn.Module):
    def __init__(
        self,
        l1_penalty_lambda: float | None = None,
        entropy_penalty_lambda: float | None = None,
        pos_weight: float = 1.0,
    ):
        super().__init__()
        self.l1_penalty_lambda = l1_penalty_lambda
        self.entropy_penalty_lambda = entropy_penalty_lambda
        self.pos_weight = pos_weight



# =============================================================================
# NOISE-ROBUST LOSS FUNCTIONS
# =============================================================================

class TopKMILLoss(nn.Module):
    """
    Top-K Involvement Loss (Multiple Instance Learning approach)
    
    Instead of forcing ALL needle pixels to match the label (which forces benign 
    pixels to be predicted as cancer), we only force the Top k% of pixels to be 
    predicted as cancer, where k is the involvement percentage.
    
    This is a form of MIL: if the core has 55% cancer, we assume the "most 
    suspicious" 55% of pixels are the cancer, and the rest are likely benign.
    
    Why it's better: It solves the "blur" problem of ProstNFound+ by allowing 
    the model to predict 0 for the benign parts of the needle trace.
    
    Args:
        pos_weight: Weight for positive (cancer) cases
        min_top_k_fraction: Minimum fraction of pixels to select (default: 0.05)
                           Prevents selecting too few pixels for very low involvement
        use_soft_topk: If True, use differentiable soft top-k. If False, use hard selection.
        benign_weight: Weight for benign pixel loss (default: 1.0, previously 0.5 was too weak)
    """
    
    def __init__(
        self,
        pos_weight: float = 1.0,
        min_top_k_fraction: float = 0.05,
        use_soft_topk: bool = False,
        temperature: float = 0.1,
        benign_weight: float = 1.0,  # INCREASED from 0.5 - need strong push to 0
    ):
        super().__init__()
        self.pos_weight = pos_weight
        self.min_top_k_fraction = min_top_k_fraction
        self.use_soft_topk = use_soft_topk
        self.temperature = temperature
        self.benign_weight = benign_weight
    


class ThresholdedInvolvementLoss(nn.Module):
    """
    Thresholded Involvement Loss
    
    Instead of comparing mean probability to involvement, we threshold the 
    predictions at 0.5 and compare the fraction of "on" pixels to involvement.
    
    This makes the loss more directly optimize for the calibration metric:
    - If involvement = 50%, we want 50% of pixels to have prob > 0.5
    
    The loss encourages:
    - For 50% involvement: ~50% of needle pixels should exceed 0.5 threshold
    - Sharp predictions (either high or low, not middle values like 0.55)
    
    Args:
        threshold: Probability threshold for considering a pixel as "on"
        use_soft_threshold: If True, use sigmoid approximation of step function
        soft_temperature: Temperature for soft threshold (lower = sharper)
        pos_weight: Weight for positive (cancer) cases
    """
    
    def __init__(
        self,
        threshold: float = 0.5,
        use_soft_threshold: bool = True,
        soft_temperature: float = 0.1,
        pos_weight: float = 1.0,
        entropy_penalty_lambda: float = None,
        tolerance_margin: float = 0.0,
        benign_tolerance_margin: float = 0.0,
        soft_mean_weight: float = 0.5,
        benign_bce_weight: float = 0.0,
    ):
        super().__init__()
        self.threshold = threshold
        self.use_soft_threshold = use_soft_threshold
        self.soft_temperature = soft_temperature
        self.pos_weight = pos_weight
        self.entropy_penalty_lambda = entropy_penalty_lambda
        self.tolerance_margin = float(tolerance_margin)
        self.benign_tolerance_margin = float(benign_tolerance_margin)
        self.soft_mean_weight = float(soft_mean_weight)
        self.benign_bce_weight = float(benign_bce_weight)
    
    def forward(self, bag_of_logits, true_prop):
        """
        Args:
            bag_of_logits: Tensor of shape (N,) or (N, 1) containing logits for needle pixels
            true_prop: Scalar involvement proportion (0 to 1)
        """
        eps = 1e-7
        
        # Flatten logits
        logits = bag_of_logits.view(-1)
        n_pixels = logits.shape[0]
        
        if n_pixels == 0:
            return torch.tensor(0.0, device=bag_of_logits.device)
        
        # Compute probabilities
        probs = torch.sigmoid(logits)
        
        if self.use_soft_threshold:
            # Soft thresholding using sigmoid
            # centered_probs = (probs - threshold) / temperature
            # soft_threshold(probs) = sigmoid((probs - threshold) / temperature)
            centered = (probs - self.threshold) / self.soft_temperature
            thresholded = torch.sigmoid(centered)
        else:
            # Hard thresholding (not differentiable, use with caution)
            thresholded = (probs > self.threshold).float()
        
        # Predicted proportion of "on" pixels
        pred_prop = thresholded.mean().clamp(min=eps, max=1 - eps)
        true_prop_clamped = true_prop.clamp(min=eps, max=1 - eps)
        
        if self.tolerance_margin > 0 or self.benign_tolerance_margin > 0:
            # Pathology involvement is an approximate, noisy length fraction.
            # Optimize an interval around q instead of treating q as an exact
            # pixel count. The thresholded term rejects the uniform p=q
            # solution; the soft-mean term stabilizes early gradients.
            margin = (
                self.benign_tolerance_margin
                if float(true_prop) == 0.0
                else self.tolerance_margin
            )

            def interval_violation(value):
                return F.relu((value - true_prop).abs() - margin)

            loss = interval_violation(pred_prop)
            loss = loss + self.soft_mean_weight * interval_violation(probs.mean())
            if float(true_prop) == 0.0 and self.benign_bce_weight > 0:
                # Benign pathology is substantially cleaner than a positive
                # percentage estimate, so retain a direct negative-instance
                # term for benign cores.
                loss = loss + self.benign_bce_weight * F.binary_cross_entropy_with_logits(
                    logits, torch.zeros_like(logits)
                )
        else:
            # Legacy Prost-RL objective retained exactly when no margin is set.
            loss = -true_prop_clamped * torch.log(pred_prop) - (1 - true_prop_clamped) * torch.log(1 - pred_prop)
            standard_bce = -true_prop * torch.log(probs.mean().clamp(min=eps, max=1 - eps)) \
                           - (1 - true_prop) * torch.log(1 - probs.mean().clamp(min=eps, max=1 - eps))
            loss = 0.5 * loss + 0.5 * standard_bce
        
        # Apply positive weight
        if true_prop.item() > 0:
            loss = loss * self.pos_weight
        
        # Entropy penalty to encourage sharp predictions
        if self.entropy_penalty_lambda:
            probs_clamped = probs.clamp(min=eps, max=1 - eps)
            entropy = -probs_clamped * torch.log(probs_clamped) - (1 - probs_clamped) * torch.log(1 - probs_clamped)
            entropy = torch.nan_to_num(entropy, nan=0.0, posinf=0.0, neginf=0.0)
            loss = loss + self.entropy_penalty_lambda * entropy.mean()
        
        return loss


class SymmetricCrossEntropyLoss(nn.Module):
    """
    Symmetric Cross Entropy (SCE) for Noise-Robust Learning
    
    From "Symmetric Cross Entropy for Robust Learning with Noisy Labels" (ICCV 2019)
    and used in Manifold DivideMix paper.
    
    Standard CE: Penalizes model when it predicts "Benign" on a "Cancer" pixel 
                 (Hard punishment for missing the noisy label)
    
    Reverse CE: Penalizes the label when the model is confident 
                (Allows model to ignore the label if it strongly disagrees)
    
    SCE = α * CE(p, q) + β * RCE(p, q)
    
    Where:
    - CE = -q * log(p)  (standard cross entropy, q is label, p is prediction)
    - RCE = -p * log(q) (reverse cross entropy)
    
    For proportion-based learning:
    - CE pushes predicted proportion toward true involvement
    - RCE allows predicted proportion to "override" noisy involvement if confident
    
    Args:
        alpha: Weight for standard cross entropy (default: 1.0)
        beta: Weight for reverse cross entropy (default: 1.0)
        pos_weight: Weight for positive (cancer) cases
        epsilon: Small value for numerical stability in RCE
    """
    
    def __init__(
        self,
        alpha: float = 1.0,
        beta: float = 1.0,
        pos_weight: float = 1.0,
        epsilon: float = 1e-4,
        entropy_penalty_lambda: float = None,
    ):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.pos_weight = pos_weight
        self.epsilon = epsilon  # Larger epsilon for RCE stability
        self.entropy_penalty_lambda = entropy_penalty_lambda
    


class TopKInvolvementMILLoss(nn.Module):
    """
    Wrapper that applies TopKMILLoss within the CancerDetectionMILLoss framework.
    
    This is a combination of:
    1. Top-K selection: Only supervise top k% pixels as cancer
    2. MIL framework: Process needle mask and handle batches properly
    
    Args:
        benign_weight: Weight for benign pixel loss (default: 1.0)
                       Higher values push non-topk pixels more strongly to 0.
    """
    
    def __init__(
        self,
        pos_weight: float = 1.0,
        min_top_k_fraction: float = 0.05,
        use_soft_topk: bool = False,
        temperature: float = 0.1,
        treat_gg1_as_benign: bool = False,
        benign_weight: float = 1.0,  # Controls how strongly non-topk pixels are pushed to 0
    ):
        super().__init__()
        self.base_loss = TopKMILLoss(
            pos_weight=pos_weight,
            min_top_k_fraction=min_top_k_fraction,
            use_soft_topk=use_soft_topk,
            temperature=temperature,
            benign_weight=benign_weight,
        )
        self.treat_gg1_as_benign = treat_gg1_as_benign



class CancerDetectionMILLoss(nn.Module):
    def __init__(self, base_loss=ProportionBCE(), treat_gg1_as_benign=False):

        super().__init__()
        self.base_loss = base_loss
        self.treat_gg1_as_benign = treat_gg1_as_benign



class NeedleAxisCancerDetectionMILLoss(nn.Module):
    """Apply proportion supervision to longitudinal needle bins.

    The ordinary Prost-RL MIL loss treats every valid 2-D pixel as an instance.
    This wrapper instead pools across the needle width and treats ordered bins
    along the needle as instances, matching the length-based pathology label.
    """

    def __init__(
        self,
        base_loss,
        num_bins: int = 32,
        contiguity_weight: float = 0.0,
        treat_gg1_as_benign: bool = False,
    ) -> None:
        super().__init__()
        self.base_loss = base_loss
        self.num_bins = int(num_bins)
        self.contiguity_weight = float(contiguity_weight)
        self.treat_gg1_as_benign = bool(treat_gg1_as_benign)

    def forward(self, data):
        cancer_logits = data["cancer_logits"]
        prostate_mask = data["prostate_mask"].to(cancer_logits.device)
        needle_mask = data["needle_mask"].to(cancer_logits.device)
        involvement = data["involvement"].to(cancer_logits.device).clone()
        involvement_valid = data.get("involvement_valid")
        involvement_valid = (
            involvement_valid.bool().to(cancer_logits.device)
            if involvement_valid is not None
            else torch.ones_like(involvement, dtype=torch.bool)
        )
        grade_group = data["grade_group"].to(cancer_logits.device)
        if self.treat_gg1_as_benign:
            involvement[grade_group == 1] = 0.0

        loss = cancer_logits.new_zeros(())
        for sample_idx in range(len(cancer_logits)):
            if not bool(involvement_valid[sample_idx]):
                continue
            valid = (prostate_mask[sample_idx] > 0.5) & (
                needle_mask[sample_idx] > 0.5
            )
            axis_logits = pool_logits_along_needle_axis(
                cancer_logits[sample_idx], valid, self.num_bins
            )
            if axis_logits.numel() == 0:
                continue
            core_loss = self.base_loss(axis_logits, involvement[sample_idx])
            if self.contiguity_weight > 0 and axis_logits.numel() > 1:
                axis_probabilities = axis_logits.sigmoid()
                core_loss = core_loss + self.contiguity_weight * (
                    axis_probabilities[1:] - axis_probabilities[:-1]
                ).abs().mean()
            loss = loss + core_loss
        # Match CancerDetectionMILLoss/TopKInvolvementMILLoss: their per-core
        # losses are summed, so the configured heatmap weight keeps the same
        # meaning when switching from 2-D pixels to longitudinal needle bins.
        return loss


class InvolvementL1Loss(nn.Module):
    def __init__(self, prostate_penalty=True, pos_weight=1):
        super().__init__()
        self.prostate_penalty = prostate_penalty
        self.pos_weight = pos_weight



class InvolvementMSELoss(nn.Module):
    def __call__(self, data):
        avg_needle_heatmap_value = data["average_needle_heatmap_value"]
        B = len(avg_needle_heatmap_value)
        device = avg_needle_heatmap_value.device
        involvement = data["involvement"].to(device)

        loss = torch.nn.functional.mse_loss(avg_needle_heatmap_value, involvement)
        return loss


class MaskedPredictionModule(nn.Module):
    """
    Computes the patch and core predictions and labels within the valid loss region for a heatmap.
    """

    def __init__(self):
        super().__init__()



class ImageLevelClassificationLoss(nn.Module):
    def __init__(self, mode, class_weight=None):
        """
        Args:
            mode: "pca" or "cspca" - classification mode
            class_weight: Optional tensor of shape (num_classes,) to weight classes.
                         If None, computes balanced weights automatically from batch.
                         If "balanced", computes weights as: n_samples / (n_classes * np.bincount(y))
        """
        super().__init__()
        self.mode = mode
        self.class_weight = class_weight

    def forward(self, data):
        """
        Computes the image-level classification loss.
        """

        if "image_level_classification_outputs" not in data:
            return torch.tensor(0.0, device=data["label"].device)

        logits = data["image_level_classification_outputs"][0]

        if self.mode == "pca":
            labels = data["label"].to(logits.device)
        elif self.mode == "three_class":
            from .classification import grade_group_to_three_class
            labels = grade_group_to_three_class(data["grade_group"].to(logits.device))
        elif self.mode == "three_way":
            logits = data["prototype_class_logits"]
            cancer = data["label"].long().to(logits.device)
            cspca = (data["grade_group"] > 2).long().to(logits.device)
            labels = torch.where(cspca.bool(), torch.full_like(cancer, 2), cancer)
        else:
            labels = (data["grade_group"] > 2).long().to(logits.device)

        # Compute class weights if needed
        weight = self.class_weight
        if weight == "balanced":
            # Compute balanced weights from current batch
            # weight[class] = n_samples / (n_classes * count[class])
            # For binary classification, we always expect 2 classes
            num_classes = logits.shape[-1]  # Get number of classes from logits
            unique_labels, counts = torch.unique(labels, return_counts=True)
            n_samples = labels.numel()
            
            # Initialize weights for all classes
            weight = torch.ones(num_classes, device=logits.device)
            
            # Update weights for classes present in the batch
            for i, label in enumerate(unique_labels):
                weight[label] = n_samples / (num_classes * counts[i].float())
        elif weight is not None and isinstance(weight, (list, tuple)):
            weight = torch.tensor(weight, device=logits.device, dtype=logits.dtype)
        elif weight is not None:
            # Assume it's already a tensor
            weight = weight.to(logits.device)

        # Compute the cross-entropy loss with optional class weights
        loss = F.cross_entropy(logits, labels, weight=weight)

        return loss


class SumLoss(nn.Module):
    def __init__(self, losses: list[nn.Module], weights=None):
        super().__init__()
        self.losses = nn.ModuleList(losses)
        self.weights = weights if weights is not None else [1.0] * len(losses)



class OutsideProstatePenaltyLoss(nn.Module):
    """
    Soft penalty for decoder predictions outside prostate.
    
    NOTE: This is a SOFT constraint - it penalizes but doesn't prevent
    predictions outside prostate. For a HARD constraint matching the 
    RL attention mask, use DecoderProstateMaskConstraint instead.
    """

    def forward(self, data: dict):
        cancer_logits = data["cancer_logits"]
        prostate_mask = data["prostate_mask"].to(cancer_logits.device)

        masks = []
        for i in range(len(cancer_logits)):
            mask = torch.ones(
                prostate_mask[i].shape, device=prostate_mask[i].device
            ).bool()
            mask &= prostate_mask[i] < 0.5
            masks.append(mask)

        masks = torch.stack(masks)
        predictions, batch_idx = MaskedPredictionModule()(cancer_logits, masks)

        loss = torch.nn.L1Loss()(predictions, torch.zeros_like(predictions))

        return loss



class AttentionAlignmentLoss(nn.Module):
    """
    Direct differentiable loss on the attention map — replaces RL reward with
    a per-pixel gradient signal.
    
    For cancer cores (involvement > 0):
      Encourage attention in the needle region proportional to involvement.
      loss = -involvement * log(mean_attn_in_needle + eps)
    
    For benign cores (involvement == 0):
      Penalize any attention in the needle region.
      loss = mean_attn_in_needle^2  (quadratic to be soft near zero)
    
    This gives the policy a SPATIAL supervision signal that the standard MIL
    loss does not: "where should you attend?" rather than "what's the label?"
    """
    
    def __init__(self, cancer_weight: float = 1.0, benign_weight: float = 1.0):
        super().__init__()
        self.cancer_weight = cancer_weight
        self.benign_weight = benign_weight
    


class StochasticAttentionRegularizationLoss(nn.Module):
    """
    Stochastic Attention Regularization (SAR): adds noise to attention during
    supervised training and penalizes output inconsistency.
    
    Key idea: during training, run the decoder twice — once with clean attention,
    once with noisy attention — and add a consistency loss between outputs.
    This teaches the decoder to be ROBUST to attention variation, improving
    generalization without any RL.
    
    The loss is the KL divergence between the clean and noisy decoder outputs
    (applied to the needle region predictions).
    
    NOTE: This loss requires a special forward mode in the training loop that
    provides both 'cancer_logits' (clean) and 'cancer_logits_noisy' (noisy).
    If 'cancer_logits_noisy' is not present, returns 0 (graceful fallback).
    """
    
    def __init__(self, consistency_weight: float = 1.0):
        super().__init__()
        self.consistency_weight = consistency_weight
    



def build_heatmap_loss(args):
    if args.loss == "needle_region_ce":
        return CancerDetectionValidRegionLoss()
    elif args.loss == "inv_l1":
        return InvolvementL1Loss(**args.loss_kw)
    elif args.loss == "inv_l1_v2":
        return InvolvementL1Loss(prostate_penalty=False)
    elif args.loss == "inv_mse":
        return InvolvementMSELoss(**args.loss_kw)
    elif args.loss == "mil_prop_bce":
        return CancerDetectionMILLoss()
    elif args.loss == "mil_prop_bce_l1_reg":
        return CancerDetectionMILLoss(
            base_loss=ProportionBCE(l1_penalty_lambda=0.001, pos_weight=1.0), 
            treat_gg1_as_benign=args.treat_gg1_as_benign
        )
    elif args.loss == "mil_prop_bce_entropy_reg":
        return CancerDetectionMILLoss(
            base_loss=ProportionBCE(entropy_penalty_lambda=0.01, pos_weight=1.0),
            treat_gg1_as_benign=args.get('treat_gg1_as_benign', False),
        )
    
    # =========================================================================
    # NOISE-ROBUST LOSS FUNCTIONS
    # =========================================================================
    
    # Top-K MIL Loss: Only supervise top k% pixels as cancer
    elif args.loss == "topk_mil":
        loss_kw = args.get('loss_kw', {})
        return TopKInvolvementMILLoss(
            pos_weight=args.get('pos_weight', 1.0),
            min_top_k_fraction=loss_kw.get('min_top_k_fraction', 0.05),
            use_soft_topk=loss_kw.get('use_soft_topk', False),
            temperature=loss_kw.get('temperature', 0.1),
            treat_gg1_as_benign=args.get('treat_gg1_as_benign', False),
            benign_weight=loss_kw.get('benign_weight', 1.0),  # How strongly to push non-topk to 0
        )
    
    elif args.loss == "topk_mil_soft":
        loss_kw = args.get('loss_kw', {})
        return TopKInvolvementMILLoss(
            pos_weight=args.get('pos_weight', 1.0),
            min_top_k_fraction=loss_kw.get('min_top_k_fraction', 0.05),
            use_soft_topk=True,  # Differentiable
            temperature=loss_kw.get('temperature', 0.1),
            treat_gg1_as_benign=args.get('treat_gg1_as_benign', False),
        )
    
    # Thresholded Involvement Loss: Compare fraction of pixels > 0.5 to involvement
    elif args.loss == "thresholded_involvement":
        loss_kw = args.get('loss_kw', {})
        return CancerDetectionMILLoss(
            base_loss=ThresholdedInvolvementLoss(
                threshold=loss_kw.get('threshold', 0.5),
                use_soft_threshold=loss_kw.get('use_soft_threshold', True),
                soft_temperature=loss_kw.get('soft_temperature', 0.1),
                pos_weight=args.get('pos_weight', 1.0),
                entropy_penalty_lambda=loss_kw.get('entropy_penalty_lambda', None),
            ),
            treat_gg1_as_benign=args.get('treat_gg1_as_benign', False),
        )
    
    elif args.loss == "thresholded_involvement_entropy_reg":
        loss_kw = args.get('loss_kw', {})
        return CancerDetectionMILLoss(
            base_loss=ThresholdedInvolvementLoss(
                threshold=loss_kw.get('threshold', 0.5),
                use_soft_threshold=True,
                soft_temperature=loss_kw.get('soft_temperature', 0.1),
                pos_weight=args.get('pos_weight', 1.0),
                entropy_penalty_lambda=loss_kw.get('entropy_penalty_lambda', 0.01),
                tolerance_margin=loss_kw.get('tolerance_margin', 0.0),
                benign_tolerance_margin=loss_kw.get('benign_tolerance_margin', 0.0),
                soft_mean_weight=loss_kw.get('soft_mean_weight', 0.5),
                benign_bce_weight=loss_kw.get('benign_bce_weight', 0.0),
            ),
            treat_gg1_as_benign=args.get('treat_gg1_as_benign', False),
        )

    elif args.loss == "needle_axis_thresholded_involvement_entropy_reg":
        loss_kw = args.get('loss_kw', {})
        return NeedleAxisCancerDetectionMILLoss(
            base_loss=ThresholdedInvolvementLoss(
                threshold=loss_kw.get('threshold', 0.5),
                use_soft_threshold=True,
                soft_temperature=loss_kw.get('soft_temperature', 0.1),
                pos_weight=args.get('pos_weight', 1.0),
                entropy_penalty_lambda=loss_kw.get('entropy_penalty_lambda', 0.01),
                tolerance_margin=loss_kw.get('tolerance_margin', 0.0),
                benign_tolerance_margin=loss_kw.get('benign_tolerance_margin', 0.0),
                soft_mean_weight=loss_kw.get('soft_mean_weight', 0.5),
                benign_bce_weight=loss_kw.get('benign_bce_weight', 0.0),
            ),
            num_bins=loss_kw.get('axis_bins', 32),
            contiguity_weight=loss_kw.get('axis_contiguity_weight', 0.0),
            treat_gg1_as_benign=args.get('treat_gg1_as_benign', False),
        )
    
    # Symmetric Cross Entropy: Noise-robust loss from Manifold DivideMix
    elif args.loss == "symmetric_ce":
        loss_kw = args.get('loss_kw', {})
        return CancerDetectionMILLoss(
            base_loss=SymmetricCrossEntropyLoss(
                alpha=loss_kw.get('alpha', 1.0),
                beta=loss_kw.get('beta', 1.0),
                pos_weight=args.get('pos_weight', 1.0),
                epsilon=loss_kw.get('epsilon', 1e-4),
                entropy_penalty_lambda=loss_kw.get('entropy_penalty_lambda', None),
            ),
            treat_gg1_as_benign=args.get('treat_gg1_as_benign', False),
        )
    
    elif args.loss == "symmetric_ce_entropy_reg":
        loss_kw = args.get('loss_kw', {})
        return CancerDetectionMILLoss(
            base_loss=SymmetricCrossEntropyLoss(
                alpha=loss_kw.get('alpha', 1.0),
                beta=loss_kw.get('beta', 1.0),
                pos_weight=args.get('pos_weight', 1.0),
                epsilon=loss_kw.get('epsilon', 1e-4),
                entropy_penalty_lambda=0.01,
            ),
            treat_gg1_as_benign=args.get('treat_gg1_as_benign', False),
        )
    
    # Combined: Top-K + Symmetric CE (most robust)
    elif args.loss == "topk_symmetric_ce":
        loss_kw = args.get('loss_kw', {})
        # This combines top-k selection with SCE base loss
        # For simplicity, we use TopKInvolvementMILLoss which internally uses BCE
        # and add SCE-style regularization through the entropy penalty
        return TopKInvolvementMILLoss(
            pos_weight=args.get('pos_weight', 1.0),
            min_top_k_fraction=loss_kw.get('min_top_k_fraction', 0.05),
            use_soft_topk=loss_kw.get('use_soft_topk', True),
            temperature=loss_kw.get('temperature', 0.1),
            treat_gg1_as_benign=args.get('treat_gg1_as_benign', False),
        )

    elif args.loss == "none":
        return None
    else:
        raise ValueError(f"Unknown loss function: {args.loss}")


def build_loss(args):

    losses = []

    hmap_loss = build_heatmap_loss(args)

    if hmap_loss is not None:
        # NOTE: Decoder prostate mask constraint is now applied in train_rl.py
        # (ProstNFoundMeta.forward) to avoid double masking.
        # The constraint uses the same boundary tolerance as RL attention.
        if args.get('decoder_prostate_mask_constraint', False):
            print("Decoder prostate mask constraint enabled (applied in forward pass, not loss).")
        losses.append(hmap_loss)

    if args.add_image_clf:
        print(f"Adding image-level classification loss: {args.add_image_clf}")
        class_weight = args.get('image_clf_class_weight', 'balanced')
        losses.append(ImageLevelClassificationLoss(mode=args.image_clf_mode, class_weight=class_weight))

    # Attention Alignment Loss: direct differentiable spatial attention supervision
    attn_align_weight = args.get('attention_alignment_weight', 0.0)
    if attn_align_weight > 0:
        print(f"Adding Attention Alignment Loss with weight={attn_align_weight}")
        attn_align_kw = args.get('attention_alignment_kw', {})
        losses.append(AttentionAlignmentLoss(
            cancer_weight=attn_align_kw.get('cancer_weight', 1.0) * attn_align_weight,
            benign_weight=attn_align_kw.get('benign_weight', 1.0) * attn_align_weight,
        ))

    # Stochastic Attention Regularization: consistency loss under attention noise
    sar_weight = args.get('sar_weight', 0.0)
    if sar_weight > 0:
        print(f"Adding Stochastic Attention Regularization with weight={sar_weight}")
        losses.append(StochasticAttentionRegularizationLoss(consistency_weight=sar_weight))

    # SOFT penalty (legacy approach - NOT recommended for consistency with RL)
    if args.outside_prostate_penalty:
        if args.get('decoder_prostate_mask_constraint', False):
            print("Warning: Both outside_prostate_penalty and decoder_prostate_mask_constraint enabled.")
            print("         The hard mask constraint is already applied. Soft penalty is redundant.")
        else:
            print("Adding outside prostate penalty loss (SOFT constraint).")
            print("         Consider using decoder_prostate_mask_constraint=true for HARD constraint matching RL.")
            losses.append(OutsideProstatePenaltyLoss())

    return SumLoss(losses)

