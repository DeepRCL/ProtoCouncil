from collections import defaultdict
import torch
from .registry import get_bags_of_predictions
from .registry import DataFrameCollector
import numpy as np
from sklearn.metrics import roc_auc_score
from torchvision.transforms import v2 as T
from matplotlib import pyplot as plt
from PIL import Image


def _auc_roc(predictions, labels):
    """Compute AUC-ROC with safety checks for empty or single-class data."""
    nanvalues = np.isnan(predictions)
    predictions = predictions[~nanvalues]
    labels = labels[~nanvalues]
    
    # Check for empty arrays or single class
    if len(predictions) == 0 or len(labels) == 0:
        return float('nan')
    
    unique_labels = np.unique(labels)
    if len(unique_labels) < 2:
        return float('nan')
    
    return roc_auc_score(labels, predictions)


@torch.no_grad()
def show_heatmap_prediction(data):

    plt.close("all")
    plt.figure()

    if "cancer_logits" in data:
        logits = data["cancer_logits"].cpu()
        pred = logits.sigmoid()
    elif "cancer_probs" in data:
        pred = data["cancer_probs"].cpu()
    else:
        raise ValueError()

    needle_mask = data["needle_mask"]
    prostate_mask = data["prostate_mask"]
    image = data["bmode"]
    label = data["label"]

    fig, ax = plt.subplots(1, 2, figsize=(8, 4))
    [ax.set_axis_off() for ax in ax.flatten()]
    kwargs = dict(vmin=0, vmax=1)

    image = T.Resize(
        (224, 224), interpolation=Image.Resampling.BICUBIC, antialias=True
    )(image)
    needle_mask = T.Resize((224, 224), interpolation=Image.Resampling.NEAREST)(
        needle_mask
    )
    prostate_mask = T.Resize((224, 224), interpolation=Image.Resampling.NEAREST)(
        prostate_mask
    )
    pred = T.Resize((224, 224), interpolation=Image.Resampling.NEAREST)(pred)

    # image and contours
    ax[0].imshow(image[0].permute(1, 2, 0), **kwargs)
    ax[0].contour(prostate_mask[0, 0], **kwargs)
    ax[0].contour(needle_mask[0, 0], **kwargs)

    # prediction
    ax[1].imshow(pred[0, 0], **kwargs)
    ax[1].contour(needle_mask[0, 0], **kwargs)
    ax[1].contour(prostate_mask[0, 0], **kwargs)

    # Build title with ground truth and model predictions
    gt_cancer = label[0].item() == 1
    involvement = data["involvement"][0].item()
    grade_group = data["grade_group"][0]

    # Heatmap-based prediction (needle average)
    pred_heatmap = None
    if "average_needle_heatmap_value" in data:
        pred_heatmap = float(data["average_needle_heatmap_value"][0].item())

    # Image-level classifier prediction (if available)
    cls_prob = None
    if "image_level_classification_outputs" in data:
        cls_outputs = data["image_level_classification_outputs"][0].detach().cpu()
        cls_probs = cls_outputs.softmax(-1)
        # Assume binary classification, cancer is class 1
        if cls_probs.ndim == 1:
            cls_prob = float(cls_probs[1].item())
        else:
            cls_prob = float(cls_probs[0, 1].item())

    title_parts = [
        f"GT: Cancer {gt_cancer}",
        f"Inv {involvement:.2f}",
        f"Grade group {grade_group}",
    ]
    if pred_heatmap is not None:
        title_parts.append(f"Heatmap p(cancer) {pred_heatmap:.2f}")
    if cls_prob is not None:
        title_parts.append(f"Image-level p(cancer) {cls_prob:.2f}")

    fig.suptitle("; ".join(title_parts))

    return fig


class CancerLogitsHeatmapsEvaluator:
    def __init__(
        self,
        log_images=False,
        log_images_every=10,
        include_patient_metrics=False,
        include_heatmap_cspca_metrics=True,
        include_balanced_accuracy=True,
    ):
        self.iter = 0
        self.log_images = log_images
        self.log_images_every = log_images_every
        self.include_patient_metrics = include_patient_metrics
        self.accumulator = DataFrameCollector()
        self._heatmap_fig = None
        self.include_heatmap_cspca_metrics = include_heatmap_cspca_metrics
        self.include_balanced_accuracy = include_balanced_accuracy
        self.results_table = None

    @torch.no_grad()
    def __call__(self, data):
        step_metrics = {}

        if "cancer_logits" in data:
            bags_of_logits = get_bags_of_predictions(
                data["cancer_logits"], data["prostate_mask"], data["needle_mask"]
            )
            bags_of_probs = [bag.sigmoid() for bag in bags_of_logits]
        elif "cancer_probs" in data:
            bags_of_probs = get_bags_of_predictions(
                data["cancer_probs"], data["prostate_mask"], data["needle_mask"]
            )

        bag_level_info = defaultdict(list)

        for probs in bags_of_probs:
            # entropy
            normalized_probs = probs / probs.sum()
            entropy = -(normalized_probs * normalized_probs.log()).sum()
            bag_level_info["entropy"].append(entropy.item())

            # topk score
            N = len(probs)
            k = int(N * 0.5)
            topk_score = torch.sort(probs, descending=True).values[:k].mean()
            bag_level_info["topk_score"].append(topk_score.item())

        tracked_data = {}
        keys = [
            "center",
            "core_id",
            "patient_id",
            "loc",
            "grade",
            "age",
            "family_history",
            "psa",
            "pct_cancer",
            "grade_group",
            "average_needle_heatmap_value",
            "average_prostate_heatmap_value",
            "thresholded_needle_involvement",
            "label",
            "involvement",
            "clinically_significant",
        ]
        if "involvement_valid" in data:
            keys.append("involvement_valid")
        for key in keys:
            if key in data:  # descriptive metadata fields (loc, grade, ...) are optional
                tracked_data[key] = data[key]
        tracked_data.update(bag_level_info)

        if "image_level_classification_outputs" in data and data["image_level_classification_outputs"] is not None:
            from .classification import image_classification_probabilities
            pca_probability, cspca_probability = image_classification_probabilities(
                data["image_level_classification_outputs"][0].detach().cpu().float()
            )
            tracked_data["image_level_cancer_logits"] = pca_probability
            if data["image_level_classification_outputs"][0].shape[-1] == 3:
                tracked_data["image_level_cspca_logits"] = cspca_probability
        if "image_level_cspca_outputs" in data and data["image_level_cspca_outputs"] is not None:
            tracked_data["image_level_cspca_logits"] = (
                data["image_level_cspca_outputs"][0]
                .detach()
                .cpu()
                .softmax(-1)[:, 1]
            )

        self.accumulator(tracked_data)

        if self.log_images and (self.iter % self.log_images_every == 0):
            figure = show_heatmap_prediction(data)
            step_metrics["heatmap_example"] = figure

        self.iter += 1
        return step_metrics

    def aggregate_metrics(self, results_table=None):
        from .metrics import calculate_metrics
        from sklearn.metrics import roc_auc_score

        results_table = results_table or self.accumulator.compute()
        self.results_table = results_table

        # core predictions
        predictions = results_table.average_needle_heatmap_value.values
        labels = results_table.label.values
        involvement = results_table.involvement.values.astype(float).copy()
        if "involvement_valid" in results_table.columns:
            involvement_valid = results_table.involvement_valid.values.astype(bool)
            involvement[~involvement_valid] = np.nan
        else:
            involvement_valid = np.ones_like(involvement, dtype=bool)

        core_probs = predictions
        core_labels = labels

        metrics = {}
        metrics_ = calculate_metrics(
            predictions,
            labels,
            log_images=self.log_images,
            include_balanced_accuracy=self.include_balanced_accuracy,
        )
        metrics.update(metrics_)

        metrics["topk_probs_auroc"] = _auc_roc(results_table.topk_score.values, labels)
        metrics["avg_bag_entropy"] = results_table["entropy"].mean()

        # prop pred err
        metrics["prop_pred_err"] = np.abs(
            results_table["average_needle_heatmap_value"].values
            - involvement
        )
        metrics["prop_pred_err"] = float(np.nanmean(metrics["prop_pred_err"]))

        # balanced prop pred err
        results_table["prop_pred_err"] = np.abs(
            results_table["average_needle_heatmap_value"] - involvement
        )
        metrics["bal_prop_pred_err"] = (
            results_table.query("label == 0")["prop_pred_err"].mean()
            + results_table.query("label == 1")["prop_pred_err"].mean()
        ) / 2

        # === THRESHOLDED INVOLVEMENT METRICS ===
        # Calculate metrics using thresholded involvement (binary activations > 0.5)
        if "thresholded_needle_involvement" in results_table.columns:
            thresholded_preds = results_table["thresholded_needle_involvement"].values
            
            # Thresholded AUC (using binary involvement as predictions)
            metrics["core_auc_thresholded"] = _auc_roc(thresholded_preds, labels)
            
            # Thresholded involvement prediction error
            metrics["prop_pred_err_thresholded"] = np.abs(
                thresholded_preds - involvement
            )
            metrics["prop_pred_err_thresholded"] = float(
                np.nanmean(metrics["prop_pred_err_thresholded"])
            )
            
            # Balanced thresholded involvement prediction error
            results_table["prop_pred_err_thresholded"] = np.abs(
                thresholded_preds - involvement
            )
            metrics["bal_prop_pred_err_thresholded"] = (
                results_table.query("label == 0")["prop_pred_err_thresholded"].mean()
                + results_table.query("label == 1")["prop_pred_err_thresholded"].mean()
            ) / 2
            
            # Correlation between thresholded predictions and true involvement
            from scipy.stats import spearmanr, pearsonr
            valid_mask = ~(np.isnan(thresholded_preds) | np.isnan(involvement))
            if valid_mask.sum() > 2:
                corr_spearman, p_val = spearmanr(
                    thresholded_preds[valid_mask], 
                    involvement[valid_mask]
                )
                metrics["thresholded_involvement_correlation_spearman"] = float(corr_spearman)
                metrics["thresholded_involvement_correlation_pvalue"] = float(p_val)
                
                corr_pearson, _ = pearsonr(
                    thresholded_preds[valid_mask], 
                    involvement[valid_mask]
                )
                metrics["thresholded_involvement_correlation_pearson"] = float(corr_pearson)
            
            # Thresholded predictions for high involvement cases
            high_inv_mask = (involvement > 0.4) | (labels == 0)
            if high_inv_mask.sum() > 0:
                metrics["core_auc_thresholded_high_involvement"] = _auc_roc(
                    thresholded_preds[high_inv_mask],
                    labels[high_inv_mask]
                )

        # high involvement core predictions
        high_involvement = involvement > 0.4
        benign = core_labels == 0
        keep = np.logical_or(high_involvement, benign)
        if keep.sum() > 0:
            core_probs = core_probs[keep]
            core_labels = core_labels[keep]
            metrics_ = calculate_metrics(
                core_probs,
                core_labels,
                log_images=self.log_images,
                include_balanced_accuracy=self.include_balanced_accuracy,
            )
            metrics.update(
                {
                    f"{metric}_high_involvement": value
                    for metric, value in metrics_.items()
                }
            )
            metrics["topk_probs_auroc_high_inv"] = _auc_roc(
                results_table.topk_score.values[keep], core_labels
            )

        # patient predictions
        if self.include_patient_metrics:
            predictions = (
                results_table.groupby("patient_id")
                .average_prostate_heatmap_value.mean()
                .values
            )
            labels = (
                results_table.groupby("patient_id").clinically_significant.sum() > 0
            ).values
            metrics_ = calculate_metrics(
                predictions,
                labels,
                log_images=self.log_images,
                include_balanced_accuracy=self.include_balanced_accuracy,
            )
            metrics.update(
                {f"{metric}_patient": value for metric, value in metrics_.items()}
            )

        if "image_level_cancer_logits" in results_table.columns:
            image_level_predictions = results_table.image_level_cancer_logits.values
            image_level_labels = results_table.label.values
            metrics_ = calculate_metrics(
                image_level_predictions,
                image_level_labels,
                log_images=self.log_images,
                include_balanced_accuracy=self.include_balanced_accuracy,
            )
            metrics.update(
                {f"{metric}_image_level": value for metric, value in metrics_.items()}
            )

            image_level_labels = (results_table.grade_group.values > 2).astype(int)
            cspca_predictions = (
                results_table.image_level_cspca_logits.values
                if "image_level_cspca_logits" in results_table.columns
                else image_level_predictions
            )
            metrics_low_vs_high = metrics_ = calculate_metrics(
                cspca_predictions,
                image_level_labels,
                log_images=self.log_images,
                include_balanced_accuracy=self.include_balanced_accuracy,
            )
            metrics.update(
                {
                    f"{metric}_image_level_cspca": value
                    for metric, value in metrics_low_vs_high.items()
                }
            )

            # Classification head AUC restricted to high-involvement cancer + benign
            # (same subset used for core_auc_high_involvement)
            hi_keep = np.logical_or(
                involvement > 0.4,
                results_table.label.values == 0,
            )
            if hi_keep.sum() > 0:
                metrics["core_auc_image_level_high_inv"] = _auc_roc(
                    image_level_predictions[hi_keep],
                    results_table.label.values[hi_keep],
                )

        if self.include_heatmap_cspca_metrics:
            heatmap_predictions = results_table["average_needle_heatmap_value"]
            image_level_labels = (results_table.grade_group.values > 2).astype(int)
            metrics_ = calculate_metrics(
                heatmap_predictions,
                image_level_labels,
                log_images=self.log_images,
                include_balanced_accuracy=self.include_balanced_accuracy,
            )
            metrics.update(
                {
                    f"{metric}_heatmap_cspca": value
                    for metric, value in metrics_.items()
                }
            )

        # Combined high-involvement AUC: heatmap head + classification head,
        # both restricted to involvement > 40% | benign.
        # Range [0, 2] — maximised when both heads are perfect (AUC = 1 each).
        # Used as tracked_metric to balance spatial heatmap quality and
        # classification head performance on the clinically important subset.
        if "core_auc_high_involvement" in metrics and "core_auc_image_level_high_inv" in metrics:
            metrics["combined_high_inv_auc"] = (
                metrics["core_auc_high_involvement"]
                + metrics["core_auc_image_level_high_inv"]
            )

        return metrics
    
