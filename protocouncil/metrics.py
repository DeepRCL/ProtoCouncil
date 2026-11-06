"""Core- and patient-level metrics used in the paper."""
import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve


def auroc(y, score):
    y, score = np.asarray(y, bool), np.asarray(score, float)
    ok = np.isfinite(score)
    y, score = y[ok], score[ok]
    return float(roc_auc_score(y, score)) if 0 < y.sum() < len(y) else float('nan')


def sensitivity_at_specificity(y, score, specificity):
    """Highest sensitivity among ROC operating points whose specificity is at least ``specificity``."""
    y, score = np.asarray(y, bool), np.asarray(score, float)
    if not 0 < y.sum() < len(y):
        return float('nan')
    fpr, tpr, _ = roc_curve(y, score)
    ok = fpr <= 1 - specificity + 1e-12
    return float(tpr[ok].max())


def selection_score(grade_group, p):
    """Mean of all-cancer and csPCa AUROC; ``p`` is [N, 3] class probabilities (used for checkpoint selection)."""
    gg = np.asarray(grade_group)
    scores = [auroc(gg >= k, s) for k, s in ((1, p[:, 1:].sum(1)), (3, p[:, 2]))]
    scores = [s for s in scores if np.isfinite(s)]
    return float(np.mean(scores)) if scores else float('nan')


def detection_metrics(table, high_involvement=.4):
    """Paper Table 1 metrics from a predictions table with columns grade_group, involvement, patient_id,
    p_cancer (= P(GG>=1)) and p_cspca (= P(GG>=3)).

    High-inv.: benign cores vs cancer cores with involvement > 40%. Patient AUROC: csPCa, mean core score per patient.
    """
    gg = table.grade_group.values
    cancer, cspca = gg >= 1, gg >= 3
    hi = (gg == 0) | (cancer & (table.involvement.values > high_involvement))
    pat = table.groupby('patient_id').agg(score=('p_cspca', 'mean'), y=('grade_group', lambda g: (g >= 3).any()))
    return dict(
        all_auroc=auroc(cancer, table.p_cancer),
        all_sens80=sensitivity_at_specificity(cancer, table.p_cancer, .8),
        high_inv_auroc=auroc(cancer[hi], table.p_cancer.values[hi]),
        cspca_auroc=auroc(cspca, table.p_cspca),
        cspca_sens60=sensitivity_at_specificity(cspca, table.p_cspca, .6),
        cspca_sens80=sensitivity_at_specificity(cspca, table.p_cspca, .8),
        patient_auroc=auroc(pat.y, pat.score),
    )
