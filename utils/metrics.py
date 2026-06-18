"""Evaluation metrics for PanCirc-Fungi."""

import numpy as np
from scipy import stats
from sklearn.metrics import (
    roc_auc_score, average_precision_score, f1_score,
    precision_recall_curve, accuracy_score, matthews_corrcoef,
)


def compute_auroc(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Area under ROC curve."""
    try:
        return roc_auc_score(y_true, y_pred)
    except ValueError:
        return 0.0


def compute_auprc(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Area under precision-recall curve."""
    try:
        return average_precision_score(y_true, y_pred)
    except ValueError:
        return 0.0


def compute_f1(y_true: np.ndarray, y_pred: np.ndarray,
               threshold: float = 0.5) -> float:
    """F1 score at threshold."""
    return f1_score(y_true, (y_pred >= threshold).astype(int))


def compute_accuracy(y_true: np.ndarray, y_pred: np.ndarray,
                     threshold: float = 0.5) -> float:
    return accuracy_score(y_true, (y_pred >= threshold).astype(int))


def compute_mcc(y_true: np.ndarray, y_pred: np.ndarray,
                threshold: float = 0.5) -> float:
    return matthews_corrcoef(y_true, (y_pred >= threshold).astype(int))


def compute_topk_accuracy(y_true: np.ndarray, y_pred: np.ndarray,
                          k: int = 1) -> float:
    """Top-k accuracy for junction ranking."""
    if y_pred.ndim == 1:
        return 0.0
    topk = np.argsort(-y_pred, axis=1)[:, :k]
    correct = 0
    total = 0
    for i in range(y_true.shape[0]):
        if y_true[i].sum() > 0:
            true_idx = np.where(y_true[i] > 0)[0]
            if np.any(np.isin(topk[i], true_idx)):
                correct += 1
            total += 1
    return correct / max(total, 1)


def compute_junction_fuzzy_accuracy(
    y_pred_flat: np.ndarray,           # (N_d*N_a,) raw junction scores
    cross_labels: np.ndarray,          # (N_d, N_a) binary true junction labels
    donor_positions: np.ndarray,       # (N_d,) genomic coord of each donor site
    acceptor_positions: np.ndarray,    # (N_a,) genomic coord of each acceptor site
    true_junctions: list,              # [(true_donor_pos, true_acceptor_pos), ...]
    tolerance: int = 5,
    k: int = 1,
) -> tuple:
    """Top-K junction accuracy with ±tolerance bp fuzzy matching.

    Instead of requiring an exact exon-index match, this checks whether
    the predicted junction coordinates fall within `tolerance` bp of any
    true backsplice junction.

    Returns (correct: int, total: int).
    """
    n_donors = len(donor_positions)
    n_acceptors = len(acceptor_positions)

    # Top-K predicted pairs (exon indices)
    flat_scores = y_pred_flat.flatten()
    topk_flat_idx = np.argsort(-flat_scores)[:k]

    total = min(len(true_junctions), 1)  # count this gene once
    correct = 0

    for flat_idx in topk_flat_idx:
        d_idx = flat_idx // n_acceptors
        a_idx = flat_idx % n_acceptors
        if d_idx >= n_donors or a_idx >= n_acceptors:
            continue
        pred_d_pos = donor_positions[d_idx]
        pred_a_pos = acceptor_positions[a_idx]

        for true_d_pos, true_a_pos in true_junctions:
            if (abs(pred_d_pos - true_d_pos) <= tolerance and
                abs(pred_a_pos - true_a_pos) <= tolerance):
                correct = 1
                break
        if correct:
            break

    return correct, total


def compute_recall_fuzzy(
    y_pred_flat: np.ndarray,
    donor_positions: np.ndarray,
    acceptor_positions: np.ndarray,
    true_junctions: list,
    tolerance: int = 5,
    k: int = 3,
) -> float:
    """Recall@K with fuzzy matching: how many true junctions are recovered in top-K."""
    n_donors = len(donor_positions)
    n_acceptors = len(acceptor_positions)

    flat_scores = y_pred_flat.flatten()
    topk_flat_idx = np.argsort(-flat_scores)[:k]

    # Build set of predicted coordinates within tolerance
    pred_intervals = []
    for flat_idx in topk_flat_idx:
        d_idx = flat_idx // n_acceptors
        a_idx = flat_idx % n_acceptors
        if d_idx < n_donors and a_idx < n_acceptors:
            pred_intervals.append((
                (donor_positions[d_idx] - tolerance, donor_positions[d_idx] + tolerance),
                (acceptor_positions[a_idx] - tolerance, acceptor_positions[a_idx] + tolerance),
            ))

    if not true_junctions:
        return 0.0

    found = 0
    for true_d_pos, true_a_pos in true_junctions:
        for (d_lo, d_hi), (a_lo, a_hi) in pred_intervals:
            if d_lo <= true_d_pos <= d_hi and a_lo <= true_a_pos <= a_hi:
                found += 1
                break

    return found / len(true_junctions)


def report_metrics(metrics: dict, prefix: str = "") -> str:
    """Format metrics for logging."""
    lines = [f"{'='*50}"]
    for k, v in metrics.items():
        lines.append(f"  {prefix}{k}: {v:.4f}")
    lines.append(f"{'='*50}")
    return "\n".join(lines)


def classification_metrics(y_true: np.ndarray, y_prob: np.ndarray
                           ) -> dict:
    """Compute all classification metrics at once."""
    return {
        "auroc": compute_auroc(y_true, y_prob),
        "auprc": compute_auprc(y_true, y_prob),
        "f1": compute_f1(y_true, y_prob),
        "accuracy": compute_accuracy(y_true, y_prob),
        "mcc": compute_mcc(y_true, y_prob),
    }


# ═══════════════════════════════════════════════════════════════════════
#  DeLong Test for ROC AUC Comparison
# ═══════════════════════════════════════════════════════════════════════

def _delong_theta(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute the Mann-Whitney statistic (theta = AUC)."""
    pos = y_pred[y_true == 1]
    neg = y_pred[y_true == 0]
    n_pos = len(pos)
    n_neg = len(neg)
    if n_pos == 0 or n_neg == 0:
        return 0.5
    # Count pairs where pos > neg (ties count as 0.5)
    count = 0
    for p in pos:
        count += np.sum(p > neg) + 0.5 * np.sum(p == neg)
    return count / (n_pos * n_neg)


def _delong_varcov(y_true: np.ndarray, y_pred_a: np.ndarray,
                   y_pred_b: np.ndarray) -> np.ndarray:
    """Compute variance-covariance matrix for two AUC estimates.

    Returns 2x2 matrix: [[var(A), cov(A,B)], [cov(A,B), var(B)]]
    """
    n_pos = int(y_true.sum())
    n_neg = int((1 - y_true).sum())

    pos_idx = y_true == 1
    neg_idx = y_true == 0

    # Components for variance calculation
    V10 = np.zeros((n_pos, 2))
    V01 = np.zeros((n_neg, 2))

    for k in range(2):
        y_pred = y_pred_a if k == 0 else y_pred_b
        pos_scores = y_pred[pos_idx]
        neg_scores = y_pred[neg_idx]

        for i in range(n_pos):
            V10[i, k] = np.mean(pos_scores[i] > neg_scores) + \
                        0.5 * np.mean(pos_scores[i] == neg_scores)

        for j in range(n_neg):
            V01[j, k] = np.mean(pos_scores > neg_scores[j]) + \
                        0.5 * np.mean(pos_scores == neg_scores[j])

    S10 = np.cov(V10, rowvar=False) if n_pos > 1 else np.zeros((2, 2))
    S01 = np.cov(V01, rowvar=False) if n_neg > 1 else np.zeros((2, 2))

    return S10 / n_pos + S01 / n_neg


def delong_roc_test(y_true: np.ndarray, y_pred_a: np.ndarray,
                    y_pred_b: np.ndarray) -> dict:
    """DeLong test for comparing two ROC AUCs.

    Parameters
    ----------
    y_true : array-like, shape (n_samples,)
        Binary ground-truth labels (0/1).
    y_pred_a : array-like, shape (n_samples,)
        Prediction scores from model A.
    y_pred_b : array-like, shape (n_samples,)
        Prediction scores from model B.

    Returns
    -------
    dict with keys:
        auc_a, auc_b, auc_diff, auc_diff_ci95,
        var_diff, z_stat, p_value (two-sided)
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred_a = np.asarray(y_pred_a, dtype=float)
    y_pred_b = np.asarray(y_pred_b, dtype=float)

    # Filter valid samples (where y_true >= 0)
    valid = y_true >= 0
    y_true = y_true[valid]
    y_pred_a = y_pred_a[valid]
    y_pred_b = y_pred_b[valid]

    n_classes = len(np.unique(y_true))
    if n_classes < 2:
        return {
            "auc_a": 0.5, "auc_b": 0.5, "auc_diff": 0.0,
            "auc_diff_ci95": (0.0, 0.0),
            "var_diff": 0.0, "z_stat": 0.0, "p_value": 1.0,
        }

    auc_a = roc_auc_score(y_true, y_pred_a)
    auc_b = roc_auc_score(y_true, y_pred_b)

    var_cov = _delong_varcov(y_true, y_pred_a, y_pred_b)
    var_diff = var_cov[0, 0] + var_cov[1, 1] - 2 * var_cov[0, 1]

    if var_diff <= 0:
        return {
            "auc_a": auc_a, "auc_b": auc_b, "auc_diff": auc_a - auc_b,
            "auc_diff_ci95": (auc_a - auc_b - 0.1, auc_a - auc_b + 0.1),
            "var_diff": 0.0, "z_stat": 0.0, "p_value": 1.0,
        }

    z_stat = (auc_a - auc_b) / np.sqrt(var_diff)
    p_value = 2 * stats.norm.sf(abs(z_stat))
    z_95 = stats.norm.ppf(0.975)
    ci_half = z_95 * np.sqrt(var_diff)

    return {
        "auc_a": auc_a,
        "auc_b": auc_b,
        "auc_diff": auc_a - auc_b,
        "auc_diff_ci95": (auc_a - auc_b - ci_half, auc_a - auc_b + ci_half),
        "var_diff": var_diff,
        "z_stat": z_stat,
        "p_value": p_value,
    }
