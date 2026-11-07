"""Shared metric helper used by both the training script and the Streamlit app."""
from sklearn.metrics import (accuracy_score, roc_auc_score, precision_score,
                             recall_score, f1_score, matthews_corrcoef)

METRIC_NAMES = ["Accuracy", "AUC", "Precision", "Recall", "F1", "MCC"]


def score_model(fitted_model, features, truth):
    """Return the six assignment metrics for a fitted classifier (no retraining)."""
    predicted = fitted_model.predict(features)
    probs = fitted_model.predict_proba(features)
    if probs.shape[1] == 2:
        auc, avg = roc_auc_score(truth, probs[:, 1]), "binary"
    else:
        auc, avg = roc_auc_score(truth, probs, multi_class="ovr", average="weighted"), "weighted"
    return {
        "Accuracy": accuracy_score(truth, predicted),
        "AUC": auc,
        "Precision": precision_score(truth, predicted, average=avg, zero_division=0),
        "Recall": recall_score(truth, predicted, average=avg, zero_division=0),
        "F1": f1_score(truth, predicted, average=avg, zero_division=0),
        "MCC": matthews_corrcoef(truth, predicted),
    }
