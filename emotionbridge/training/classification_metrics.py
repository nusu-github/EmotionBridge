from typing import Any

import numpy as np
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score


def compute_classification_metrics(
    logits: np.ndarray,
    true_labels: np.ndarray,
    label_names: list[str],
) -> dict[str, Any]:
    if logits.ndim != 2:
        msg = f"logits must be 2D (N, C), got shape={logits.shape}"
        raise ValueError(msg)

    if true_labels.ndim != 1:
        msg = f"true_labels must be 1D (N,), got shape={true_labels.shape}"
        raise ValueError(msg)

    if logits.shape[0] != true_labels.shape[0]:
        msg = (
            "logits and true_labels row count mismatch: "
            f"{logits.shape[0]} != {true_labels.shape[0]}"
        )
        raise ValueError(msg)

    if logits.shape[1] != len(label_names):
        msg = (
            "number of classes in logits must match label_names: "
            f"{logits.shape[1]} != {len(label_names)}"
        )
        raise ValueError(msg)

    pred_labels = np.argmax(logits, axis=1).astype(np.int64)
    true_labels = true_labels.astype(np.int64)

    per_f1 = f1_score(
        true_labels,
        pred_labels,
        average=None,
        labels=np.arange(len(label_names)),
        zero_division=0,
    )
    per_precision = precision_score(
        true_labels,
        pred_labels,
        average=None,
        labels=np.arange(len(label_names)),
        zero_division=0,
    )
    per_recall = recall_score(
        true_labels,
        pred_labels,
        average=None,
        labels=np.arange(len(label_names)),
        zero_division=0,
    )

    return {
        "accuracy": float(np.mean(pred_labels == true_labels)),
        "macro_f1": float(
            f1_score(
                true_labels,
                pred_labels,
                average="macro",
                labels=np.arange(len(label_names)),
                zero_division=0,
            ),
        ),
        "per_class_f1": {
            name: float(value) for name, value in zip(label_names, per_f1, strict=True)
        },
        "per_class_precision": {
            name: float(value)
            for name, value in zip(label_names, per_precision, strict=True)
        },
        "per_class_recall": {
            name: float(value)
            for name, value in zip(label_names, per_recall, strict=True)
        },
        "confusion_matrix": confusion_matrix(
            true_labels,
            pred_labels,
            labels=np.arange(len(label_names)),
        ).tolist(),
    }
