from typing import Any

import numpy as np

from emotionbridge.constants import (
    EMOTION_LABELS,
    LOW_VARIANCE_EMOTION_LABELS,
    MAJOR_EMOTION_LABELS,
)


def _safe_pearson(x: np.ndarray, y: np.ndarray) -> float:
    if x.size == 0 or y.size == 0:
        return 0.0
    if np.isclose(np.std(x), 0.0) or np.isclose(np.std(y), 0.0):
        return 0.0
    value = np.corrcoef(x, y)[0, 1]
    if np.isnan(value):
        return 0.0
    return float(value)


def compute_regression_metrics(
    predictions: np.ndarray,
    targets: np.ndarray,
) -> dict[str, Any]:
    if predictions.shape != targets.shape:
        msg = f"predictions and targets shape mismatch: {predictions.shape} != {targets.shape}"
        raise ValueError(
            msg,
        )

    mse_per_dim = np.mean(np.square(predictions - targets), axis=0)
    mse_macro = float(np.mean(mse_per_dim))

    pearson_per_dim = np.asarray(
        [
            _safe_pearson(predictions[:, i], targets[:, i])
            for i in range(targets.shape[1])
        ],
        dtype=np.float32,
    )
    top1_pred = np.argmax(predictions, axis=1)
    top1_true = np.argmax(targets, axis=1)
    top1_accuracy = float(np.mean(top1_pred == top1_true))

    mse_per_emotion = {
        emotion: float(value)
        for emotion, value in zip(EMOTION_LABELS, mse_per_dim, strict=True)
    }
    pearson_per_emotion = {
        emotion: float(value)
        for emotion, value in zip(EMOTION_LABELS, pearson_per_dim, strict=True)
    }

    return {
        "mse_per_emotion": mse_per_emotion,
        "mse_macro": mse_macro,
        "pearson_per_emotion": pearson_per_emotion,
        "pearson_min": float(np.min(pearson_per_dim)),
        "pearson_top6_min": float(
            min(pearson_per_emotion[emotion] for emotion in MAJOR_EMOTION_LABELS),
        ),
        "pearson_anger_trust_min": float(
            min(
                pearson_per_emotion[emotion] for emotion in LOW_VARIANCE_EMOTION_LABELS
            ),
        ),
        "top1_accuracy": top1_accuracy,
    }
