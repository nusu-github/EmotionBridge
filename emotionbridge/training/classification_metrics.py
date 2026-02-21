from typing import Any

import numpy as np

from emotionbridge.eval import ClassificationMetricSuite


_DEFAULT_CLASSIFICATION_SUITE = ClassificationMetricSuite()


def compute_classification_metrics(
    logits: np.ndarray,
    true_distributions: np.ndarray,
    label_names: list[str],
    key_emotions: list[str] | None = None,
) -> dict[str, Any]:
    return _DEFAULT_CLASSIFICATION_SUITE.compute(
        logits,
        true_distributions,
        label_names,
        key_emotions,
    )
