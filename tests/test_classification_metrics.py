from __future__ import annotations

from unittest.mock import patch

import numpy as np
import pytest

from emotionbridge.eval.classification import ClassificationMetricSuite
from emotionbridge.training.classification_metrics import compute_classification_metrics


@pytest.fixture
def sample_inputs() -> tuple[np.ndarray, np.ndarray, list[str], list[str]]:
    label_names = ["anger", "happy", "sad"]
    key_emotions = ["anger", "sad"]
    logits = np.asarray(
        [
            [2.0, 0.1, -0.3],
            [0.2, 1.5, 0.7],
            [0.1, 0.4, 1.8],
            [1.1, 0.9, 0.4],
        ],
        dtype=np.float64,
    )
    targets = np.asarray(
        [
            [0.80, 0.10, 0.10],
            [0.20, 0.60, 0.20],
            [0.05, 0.15, 0.80],
            [0.55, 0.30, 0.15],
        ],
        dtype=np.float64,
    )
    return logits, targets, label_names, key_emotions


def _softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - np.max(logits, axis=1, keepdims=True)
    exp_values = np.exp(shifted)
    return exp_values / np.sum(exp_values, axis=1, keepdims=True)


def test_suite_returns_expected_contract(
    sample_inputs: tuple[np.ndarray, np.ndarray, list[str], list[str]],
) -> None:
    logits, targets, label_names, key_emotions = sample_inputs
    suite = ClassificationMetricSuite()

    result = suite.compute(logits, targets, label_names, key_emotions)

    probs = _softmax(logits)
    eps = 1e-8
    safe_probs = np.clip(probs, eps, 1.0)
    safe_targets = np.clip(targets, eps, 1.0)
    log_probs = np.log(safe_probs)
    log_targets = np.log(safe_targets)

    kl = np.sum(targets * (log_targets - log_probs), axis=1)
    ce = -np.sum(targets * log_probs, axis=1)
    brier = np.sum(np.square(probs - targets), axis=1)
    per_emotion_mae = np.mean(np.abs(probs - targets), axis=0)
    target_entropy = -np.sum(targets * log_targets, axis=1)
    pred_entropy = -np.sum(probs * log_probs, axis=1)

    key_indices = [i for i, label in enumerate(label_names) if label in key_emotions]
    expected_key_mae = float(np.mean(per_emotion_mae[key_indices]))

    assert result["mean_kl"] == pytest.approx(float(np.mean(kl)))
    assert result["mean_cross_entropy"] == pytest.approx(float(np.mean(ce)))
    assert result["brier_score"] == pytest.approx(float(np.mean(brier)))
    assert result["key_emotion_mae"] == pytest.approx(expected_key_mae)
    assert result["mean_target_entropy"] == pytest.approx(float(np.mean(target_entropy)))
    assert result["mean_prediction_entropy"] == pytest.approx(float(np.mean(pred_entropy)))
    assert result["per_emotion_mae"] == {
        label: pytest.approx(float(value))
        for label, value in zip(label_names, per_emotion_mae, strict=True)
    }


def test_suite_rejects_invalid_shapes(
    sample_inputs: tuple[np.ndarray, np.ndarray, list[str], list[str]],
) -> None:
    logits, targets, label_names, key_emotions = sample_inputs
    suite = ClassificationMetricSuite()

    with pytest.raises(ValueError, match="logits must be 2D"):
        suite.compute(logits.reshape(-1), targets, label_names, key_emotions)

    with pytest.raises(ValueError, match="true_distributions must be 2D"):
        suite.compute(logits, targets.reshape(-1), label_names, key_emotions)

    with pytest.raises(ValueError, match="row count mismatch"):
        suite.compute(logits[:-1], targets, label_names, key_emotions)

    with pytest.raises(ValueError, match="number of classes in logits"):
        suite.compute(logits[:, :2], targets[:, :2], label_names, key_emotions)

    with pytest.raises(ValueError, match="number of classes in true_distributions"):
        suite.compute(logits, targets[:, :2], label_names, key_emotions)


def test_wrapper_delegates_to_default_suite(
    sample_inputs: tuple[np.ndarray, np.ndarray, list[str], list[str]],
) -> None:
    logits, targets, label_names, key_emotions = sample_inputs
    sentinel = {"mean_kl": 0.123}

    class _SuiteStub:
        def __init__(self) -> None:
            self.calls: list[tuple[np.ndarray, np.ndarray, list[str], list[str] | None]] = []

        def compute(
            self,
            logits: np.ndarray,
            true_distributions: np.ndarray,
            label_names: list[str],
            key_emotions: list[str] | None,
        ) -> dict[str, float]:
            self.calls.append((logits, true_distributions, label_names, key_emotions))
            return sentinel

    suite_stub = _SuiteStub()

    with patch(
        "emotionbridge.training.classification_metrics._DEFAULT_CLASSIFICATION_SUITE",
        suite_stub,
    ):
        result = compute_classification_metrics(logits, targets, label_names, key_emotions)

    assert result is sentinel
    assert len(suite_stub.calls) == 1
