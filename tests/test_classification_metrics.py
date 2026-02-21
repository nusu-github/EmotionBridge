from __future__ import annotations

from unittest.mock import patch

import numpy as np
import pytest
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score

from emotionbridge.eval.classification import ClassificationMetricSuite
from emotionbridge.training.classification_metrics import compute_classification_metrics


class _FakeMetric:
    def __init__(self, name: str) -> None:
        self.name = name

    def compute(self, **kwargs):
        predictions = np.asarray(kwargs.get("predictions", []), dtype=np.int64)
        references = np.asarray(kwargs.get("references", []), dtype=np.int64)
        labels = kwargs.get("labels")

        if self.name == "accuracy":
            return {"accuracy": float(np.mean(predictions == references))}

        if self.name == "f1":
            average = kwargs.get("average")
            return {
                "f1": f1_score(
                    references,
                    predictions,
                    average=average,
                    labels=labels,
                    zero_division=0,
                ),
            }

        if self.name == "precision":
            average = kwargs.get("average")
            return {
                "precision": precision_score(
                    references,
                    predictions,
                    average=average,
                    labels=labels,
                    zero_division=0,
                ),
            }

        if self.name == "recall":
            average = kwargs.get("average")
            return {
                "recall": recall_score(
                    references,
                    predictions,
                    average=average,
                    labels=labels,
                    zero_division=0,
                ),
            }

        if self.name == "confusion_matrix":
            matrix = confusion_matrix(references, predictions, labels=labels)
            return {"confusion_matrix": matrix.tolist()}

        msg = f"Unsupported fake metric: {self.name}"
        raise ValueError(msg)


class _FakeRegistry:
    def __init__(self) -> None:
        self.loaded_names: list[str] = []

    def load(self, name: str, **_kwargs):
        self.loaded_names.append(name)
        return _FakeMetric(name)


@pytest.fixture
def sample_inputs() -> tuple[np.ndarray, np.ndarray, list[str]]:
    label_names = ["anger", "happy", "sad"]
    logits = np.asarray(
        [
            [9.0, 1.0, 0.0],
            [0.5, 3.0, 0.1],
            [1.2, 0.8, 1.0],
            [0.2, 0.1, 2.5],
            [1.0, 1.1, 0.1],
            [0.2, 0.7, 1.2],
        ],
        dtype=np.float64,
    )
    true = np.asarray([0, 1, 2, 2, 1, 2], dtype=np.int64)
    return logits, true, label_names


def test_suite_returns_expected_contract(
    sample_inputs: tuple[np.ndarray, np.ndarray, list[str]],
) -> None:
    logits, true, label_names = sample_inputs
    registry = _FakeRegistry()
    suite = ClassificationMetricSuite(registry=registry)

    result = suite.compute(logits, true, label_names)

    pred = np.argmax(logits, axis=1)
    class_indices = np.arange(len(label_names))

    expected_accuracy = float(np.mean(pred == true))
    expected_macro_f1 = float(
        f1_score(true, pred, average="macro", labels=class_indices, zero_division=0),
    )
    expected_per_f1 = f1_score(
        true,
        pred,
        average=None,
        labels=class_indices,
        zero_division=0,
    )
    expected_per_precision = precision_score(
        true,
        pred,
        average=None,
        labels=class_indices,
        zero_division=0,
    )
    expected_per_recall = recall_score(
        true,
        pred,
        average=None,
        labels=class_indices,
        zero_division=0,
    )
    expected_confusion = confusion_matrix(true, pred, labels=class_indices).tolist()

    assert result["accuracy"] == pytest.approx(expected_accuracy)
    assert result["macro_f1"] == pytest.approx(expected_macro_f1)
    assert result["confusion_matrix"] == expected_confusion
    assert result["per_class_f1"] == {
        label: float(score) for label, score in zip(label_names, expected_per_f1, strict=True)
    }
    assert result["per_class_precision"] == {
        label: float(score)
        for label, score in zip(label_names, expected_per_precision, strict=True)
    }
    assert result["per_class_recall"] == {
        label: float(score) for label, score in zip(label_names, expected_per_recall, strict=True)
    }
    assert registry.loaded_names == ["accuracy", "f1", "precision", "recall", "confusion_matrix"]


def test_suite_rejects_invalid_shapes(
    sample_inputs: tuple[np.ndarray, np.ndarray, list[str]],
) -> None:
    logits, true, label_names = sample_inputs
    suite = ClassificationMetricSuite(registry=_FakeRegistry())

    with pytest.raises(ValueError, match="logits must be 2D"):
        suite.compute(logits.reshape(-1), true, label_names)

    with pytest.raises(ValueError, match="true_labels must be 1D"):
        suite.compute(logits, true.reshape(-1, 1), label_names)

    with pytest.raises(ValueError, match="number of classes in logits"):
        suite.compute(logits[:, :2], true, label_names)


def test_legacy_wrapper_delegates_to_default_suite(
    sample_inputs: tuple[np.ndarray, np.ndarray, list[str]],
) -> None:
    logits, true, label_names = sample_inputs
    sentinel = {
        "accuracy": 1.0,
        "macro_f1": 1.0,
        "per_class_f1": {},
        "per_class_precision": {},
        "per_class_recall": {},
        "confusion_matrix": [],
    }

    class _SuiteStub:
        def __init__(self) -> None:
            self.calls: list[tuple[np.ndarray, np.ndarray, list[str]]] = []

        def compute(
            self,
            logits: np.ndarray,
            true_labels: np.ndarray,
            label_names: list[str],
        ) -> dict[str, float | dict[str, float] | list[list[float]]]:
            self.calls.append((logits, true_labels, label_names))
            return sentinel

    suite_stub = _SuiteStub()

    with patch(
        "emotionbridge.training.classification_metrics._DEFAULT_CLASSIFICATION_SUITE", suite_stub
    ):
        result = compute_classification_metrics(logits, true, label_names)

    assert result is sentinel
    assert len(suite_stub.calls) == 1
