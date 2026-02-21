from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol

import numpy as np

from .registry import HFMetricRegistry


class MetricRegistry(Protocol):
    def load(
        self,
        name: str,
        *,
        config_name: str | None = None,
        revision: str | None = None,
        module_type: str | None = None,
    ) -> Any: ...


@dataclass(slots=True)
class ClassificationMetricSuite:
    registry: MetricRegistry = field(default_factory=HFMetricRegistry)

    def compute(
        self,
        logits: np.ndarray,
        true_labels: np.ndarray,
        label_names: list[str],
    ) -> dict[str, Any]:
        self._validate_inputs(logits, true_labels, label_names)

        pred_labels = np.argmax(logits, axis=1).astype(np.int64)
        true_labels_int = true_labels.astype(np.int64)
        label_indices = list(range(len(label_names)))

        pred_list = pred_labels.tolist()
        true_list = true_labels_int.tolist()

        accuracy_metric = self.registry.load("accuracy")
        f1_metric = self.registry.load("f1")
        precision_metric = self.registry.load("precision")
        recall_metric = self.registry.load("recall")
        confusion_metric = self.registry.load("confusion_matrix")

        accuracy = accuracy_metric.compute(predictions=pred_list, references=true_list)
        macro_f1 = f1_metric.compute(
            predictions=pred_list,
            references=true_list,
            average="macro",
            labels=label_indices,
            zero_division=0,
        )
        per_f1 = f1_metric.compute(
            predictions=pred_list,
            references=true_list,
            average=None,
            labels=label_indices,
            zero_division=0,
        )
        per_precision = precision_metric.compute(
            predictions=pred_list,
            references=true_list,
            average=None,
            labels=label_indices,
            zero_division=0,
        )
        per_recall = recall_metric.compute(
            predictions=pred_list,
            references=true_list,
            average=None,
            labels=label_indices,
            zero_division=0,
        )
        confusion = confusion_metric.compute(
            predictions=pred_list,
            references=true_list,
            labels=label_indices,
        )

        per_f1_values = np.asarray(per_f1["f1"], dtype=np.float64)
        per_precision_values = np.asarray(per_precision["precision"], dtype=np.float64)
        per_recall_values = np.asarray(per_recall["recall"], dtype=np.float64)

        return {
            "accuracy": float(accuracy["accuracy"]),
            "macro_f1": float(macro_f1["f1"]),
            "per_class_f1": {
                name: float(value) for name, value in zip(label_names, per_f1_values, strict=True)
            },
            "per_class_precision": {
                name: float(value)
                for name, value in zip(label_names, per_precision_values, strict=True)
            },
            "per_class_recall": {
                name: float(value)
                for name, value in zip(label_names, per_recall_values, strict=True)
            },
            "confusion_matrix": confusion["confusion_matrix"],
        }

    @staticmethod
    def _validate_inputs(
        logits: np.ndarray,
        true_labels: np.ndarray,
        label_names: list[str],
    ) -> None:
        if logits.ndim != 2:
            msg = f"logits must be 2D (N, C), got shape={logits.shape}"
            raise ValueError(msg)

        if true_labels.ndim != 1:
            msg = f"true_labels must be 1D (N,), got shape={true_labels.shape}"
            raise ValueError(msg)

        if logits.shape[0] != true_labels.shape[0]:
            msg = f"logits and true_labels row count mismatch: {logits.shape[0]} != {true_labels.shape[0]}"
            raise ValueError(msg)

        if logits.shape[1] != len(label_names):
            msg = f"number of classes in logits must match label_names: {logits.shape[1]} != {len(label_names)}"
            raise ValueError(msg)
