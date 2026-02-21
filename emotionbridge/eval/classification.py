from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(slots=True)
class ClassificationMetricSuite:
    epsilon: float = 1e-8

    def compute(
        self,
        logits: np.ndarray,
        true_distributions: np.ndarray,
        label_names: list[str],
        key_emotions: list[str] | None = None,
    ) -> dict[str, Any]:
        self._validate_inputs(logits, true_distributions, label_names)

        probs = self._softmax(logits)
        targets = self._normalize_distributions(true_distributions)

        safe_probs = np.clip(probs, self.epsilon, 1.0)
        safe_targets = np.clip(targets, self.epsilon, 1.0)

        log_probs = np.log(safe_probs)
        log_targets = np.log(safe_targets)

        kl_per_sample = np.sum(targets * (log_targets - log_probs), axis=1)
        ce_per_sample = -np.sum(targets * log_probs, axis=1)
        brier_per_sample = np.sum(np.square(probs - targets), axis=1)
        per_emotion_mae_values = np.mean(np.abs(probs - targets), axis=0)

        effective_key_emotions = key_emotions or label_names
        key_indices = [
            index for index, label in enumerate(label_names) if label in effective_key_emotions
        ]
        if not key_indices:
            key_indices = list(range(len(label_names)))

        key_emotion_mae = float(np.mean(per_emotion_mae_values[key_indices]))
        target_entropy = -np.sum(targets * log_targets, axis=1)
        pred_entropy = -np.sum(probs * log_probs, axis=1)

        return {
            "mean_kl": float(np.mean(kl_per_sample)),
            "mean_cross_entropy": float(np.mean(ce_per_sample)),
            "brier_score": float(np.mean(brier_per_sample)),
            "key_emotion_mae": key_emotion_mae,
            "per_emotion_mae": {
                name: float(value)
                for name, value in zip(label_names, per_emotion_mae_values, strict=True)
            },
            "mean_target_entropy": float(np.mean(target_entropy)),
            "mean_prediction_entropy": float(np.mean(pred_entropy)),
        }

    @staticmethod
    def _softmax(logits: np.ndarray) -> np.ndarray:
        shifted = logits - np.max(logits, axis=1, keepdims=True)
        exp_values = np.exp(shifted)
        sums = np.sum(exp_values, axis=1, keepdims=True)
        return exp_values / np.maximum(sums, 1e-12)

    @staticmethod
    def _normalize_distributions(distributions: np.ndarray) -> np.ndarray:
        clipped = np.clip(distributions.astype(np.float64), 0.0, None)
        sums = np.sum(clipped, axis=1, keepdims=True)
        return clipped / np.maximum(sums, 1e-12)

    @staticmethod
    def _validate_inputs(
        logits: np.ndarray,
        true_distributions: np.ndarray,
        label_names: list[str],
    ) -> None:
        if logits.ndim != 2:
            msg = f"logits must be 2D (N, C), got shape={logits.shape}"
            raise ValueError(msg)

        if true_distributions.ndim != 2:
            msg = f"true_distributions must be 2D (N, C), got shape={true_distributions.shape}"
            raise ValueError(msg)

        if logits.shape[0] != true_distributions.shape[0]:
            msg = (
                "logits and true_distributions row count mismatch: "
                f"{logits.shape[0]} != {true_distributions.shape[0]}"
            )
            raise ValueError(msg)

        if logits.shape[1] != len(label_names):
            msg = (
                "number of classes in logits must match label_names: "
                f"{logits.shape[1]} != {len(label_names)}"
            )
            raise ValueError(msg)

        if true_distributions.shape[1] != len(label_names):
            msg = (
                "number of classes in true_distributions must match label_names: "
                f"{true_distributions.shape[1]} != {len(label_names)}"
            )
            raise ValueError(msg)
