from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from emotionbridge.constants import (
    CIRCUMPLEX_AXIS_NAMES,
    EMOTION_CIRCUMPLEX_COORDS,
    EMOTION_LABELS,
)

if TYPE_CHECKING:
    from collections.abc import Sequence

_CIRCUMPLEX_MATRIX = np.asarray(
    [EMOTION_CIRCUMPLEX_COORDS[label] for label in EMOTION_LABELS],
    dtype=np.float32,
)


def emotion8d_to_av(
    emotion_vec: Sequence[float] | np.ndarray,
    *,
    normalize_weights: bool = True,
    eps: float = 1e-8,
) -> np.ndarray:
    vector = np.asarray(emotion_vec, dtype=np.float32)
    expected = (len(EMOTION_LABELS),)
    if vector.shape != expected:
        msg = f"Expected shape {expected}, got {vector.shape}"
        raise ValueError(msg)

    clipped = np.clip(vector, 0.0, 1.0)
    if normalize_weights:
        denom = float(clipped.sum())
        if denom <= eps:
            return np.zeros((2,), dtype=np.float32)
        weights = clipped / denom
    else:
        weights = clipped

    return (weights @ _CIRCUMPLEX_MATRIX).astype(np.float32)


def emotion8d_batch_to_av(
    emotion_matrix: Sequence[Sequence[float]] | np.ndarray,
    *,
    normalize_weights: bool = True,
    eps: float = 1e-8,
) -> np.ndarray:
    matrix = np.asarray(emotion_matrix, dtype=np.float32)
    if matrix.ndim != 2 or matrix.shape[1] != len(EMOTION_LABELS):
        msg = f"Expected shape (N, {len(EMOTION_LABELS)}) but got {matrix.shape}"
        raise ValueError(msg)

    clipped = np.clip(matrix, 0.0, 1.0)
    if normalize_weights:
        sums = clipped.sum(axis=1, keepdims=True)
        weights = np.divide(
            clipped,
            sums,
            out=np.zeros_like(clipped),
            where=sums > eps,
        )
    else:
        weights = clipped

    return (weights @ _CIRCUMPLEX_MATRIX).astype(np.float32)


def emotion8d_to_av_dict(
    emotion_vec: Sequence[float] | np.ndarray,
    *,
    normalize_weights: bool = True,
) -> dict[str, float]:
    av = emotion8d_to_av(
        emotion_vec,
        normalize_weights=normalize_weights,
    )
    return {
        CIRCUMPLEX_AXIS_NAMES[0]: float(av[0]),
        CIRCUMPLEX_AXIS_NAMES[1]: float(av[1]),
    }
