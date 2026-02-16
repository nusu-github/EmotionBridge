"""Phase 2 Approach B+ 用カテゴリマッピング。

WRIME 8D 感情ベクトルと emotion2vec+ logits 9D から、
共通6感情 [joy, sadness, surprise, anger, fear, disgust] を抽出し、
コサイン類似度を計算する。
"""

from collections.abc import Sequence

import numpy as np

from emotionbridge.analysis.emotion2vec import EMOTION2VEC_LABELS
from emotionbridge.constants import EMOTION_LABELS

COMMON_EMOTION_LABELS = ["joy", "sadness", "surprise", "anger", "fear", "disgust"]

_WRIME_COMMON_INDICES = tuple(
    EMOTION_LABELS.index(label) for label in COMMON_EMOTION_LABELS
)

_WRIME_TO_EMOTION2VEC = {
    "joy": "happy",
    "sadness": "sad",
    "surprise": "surprised",
    "anger": "angry",
    "fear": "fearful",
    "disgust": "disgusted",
}

_EMOTION2VEC_COMMON_INDICES = tuple(
    EMOTION2VEC_LABELS.index(_WRIME_TO_EMOTION2VEC[label])
    for label in COMMON_EMOTION_LABELS
)


def _to_vector(
    values: Sequence[float] | np.ndarray,
    *,
    expected_dim: int,
    name: str,
) -> np.ndarray:
    vector = np.asarray(values, dtype=np.float32)
    if vector.shape != (expected_dim,):
        msg = f"{name} must be shape ({expected_dim},), got {vector.shape}"
        raise ValueError(msg)
    return vector


def extract_common_from_wrime(emotion_vec: Sequence[float] | np.ndarray) -> np.ndarray:
    """WRIME 8D から共通6感情を抽出する。"""
    vector = _to_vector(emotion_vec, expected_dim=8, name="emotion_vec")
    return vector[np.asarray(_WRIME_COMMON_INDICES, dtype=np.int64)]


def extract_common_from_emotion2vec_logits(
    logits: Sequence[float] | np.ndarray,
) -> np.ndarray:
    """emotion2vec+ logits 9D から共通6感情を抽出する。"""
    vector = _to_vector(logits, expected_dim=9, name="logits")
    return vector[np.asarray(_EMOTION2VEC_COMMON_INDICES, dtype=np.int64)]


def l2_normalize(vector: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """L2正規化。ゼロベクトルに近い場合はゼロベクトルを返す。"""
    norm = float(np.linalg.norm(vector))
    if norm < eps:
        return np.zeros_like(vector)
    return vector / norm


def cosine_similarity_common6(
    emotion_vec: Sequence[float] | np.ndarray,
    logits: Sequence[float] | np.ndarray,
) -> float:
    """共通6感情でのコサイン類似度を返す。"""
    text_common = extract_common_from_wrime(emotion_vec)
    audio_common = extract_common_from_emotion2vec_logits(logits)

    text_norm = l2_normalize(text_common)
    audio_norm = l2_normalize(audio_common)

    return float(np.dot(text_norm, audio_norm))
