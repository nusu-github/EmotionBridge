"""Phase 1暫定: 感情ベクトル→ControlVector変換のヒューリスティックマッパー。

学習済みパラメータ生成器が存在しないPhase 1時点での暫定実装。
感情ベクトルの各要素を重みとして、事前定義テンプレートの加重平均を計算し
ControlVectorを返す。Phase 3以降で学習済みモデルに置き換え予定。
"""

import numpy as np

from emotionbridge.constants import (
    EMOTION_LABELS,
    NUM_CONTROL_PARAMS,
)
from emotionbridge.tts.types import ControlVector

# 8感情それぞれに対応する制御パラメータテンプレート
EMOTION_TO_CONTROL: dict[str, ControlVector] = {
    "joy": ControlVector(
        pitch_shift=0.4,
        pitch_range=0.5,
        speed=0.2,
        energy=0.4,
        pause_weight=-0.2,
    ),
    "sadness": ControlVector(
        pitch_shift=-0.3,
        pitch_range=-0.3,
        speed=-0.3,
        energy=-0.3,
        pause_weight=0.3,
    ),
    "anticipation": ControlVector(
        pitch_shift=0.1,
        pitch_range=0.2,
        speed=0.1,
        energy=0.1,
        pause_weight=-0.1,
    ),
    "surprise": ControlVector(
        pitch_shift=0.5,
        pitch_range=0.6,
        speed=0.3,
        energy=0.5,
        pause_weight=-0.3,
    ),
    "anger": ControlVector(
        pitch_shift=0.2,
        pitch_range=0.3,
        speed=0.2,
        energy=0.6,
        pause_weight=-0.2,
    ),
    "fear": ControlVector(
        pitch_shift=0.2,
        pitch_range=-0.2,
        speed=0.2,
        energy=-0.2,
        pause_weight=0.2,
    ),
    "disgust": ControlVector(
        pitch_shift=-0.2,
        pitch_range=-0.1,
        speed=-0.1,
        energy=0.1,
        pause_weight=0.1,
    ),
    "trust": ControlVector(
        pitch_shift=0.0,
        pitch_range=0.1,
        speed=-0.1,
        energy=0.0,
        pause_weight=0.0,
    ),
}


def heuristic_map(emotion_vec: np.ndarray) -> ControlVector:
    """8D感情ベクトルから支配的感情に基づくControlVectorを返す。

    感情ベクトルの各要素を重みとして、事前定義テンプレートの加重平均を計算する。
    全感情が弱い場合（max < 0.1）はニュートラル（全0）を返す。

    Args:
        emotion_vec: shape (8,) の感情ベクトル。順序はEMOTION_LABELS準拠。
    Returns:
        加重平均で算出されたControlVector。

    """
    if emotion_vec.shape != (8,):
        msg = f"Expected shape (8,), got {emotion_vec.shape}"
        raise ValueError(msg)

    if emotion_vec.max() < 0.1:
        return ControlVector()  # ニュートラル

    weighted = np.zeros(NUM_CONTROL_PARAMS, dtype=np.float32)
    for i, label in enumerate(EMOTION_LABELS):
        template = EMOTION_TO_CONTROL[label].to_numpy()
        weighted += emotion_vec[i] * template

    # 正規化して [-1, +1] にクリップ
    weight_sum = emotion_vec.sum()
    if weight_sum > 0:
        weighted /= weight_sum
    weighted = np.clip(weighted, -1.0, 1.0)

    return ControlVector.from_numpy(weighted)
