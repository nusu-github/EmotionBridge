import json
from pathlib import Path

import torch
from torch import nn

from emotionbridge.constants import (
    CONTROL_PARAM_NAMES,
    JVNV_EMOTION_LABELS,
    NUM_CONTROL_PARAMS,
    NUM_JVNV_EMOTIONS,
)
import numpy as np


class ParameterGenerator(nn.Module):
    """6D感情確率ベクトルを5D制御パラメータへ写像する（NN版）。

    テキスト埋め込みを入力に追加する際にはこちらを拡張する。
    デフォルトはDeterministicMixer。
    """

    def __init__(
        self,
        num_emotions: int = NUM_JVNV_EMOTIONS,
        hidden_dim: int = 64,
        num_params: int = NUM_CONTROL_PARAMS,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        self.fc1 = nn.Linear(num_emotions, hidden_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_dim, num_params)
        self.tanh = nn.Tanh()

    def forward(self, emotion_probs: torch.Tensor) -> torch.Tensor:
        x = self.fc1(emotion_probs)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return self.tanh(x)


class DeterministicMixer(nn.Module):
    """6D感情確率ベクトルを教師表との線形混合で5D制御パラメータへ変換する。

    params = tanh(emotion_probs @ teacher_matrix)

    教師データが感情ごとの固定推奨パラメータ（中央値）である場合、
    NNが学習する最適解は本質的にこの線形混合に収束する。
    中間層を排除することで過学習・再現性コストを排除する。
    """

    teacher_matrix: torch.Tensor

    def __init__(self, teacher_matrix: torch.Tensor) -> None:
        super().__init__()
        self.register_buffer("teacher_matrix", teacher_matrix)
        self.tanh = nn.Tanh()

    @classmethod
    def from_numpy(cls, matrix: np.ndarray) -> "DeterministicMixer":
        return cls(torch.tensor(matrix, dtype=torch.float32))

    @classmethod
    def from_json(cls, path: str | Path) -> "DeterministicMixer":
        """教師表JSON（recommended_params.json形式）からロードする。"""
        with Path(path).open("r", encoding="utf-8") as f:
            payload = json.load(f)

        rows = payload["table"]
        row_by_emotion = {r["emotion"]: r for r in rows}
        matrix: list[list[float]] = []
        for emotion in JVNV_EMOTION_LABELS:
            row = row_by_emotion[emotion]
            matrix.append([float(row[f"ctrl_{name}"]) for name in CONTROL_PARAM_NAMES])

        return cls(torch.tensor(matrix, dtype=torch.float32))

    def forward(self, emotion_probs: torch.Tensor) -> torch.Tensor:
        return self.tanh(emotion_probs @ self.teacher_matrix)
