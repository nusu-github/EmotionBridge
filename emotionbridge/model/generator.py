import json
from pathlib import Path

import numpy as np
import torch
from huggingface_hub import PyTorchModelHubMixin
from torch import nn

from emotionbridge.constants import (
    CONTROL_PARAM_NAMES,
    JVNV_EMOTION_LABELS,
    NUM_CONTROL_PARAMS,
    NUM_JVNV_EMOTIONS,
)


class DeterministicMixer(
    nn.Module,
    PyTorchModelHubMixin,
    library_name="emotionbridge",
    tags=["emotion-tts", "deterministic-mixer"],
):
    """6D感情確率ベクトルを教師表との線形混合で5D制御パラメータへ変換する。

    params = tanh(emotion_probs @ teacher_matrix)

    教師データが感情ごとの固定推奨パラメータ（中央値）である場合、
    NNが学習する最適解は本質的にこの線形混合に収束する。
    中間層を排除することで過学習・再現性コストを排除する。
    """

    teacher_matrix: torch.Tensor

    def __init__(self, teacher_matrix_list: list[list[float]]) -> None:
        super().__init__()
        if len(teacher_matrix_list) != NUM_JVNV_EMOTIONS:
            msg = (
                f"teacher_matrix_list must have {NUM_JVNV_EMOTIONS} rows, "
                f"got {len(teacher_matrix_list)}"
            )
            raise ValueError(msg)

        row_lengths = {len(row) for row in teacher_matrix_list}
        if row_lengths != {NUM_CONTROL_PARAMS}:
            msg = f"Each teacher matrix row must have {NUM_CONTROL_PARAMS} values"
            raise ValueError(msg)

        teacher_matrix = torch.tensor(teacher_matrix_list, dtype=torch.float32)
        self.register_buffer("teacher_matrix", teacher_matrix)
        self.tanh = nn.Tanh()

    @classmethod
    def from_numpy(cls, matrix: np.ndarray) -> "DeterministicMixer":
        return cls(matrix.tolist())

    @classmethod
    def from_json(cls, path: str | Path) -> "DeterministicMixer":
        """教師表JSON（recommended_params.json形式）からロードする。"""
        with Path(path).open("r", encoding="utf-8") as f:
            payload = json.load(f)

        rows = payload["table"]
        row_by_emotion = {r["emotion"]: r for r in rows}
        matrix_list: list[list[float]] = []
        for emotion in JVNV_EMOTION_LABELS:
            row = row_by_emotion[emotion]
            matrix_list.append([float(row[f"ctrl_{name}"]) for name in CONTROL_PARAM_NAMES])

        return cls(matrix_list)

    def forward(self, emotion_probs: torch.Tensor) -> torch.Tensor:
        return self.tanh(emotion_probs @ self.teacher_matrix)
