from __future__ import annotations

import torch
from torch import nn

from emotionbridge.constants import NUM_CONTROL_PARAMS, NUM_JVNV_EMOTIONS


class ParameterGenerator(nn.Module):
    """6D感情確率ベクトルを5D制御パラメータへ写像する。"""

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
