from collections.abc import Iterable

import torch
from torch import nn
from transformers import AutoModel

from emotionbridge.constants import NUM_JVNV_EMOTIONS


class TextEmotionClassifier(nn.Module):
    def __init__(
        self,
        pretrained_model_name: str,
        num_classes: int = NUM_JVNV_EMOTIONS,
        bottleneck_dim: int = 256,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.encoder = AutoModel.from_pretrained(pretrained_model_name)
        hidden_size = int(self.encoder.config.hidden_size)

        self.dropout1 = nn.Dropout(dropout)
        self.fc1 = nn.Linear(hidden_size, bottleneck_dim)
        self.relu = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)
        self.fc2 = nn.Linear(bottleneck_dim, num_classes)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        token_type_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if token_type_ids is None:
            outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        else:
            outputs = self.encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
            )

        cls_embedding = outputs.last_hidden_state[:, 0, :]
        x = self.dropout1(cls_embedding)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout2(x)
        return self.fc2(x)

    def predict_proba(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        token_type_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        logits = self.forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )
        return torch.softmax(logits, dim=-1)

    def head_parameters(self) -> Iterable[torch.nn.Parameter]:
        modules = [
            self.dropout1,
            self.fc1,
            self.relu,
            self.dropout2,
            self.fc2,
        ]
        for module in modules:
            yield from module.parameters()
