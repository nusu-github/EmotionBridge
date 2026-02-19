from __future__ import annotations

import torch
from torch import nn
from transformers import AutoModelForSequenceClassification

from emotionbridge.constants import NUM_JVNV_EMOTIONS


class TextEmotionClassifier(nn.Module):
    """AutoModelForSequenceClassification を使用したテキスト感情分類器。"""

    def __init__(
        self,
        pretrained_model_name: str,
        num_classes: int = NUM_JVNV_EMOTIONS,
        dropout: float | None = None,
        id2label: dict[int, str] | None = None,
        label2id: dict[str, int] | None = None,
    ) -> None:
        super().__init__()
        # AutoModelForSequenceClassification を利用して標準的な分類ヘッドを構築
        self.model = AutoModelForSequenceClassification.from_pretrained(
            pretrained_model_name,
            num_labels=num_classes,
            id2label=id2label,
            label2id=label2id,
        )
        if dropout is not None and hasattr(self.model.config, "hidden_dropout_prob"):
            self.model.config.hidden_dropout_prob = dropout

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        token_type_ids: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
    ):
        return self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            labels=labels,
        )

    def predict_proba(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        token_type_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        outputs = self.forward(input_ids, attention_mask, token_type_ids)
        return torch.softmax(outputs.logits, dim=-1)

    def get_encoder_parameters(self):
        """バックボーン（BERT等）のパラメータを返す。"""
        # AutoModelForSequenceClassification の構造に依存するが、
        # 通常ベースモデルは最初の属性（bert, roberta等）として保持される
        base_model_name = self.model.base_model_prefix
        if hasattr(self.model, base_model_name):
            return getattr(self.model, base_model_name).parameters()
        return self.model.parameters()

    def get_head_parameters(self):
        """分類ヘッドのパラメータを返す。"""
        if hasattr(self.model, "classifier"):
            return self.model.classifier.parameters()
        # その他、モデルアーキテクチャに応じた分類層の名称
        for name in ["score", "fc"]:
            if hasattr(self.model, name):
                return getattr(self.model, name).parameters()
        return []
