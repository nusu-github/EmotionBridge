from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from emotionbridge.constants import JVNV_EMOTION_LABELS


class EmotionEncoder:
    """テキストから感情確率ベクトルを抽出するエンコーダ。

    Trainer で保存された HF 形式のモデルディレクトリからロードします。
    """

    def __init__(self, checkpoint_path: str, device: str = "cuda") -> None:
        self.checkpoint_path = Path(checkpoint_path)
        if not self.checkpoint_path.exists():
            msg = f"Checkpoint not found: {self.checkpoint_path}"
            raise FileNotFoundError(msg)

        self.device = torch.device(
            "cuda" if (device == "cuda" and torch.cuda.is_available()) else "cpu",
        )

        # HF 標準の方法でモデルとトークナイザをロード
        # モデルディレクトリに保存された config.json から num_labels, id2label 等が自動復元される
        self.model = AutoModelForSequenceClassification.from_pretrained(str(self.checkpoint_path))
        self.model.to(self.device)
        self.model.eval()

        self.tokenizer = AutoTokenizer.from_pretrained(str(self.checkpoint_path))

        # ラベル名のリストを保持（JVNV の順序を期待）
        if self.model.config.id2label:
            self._label_names = [
                self.model.config.id2label[i] for i in range(len(self.model.config.id2label))
            ]
        else:
            self._label_names = JVNV_EMOTION_LABELS

        self.max_length = getattr(self.model.config, "max_position_embeddings", 512)

    def encode(self, text: str) -> np.ndarray:
        result = self.encode_batch([text])
        return result[0]

    def encode_batch(self, texts: list[str], batch_size: int = 32) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.num_emotions), dtype=np.float32)

        outputs: list[np.ndarray] = []
        with torch.no_grad():
            for start in range(0, len(texts), batch_size):
                batch_texts = texts[start : start + batch_size]
                encoded = self.tokenizer(
                    batch_texts,
                    padding=True,
                    truncation=True,
                    max_length=self.max_length,
                    return_tensors="pt",
                )
                encoded = {k: v.to(self.device) for k, v in encoded.items()}
                logits = self.model(**encoded).logits
                probs = torch.softmax(logits, dim=-1)
                outputs.append(probs.detach().cpu().numpy())

        return np.vstack(outputs).astype(np.float32)

    @property
    def is_classifier(self) -> bool:
        return True

    @property
    def label_names(self) -> list[str]:
        return list(self._label_names)

    @property
    def num_emotions(self) -> int:
        return len(self._label_names)
