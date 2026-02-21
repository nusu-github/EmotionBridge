from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
from transformers import pipeline

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
        if not self.checkpoint_path.is_dir():
            msg = f"Checkpoint path must be a directory: {self.checkpoint_path}"
            raise ValueError(msg)

        self.device = torch.device(
            "cuda" if (device == "cuda" and torch.cuda.is_available()) else "cpu",
        )
        pipeline_device = 0 if self.device.type == "cuda" else -1

        self._pipeline = pipeline(
            task="text-classification",
            model=str(self.checkpoint_path),
            tokenizer=str(self.checkpoint_path),
            top_k=None,
            device=pipeline_device,
        )

        self.model = self._pipeline.model
        tokenizer = self._pipeline.tokenizer
        assert tokenizer is not None
        self.tokenizer = tokenizer

        # ラベル名のリストを保持（JVNV の順序を期待）
        id2label = getattr(self.model.config, "id2label", None)
        if isinstance(id2label, dict) and id2label:

            def _sort_key(item: tuple[Any, str]) -> tuple[int, int | str]:
                key, _value = item
                if isinstance(key, int):
                    return (0, key)
                if isinstance(key, str) and key.isdigit():
                    return (0, int(key))
                return (1, str(key))

            sorted_items = sorted(id2label.items(), key=_sort_key)
            self._label_names = [str(label) for _index, label in sorted_items]
        else:
            self._label_names = list(JVNV_EMOTION_LABELS)

        self.max_length = getattr(self.model.config, "max_position_embeddings", 512)

    def encode(self, text: str) -> np.ndarray:
        result = self.encode_batch([text])
        return result[0]

    def _scores_to_vector(self, score_items: list[dict[str, Any]]) -> np.ndarray:
        score_by_label: dict[str, float] = {}
        for item in score_items:
            label = str(item.get("label", ""))
            score = float(item.get("score", 0.0))
            score_by_label[label] = score

        missing_labels = [label for label in self._label_names if label not in score_by_label]
        if not missing_labels:
            return np.asarray(
                [score_by_label[label] for label in self._label_names], dtype=np.float32
            )

        if all(label.startswith("LABEL_") for label in score_by_label):
            sorted_pairs = sorted(
                score_by_label.items(),
                key=lambda item: int(item[0].split("_")[-1]),
            )
            if len(sorted_pairs) == self.num_emotions:
                return np.asarray([score for _label, score in sorted_pairs], dtype=np.float32)

        msg = f"Classifier output is missing labels: {missing_labels}"
        raise ValueError(msg)

    def encode_batch(self, texts: list[str], batch_size: int = 32) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.num_emotions), dtype=np.float32)

        raw_results = self._pipeline(
            texts,
            batch_size=batch_size,
            truncation=True,
            max_length=self.max_length,
        )

        if raw_results and isinstance(raw_results[0], dict):
            raw_results = [raw_results]

        outputs: list[np.ndarray] = []
        for sample_result in raw_results:
            if isinstance(sample_result, dict):
                score_items = [sample_result]
            else:
                score_items = list(sample_result)
            outputs.append(self._scores_to_vector(score_items))

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
