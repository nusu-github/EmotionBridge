from pathlib import Path

import numpy as np
import torch
from transformers import AutoTokenizer

from emotionbridge.constants import EMOTION_LABELS, JVNV_EMOTION_LABELS
from emotionbridge.inference.axes import emotion8d_batch_to_av
from emotionbridge.model import TextEmotionClassifier, TextEmotionRegressor


class EmotionEncoder:
    def __init__(self, checkpoint_path: str, device: str = "cuda") -> None:
        self.checkpoint_path = Path(checkpoint_path)
        if not self.checkpoint_path.exists():
            msg = f"Checkpoint not found: {self.checkpoint_path}"
            raise FileNotFoundError(msg)

        self.device = torch.device(
            "cuda" if (device == "cuda" and torch.cuda.is_available()) else "cpu",
        )

        checkpoint = torch.load(self.checkpoint_path, map_location=self.device)
        model_type = str(checkpoint.get("model_type", "regressor"))
        model_config = checkpoint.get("model_config", {})
        tokenizer_name = checkpoint.get("tokenizer_name")
        if tokenizer_name is None:
            msg = "tokenizer_name not found in checkpoint"
            raise ValueError(msg)

        pretrained_model_name = model_config.get(
            "pretrained_model_name",
            tokenizer_name,
        )
        self.max_length = int(checkpoint.get("max_length", 128))
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

        self._is_classifier = model_type == "classifier"
        if self._is_classifier:
            default_labels = JVNV_EMOTION_LABELS
            num_classes = int(
                model_config.get(
                    "num_classes",
                    len(checkpoint.get("emotion_labels", default_labels)),
                ),
            )
            self.model = TextEmotionClassifier(
                pretrained_model_name=pretrained_model_name,
                num_classes=num_classes,
                bottleneck_dim=model_config.get("bottleneck_dim", 256),
                dropout=model_config.get("dropout", 0.1),
            ).to(self.device)
        else:
            default_labels = EMOTION_LABELS
            self.model = TextEmotionRegressor(
                pretrained_model_name=pretrained_model_name,
                bottleneck_dim=model_config.get("bottleneck_dim", 256),
                dropout=model_config.get("dropout", 0.1),
            ).to(self.device)

        self._label_names = list(checkpoint.get("emotion_labels", default_labels))
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.eval()

    def encode(self, text: str) -> np.ndarray:
        result = self.encode_batch([text])
        return result[0]

    def encode_av(self, text: str, *, normalize_weights: bool = True) -> np.ndarray:
        result = self.encode_batch_av(
            [text],
            normalize_weights=normalize_weights,
        )
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
                if self._is_classifier:
                    preds = self.model.predict_proba(**encoded)
                else:
                    preds = self.model(**encoded)
                outputs.append(preds.detach().cpu().numpy())

        return np.vstack(outputs).astype(np.float32)

    def encode_batch_av(
        self,
        texts: list[str],
        *,
        batch_size: int = 32,
        normalize_weights: bool = True,
    ) -> np.ndarray:
        if self._is_classifier:
            msg = (
                "encode_av is only available for 8D regressor checkpoints. "
                "Use encode()/encode_batch() for classifier probabilities."
            )
            raise NotImplementedError(msg)

        emotion_matrix = self.encode_batch(texts, batch_size=batch_size)
        return emotion8d_batch_to_av(
            emotion_matrix,
            normalize_weights=normalize_weights,
        )

    @property
    def label_names(self) -> list[str]:
        return list(self._label_names)

    @property
    def num_emotions(self) -> int:
        return len(self._label_names)

    @property
    def is_classifier(self) -> bool:
        return self._is_classifier
