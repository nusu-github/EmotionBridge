from pathlib import Path

import numpy as np
import torch
from transformers import AutoTokenizer

from emotionbridge.constants import JVNV_EMOTION_LABELS
from emotionbridge.model import TextEmotionClassifier


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

        if model_type != "classifier":
            msg = (
                f"Unsupported model_type '{model_type}'. "
                "Only classifier checkpoints are supported."
            )
            raise ValueError(msg)

        pretrained_model_name = model_config.get(
            "pretrained_model_name",
            tokenizer_name,
        )
        self.max_length = int(checkpoint.get("max_length", 128))
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

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

        self._label_names = list(checkpoint.get("emotion_labels", default_labels))
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.eval()

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
                preds = self.model.predict_proba(**encoded)
                outputs.append(preds.detach().cpu().numpy())

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
