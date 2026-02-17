from pathlib import Path

import numpy as np
import torch
from transformers import AutoTokenizer

from emotionbridge.constants import EMOTION_LABELS
from emotionbridge.inference.axes import emotion8d_batch_to_av
from emotionbridge.model import TextEmotionRegressor


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
        model_config = checkpoint.get("model_config", {})
        tokenizer_name = checkpoint.get("tokenizer_name")
        if tokenizer_name is None:
            msg = "tokenizer_name not found in checkpoint"
            raise ValueError(msg)

        self.max_length = int(checkpoint.get("max_length", 128))
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.model = TextEmotionRegressor(
            pretrained_model_name=model_config["pretrained_model_name"],
            bottleneck_dim=model_config.get("bottleneck_dim", 256),
            dropout=model_config.get("dropout", 0.1),
        ).to(self.device)
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
            return np.zeros((0, len(EMOTION_LABELS)), dtype=np.float32)

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
        emotion_matrix = self.encode_batch(texts, batch_size=batch_size)
        return emotion8d_batch_to_av(
            emotion_matrix,
            normalize_weights=normalize_weights,
        )
