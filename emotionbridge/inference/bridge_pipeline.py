from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

from emotionbridge.constants import CONTROL_PARAM_NAMES, JVNV_EMOTION_LABELS
from emotionbridge.inference.encoder import EmotionEncoder
from emotionbridge.model import DeterministicMixer, ParameterGenerator
from emotionbridge.tts.adapter import VoicevoxAdapter
from emotionbridge.tts.types import ControlVector
from emotionbridge.tts.voicevox_client import VoicevoxClient


@dataclass(frozen=True, slots=True)
class SynthesisResult:
    audio_bytes: bytes
    audio_path: Path | None
    emotion_probs: dict[str, float]
    dominant_emotion: str
    control_params: dict[str, float]
    style_id: int
    style_name: str
    confidence: float
    is_fallback: bool
    metadata: dict[str, Any]


class RuleBasedStyleSelector:
    def __init__(self, mapping_path: str | Path) -> None:
        path = Path(mapping_path)
        if not path.exists():
            msg = f"style mapping not found: {path}"
            raise FileNotFoundError(msg)

        import json

        with path.open("r", encoding="utf-8") as file:
            payload = json.load(file)

        characters = payload.get("characters")
        if not isinstance(characters, dict):
            msg = "Invalid style mapping: 'characters' must be dict"
            raise ValueError(msg)

        self._characters = characters

    def select(
        self,
        emotion_probs: dict[str, float],
        character: str,
    ) -> tuple[int, str]:
        if character not in self._characters:
            msg = f"character '{character}' is not found in style mapping"
            raise KeyError(msg)

        dominant = max(emotion_probs, key=lambda key: emotion_probs[key])
        mapping = self._characters[character].get("mapping", {})
        selected = mapping.get(dominant)
        if selected is None:
            return self.default_style(character)

        return int(selected["style_id"]), str(selected["style_name"])

    def default_style(self, character: str) -> tuple[int, str]:
        if character not in self._characters:
            msg = f"character '{character}' is not found in style mapping"
            raise KeyError(msg)

        default_style = self._characters[character].get("default_style")
        if not isinstance(default_style, dict):
            msg = f"default_style not found for character: {character}"
            raise ValueError(msg)

        return int(default_style["style_id"]), str(default_style["style_name"])


class EmotionBridgePipeline:
    def __init__(
        self,
        *,
        classifier: EmotionEncoder,
        generator: nn.Module,
        generator_device: torch.device,
        style_selector: RuleBasedStyleSelector,
        voicevox_client: VoicevoxClient,
        adapter: VoicevoxAdapter,
        character: str,
        fallback_threshold: float,
    ) -> None:
        if not classifier.is_classifier:
            msg = "EmotionBridgePipeline requires classifier checkpoint"
            raise ValueError(msg)

        self._classifier = classifier
        self._generator = generator
        self._generator_device = generator_device
        self._style_selector = style_selector
        self._voicevox = voicevox_client
        self._adapter = adapter
        self._character = character
        self._fallback_threshold = fallback_threshold

    async def close(self) -> None:
        await self._voicevox.close()

    def _classifier_probs_common6(self, text: str) -> np.ndarray:
        probs = self._classifier.encode(text)
        label_to_index = {
            label: idx for idx, label in enumerate(self._classifier.label_names)
        }

        missing = [
            label for label in JVNV_EMOTION_LABELS if label not in label_to_index
        ]
        if missing:
            msg = f"Classifier labels do not cover JVNV common6: missing={missing}"
            raise ValueError(msg)

        return np.array(
            [probs[label_to_index[label]] for label in JVNV_EMOTION_LABELS],
            dtype=np.float32,
        )

    def _predict_control(self, probs_common6: np.ndarray) -> ControlVector:
        with torch.no_grad():
            tensor = torch.tensor(
                probs_common6,
                dtype=torch.float32,
                device=self._generator_device,
            ).unsqueeze(0)
            pred = self._generator(tensor).squeeze(0).detach().cpu().numpy()
        pred = np.clip(pred, -1.0, 1.0)
        return ControlVector.from_numpy(pred.astype(np.float32))

    async def synthesize(
        self,
        text: str,
        *,
        output_path: str | Path | None = None,
    ) -> SynthesisResult:
        probs_common6 = self._classifier_probs_common6(text)
        emotion_probs = {
            label: float(value)
            for label, value in zip(JVNV_EMOTION_LABELS, probs_common6, strict=True)
        }
        dominant_emotion = max(emotion_probs, key=lambda key: emotion_probs[key])
        confidence = float(max(emotion_probs.values()))

        is_fallback = confidence < self._fallback_threshold
        if is_fallback:
            style_id, style_name = self._style_selector.default_style(self._character)
            control = ControlVector()
        else:
            style_id, style_name = self._style_selector.select(
                emotion_probs,
                self._character,
            )
            control = self._predict_control(probs_common6)

        query = await self._voicevox.audio_query(text=text, speaker_id=style_id)
        modified = self._adapter.apply(query, control)
        audio_bytes = await self._voicevox.synthesis(modified, speaker_id=style_id)

        saved_path: Path | None = None
        if output_path is not None:
            saved_path = Path(output_path)
            saved_path.parent.mkdir(parents=True, exist_ok=True)
            saved_path.write_bytes(audio_bytes)

        return SynthesisResult(
            audio_bytes=audio_bytes,
            audio_path=saved_path,
            emotion_probs=emotion_probs,
            dominant_emotion=dominant_emotion,
            control_params={
                name: float(value)
                for name, value in zip(
                    CONTROL_PARAM_NAMES,
                    control.to_numpy(),
                    strict=True,
                )
            },
            style_id=style_id,
            style_name=style_name,
            confidence=confidence,
            is_fallback=is_fallback,
            metadata={
                "character": self._character,
                "fallback_threshold": float(self._fallback_threshold),
            },
        )


def _load_generator_checkpoint(
    checkpoint_path: str | Path,
    *,
    device: str,
) -> tuple[nn.Module, torch.device]:
    path = Path(checkpoint_path)
    if not path.exists():
        msg = f"generator checkpoint not found: {path}"
        raise FileNotFoundError(msg)

    device_obj = torch.device(
        "cuda" if (device == "cuda" and torch.cuda.is_available()) else "cpu",
    )

    checkpoint = torch.load(path, map_location=device_obj)
    model_type = checkpoint.get("model_type", "parameter_generator")

    model_state_dict = checkpoint.get("model_state_dict")
    if model_state_dict is None:
        msg = "model_state_dict not found in generator checkpoint"
        raise ValueError(msg)

    model: nn.Module
    if model_type == "deterministic_mixer":
        recommended = checkpoint.get("recommended_params")
        if recommended is None:
            msg = "recommended_params not found in deterministic_mixer checkpoint"
            raise ValueError(msg)
        teacher_matrix = torch.tensor(recommended, dtype=torch.float32)
        model = DeterministicMixer(teacher_matrix).to(device_obj)
        model.load_state_dict(model_state_dict, strict=True)
    else:
        model_config = (
            checkpoint.get("config", {}).get("model", {})
            if isinstance(checkpoint, dict)
            else {}
        )
        hidden_dim = int(model_config.get("hidden_dim", 64))
        dropout = float(model_config.get("dropout", 0.3))

        model = ParameterGenerator(hidden_dim=hidden_dim, dropout=dropout).to(
            device_obj,
        )
        current_state_dict = model.state_dict()
        compatible_state_dict = {
            key: value
            for key, value in model_state_dict.items()
            if key in current_state_dict
            and current_state_dict[key].shape == value.shape
        }
        model.load_state_dict(compatible_state_dict, strict=False)

    model.eval()
    return model, device_obj


async def create_pipeline(
    *,
    classifier_checkpoint: str | Path,
    generator_checkpoint: str | Path,
    style_mapping: str | Path,
    voicevox_url: str,
    character: str,
    fallback_threshold: float,
    device: str,
) -> EmotionBridgePipeline:
    classifier = EmotionEncoder(str(classifier_checkpoint), device=device)
    if not classifier.is_classifier:
        msg = "bridge requires a classifier checkpoint for --classifier-checkpoint"
        raise ValueError(msg)

    generator, generator_device = _load_generator_checkpoint(
        generator_checkpoint,
        device=device,
    )

    style_selector = RuleBasedStyleSelector(style_mapping)
    voicevox_client = VoicevoxClient(base_url=voicevox_url)

    health = await voicevox_client.health_check()
    if not health:
        await voicevox_client.close()
        msg = f"VOICEVOX Engine に接続できません: {voicevox_url}"
        raise ConnectionError(msg)

    adapter = VoicevoxAdapter()

    return EmotionBridgePipeline(
        classifier=classifier,
        generator=generator,
        generator_device=generator_device,
        style_selector=style_selector,
        voicevox_client=voicevox_client,
        adapter=adapter,
        character=character,
        fallback_threshold=fallback_threshold,
    )
