from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import numpy as np
import torch
from torch import nn

from emotionbridge.constants import CONTROL_PARAM_NAMES, JVNV_EMOTION_LABELS
from emotionbridge.inference.encoder import EmotionEncoder
from emotionbridge.model import DeterministicMixer
from emotionbridge.tts.adapter import VoicevoxAdapter
from emotionbridge.tts.types import ControlVector
from emotionbridge.tts.voicevox_client import VoicevoxClient

if TYPE_CHECKING:
    from emotionbridge.dsp import EmotionDSPMapper, EmotionDSPProcessor


@dataclass(frozen=True, slots=True)
class SynthesisResult:
    audio_bytes: bytes
    audio_path: Path | None
    emotion_probs: dict[str, float]
    dominant_emotion: str
    control_params: dict[str, float]
    dsp_params: dict[str, float] | None
    dsp_applied: bool
    dsp_seed: int | None
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
    """テキストから感情音声を生成する統合パイプライン。"""

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
        dsp_enabled: bool = False,
        dsp_mapper: EmotionDSPMapper | None = None,
        dsp_processor: EmotionDSPProcessor | None = None,
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
        self._dsp_enabled = dsp_enabled
        self._dsp_mapper = dsp_mapper
        self._dsp_processor = dsp_processor

    async def close(self) -> None:
        await self._voicevox.close()

    def _classifier_probs_common6(self, text: str) -> np.ndarray:
        probs = self._classifier.encode(text)
        label_to_index = {label: idx for idx, label in enumerate(self._classifier.label_names)}

        missing = [label for label in JVNV_EMOTION_LABELS if label not in label_to_index]
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

    @staticmethod
    def _build_dsp_seed(
        *,
        text: str,
        style_id: int,
        dominant_emotion: str,
        dsp_params: dict[str, float],
    ) -> int:
        payload = json.dumps(
            {
                "text": text,
                "style_id": style_id,
                "dominant_emotion": dominant_emotion,
                "dsp_params": dsp_params,
            },
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        digest = hashlib.blake2b(payload.encode("utf-8"), digest_size=8).digest()
        return int.from_bytes(digest[:4], byteorder="big", signed=False)

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

        dsp_params: dict[str, float] | None = None
        dsp_applied = False
        dsp_seed: int | None = None

        if self._dsp_enabled and not is_fallback:
            if self._dsp_mapper is None or self._dsp_processor is None:
                msg = "DSP is enabled but mapper or processor is not initialized"
                raise RuntimeError(msg)

            dsp_vector = self._dsp_mapper.generate(probs_common6)
            dsp_params = dsp_vector.to_dict()
            dsp_seed = self._build_dsp_seed(
                text=text,
                style_id=style_id,
                dominant_emotion=dominant_emotion,
                dsp_params=dsp_params,
            )

            try:
                audio_bytes = self._dsp_processor.process_bytes(audio_bytes, dsp_vector, dsp_seed)
            except Exception as exc:
                msg = f"DSP processing failed: {exc}"
                raise RuntimeError(msg) from exc
            dsp_applied = True

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
            dsp_params=dsp_params,
            dsp_applied=dsp_applied,
            dsp_seed=dsp_seed,
            style_id=style_id,
            style_name=style_name,
            confidence=confidence,
            is_fallback=is_fallback,
            metadata={
                "character": self._character,
                "fallback_threshold": float(self._fallback_threshold),
                "dsp_enabled": bool(self._dsp_enabled),
            },
        )


def _load_generator_model_dir(
    model_dir: str | Path,
    *,
    device: str,
) -> tuple[nn.Module, torch.device]:
    """ジェネレータモデルディレクトリからモデルをロードする。"""
    path = Path(model_dir)
    if path.suffix == ".pt":
        msg = (
            "Legacy .pt generator checkpoints are not supported. "
            "Pass a model directory created by save_pretrained."
        )
        raise ValueError(msg)

    if not path.exists():
        msg = f"generator model directory not found: {path}"
        raise FileNotFoundError(msg)

    if not path.is_dir():
        msg = f"generator model path must be a directory: {path}"
        raise ValueError(msg)

    device_obj = torch.device(
        "cuda" if (device == "cuda" and torch.cuda.is_available()) else "cpu",
    )

    config_path = path / "config.json"
    if not config_path.exists():
        msg = f"config.json not found in generator model directory: {path}"
        raise FileNotFoundError(msg)

    with config_path.open("r", encoding="utf-8") as file:
        model_config = json.load(file)

    if not isinstance(model_config, dict):
        msg = f"Invalid generator config in {config_path}: expected JSON object"
        raise ValueError(msg)

    if "teacher_matrix_list" not in model_config:
        msg = (
            "Invalid generator config: expected DeterministicMixer checkpoint "
            "with 'teacher_matrix_list' in config.json"
        )
        raise ValueError(msg)

    model: nn.Module = DeterministicMixer.from_pretrained(str(path))

    model = model.to(device_obj)
    model.eval()
    return model, device_obj


async def create_pipeline(
    *,
    classifier_checkpoint: str | Path,
    generator_model_dir: str | Path,
    style_mapping: str | Path,
    voicevox_url: str,
    character: str,
    fallback_threshold: float,
    device: str,
    enable_dsp: bool = False,
    dsp_features_path: str | Path = "artifacts/prosody/v01/jvnv_egemaps_normalized.parquet",
    dsp_f0_extractor: str = "dio",
) -> EmotionBridgePipeline:
    """EmotionBridgePipeline を構築するファクトリ関数。"""
    classifier = EmotionEncoder(str(classifier_checkpoint), device=device)
    if not classifier.is_classifier:
        msg = "bridge requires a classifier checkpoint for --classifier-checkpoint"
        raise ValueError(msg)

    generator, generator_device = _load_generator_model_dir(
        generator_model_dir,
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
    dsp_mapper: EmotionDSPMapper | None = None
    dsp_processor: EmotionDSPProcessor | None = None

    if enable_dsp:
        from emotionbridge.dsp import EmotionDSPMapper, EmotionDSPProcessor
        from emotionbridge.dsp.config import DSPProcessorConfig

        if dsp_f0_extractor not in {"dio", "harvest"}:
            msg = f"Unsupported dsp_f0_extractor: {dsp_f0_extractor}"
            raise ValueError(msg)

        dsp_mapper = EmotionDSPMapper(features_path=dsp_features_path)
        dsp_processor = EmotionDSPProcessor(
            DSPProcessorConfig(
                f0_extractor=cast("Any", dsp_f0_extractor),
            ),
        )

    return EmotionBridgePipeline(
        classifier=classifier,
        generator=generator,
        generator_device=generator_device,
        style_selector=style_selector,
        voicevox_client=voicevox_client,
        adapter=adapter,
        character=character,
        fallback_threshold=fallback_threshold,
        dsp_enabled=enable_dsp,
        dsp_mapper=dsp_mapper,
        dsp_processor=dsp_processor,
    )
