from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
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

        if str(payload.get("version")) != "2.0":
            msg = (
                "Unsupported style mapping schema version. "
                "Run build_style_mapping.py to regenerate style_mapping.json (v2.0)."
            )
            raise ValueError(msg)

        selection_policy = payload.get("selection_policy")
        if selection_policy not in {"prob_distance_only", "prob_distance_with_control_compat"}:
            msg = (
                "Invalid style mapping: 'selection_policy' must be "
                "'prob_distance_only' or 'prob_distance_with_control_compat'"
            )
            raise ValueError(msg)

        scoring = payload.get("scoring")
        if not isinstance(scoring, dict):
            msg = "Invalid style mapping: 'scoring' must be dict"
            raise ValueError(msg)

        characters = payload.get("characters")
        if not isinstance(characters, dict):
            msg = "Invalid style mapping: 'characters' must be dict"
            raise ValueError(msg)

        styles_used_raw = payload.get("styles_used")
        if not isinstance(styles_used_raw, list) or not styles_used_raw:
            msg = "Invalid style mapping: 'styles_used' must be non-empty list"
            raise ValueError(msg)
        style_ids = sorted({int(style_id) for style_id in styles_used_raw})
        style_index_by_id = {style_id: idx for idx, style_id in enumerate(style_ids)}

        artifacts = payload.get("artifacts")
        if not isinstance(artifacts, dict):
            msg = "Invalid style mapping: 'artifacts' must be dict"
            raise ValueError(msg)
        if "style_control_compatibility" not in artifacts:
            msg = "Invalid style mapping: 'artifacts.style_control_compatibility' is required"
            raise ValueError(msg)

        self._mapping_dir = path.resolve().parent
        self._characters = characters
        self._selection_policy = str(selection_policy)
        self._compat_lambda = float(scoring.get("lambda", 0.0))
        self._style_ids = style_ids
        self._style_index_by_id = style_index_by_id

        distance_payload = self._load_json_artifact(artifacts.get("distance_matrix"), "distance_matrix")
        distance_nested = distance_payload.get("distance_matrix")
        if not isinstance(distance_nested, dict):
            msg = "Invalid distance matrix artifact: 'distance_matrix' must be dict"
            raise ValueError(msg)

        distance_rows: list[np.ndarray] = []
        for emotion in JVNV_EMOTION_LABELS:
            row = distance_nested.get(emotion)
            if not isinstance(row, dict):
                msg = f"Invalid distance matrix artifact: missing row for emotion '{emotion}'"
                raise ValueError(msg)
            values: list[float] = []
            for style_id in self._style_ids:
                value = row.get(str(style_id))
                if not isinstance(value, (int, float)):
                    msg = (
                        "Invalid distance matrix artifact: "
                        f"missing distance for emotion={emotion}, style_id={style_id}"
                    )
                    raise ValueError(msg)
                values.append(float(value))
            distance_rows.append(np.array(values, dtype=np.float64))
        self._distance_matrix = np.vstack(distance_rows)

        self._style_name_by_id: dict[int, str] = {
            style_id: f"style_{style_id}" for style_id in self._style_ids
        }
        self._style_character_by_id: dict[int, str] = {}
        style_profiles_raw = artifacts.get("style_profiles")
        if isinstance(style_profiles_raw, str) and style_profiles_raw:
            style_profiles_payload = self._load_json_artifact(style_profiles_raw, "style_profiles")
            styles_payload = style_profiles_payload.get("styles")
            if isinstance(styles_payload, dict):
                for style_id in self._style_ids:
                    item = styles_payload.get(str(style_id))
                    if not isinstance(item, dict):
                        continue
                    style_name = item.get("style_name")
                    character = item.get("character")
                    if isinstance(style_name, str) and style_name:
                        self._style_name_by_id[style_id] = style_name
                    if isinstance(character, str) and character:
                        self._style_character_by_id[style_id] = character

        for character, character_payload in self._characters.items():
            if not isinstance(character_payload, dict):
                continue
            mapping = character_payload.get("mapping")
            if isinstance(mapping, dict):
                for item in mapping.values():
                    if not isinstance(item, dict):
                        continue
                    style_id = item.get("style_id")
                    style_name = item.get("style_name")
                    if isinstance(style_id, int) and isinstance(style_name, str) and style_name:
                        self._style_name_by_id[style_id] = style_name
                        self._style_character_by_id.setdefault(style_id, str(character))
            default_style = character_payload.get("default_style")
            if isinstance(default_style, dict):
                style_id = default_style.get("style_id")
                style_name = default_style.get("style_name")
                if isinstance(style_id, int) and isinstance(style_name, str) and style_name:
                    self._style_name_by_id[style_id] = style_name
                    self._style_character_by_id.setdefault(style_id, str(character))

        self._character_style_ids: dict[str, list[int]] = {}
        for character in self._characters:
            character_style_ids = [
                style_id
                for style_id in self._style_ids
                if self._style_character_by_id.get(style_id, character) == character
            ]
            self._character_style_ids[character] = (
                character_style_ids if character_style_ids else list(self._style_ids)
            )

        self._compat_feature_names: list[str] = []
        self._compat_models: dict[int, tuple[np.ndarray, np.ndarray]] = {}
        self._jvnv_centroid_matrix: np.ndarray | None = None
        if self._selection_policy == "prob_distance_with_control_compat":
            compatibility_raw = artifacts.get("style_control_compatibility")
            if not isinstance(compatibility_raw, str) or not compatibility_raw:
                msg = (
                    "selection_policy=prob_distance_with_control_compat requires "
                    "'artifacts.style_control_compatibility'"
                )
                raise ValueError(msg)
            compatibility_payload = self._load_json_artifact(
                compatibility_raw,
                "style_control_compatibility",
            )
            self._load_compatibility_payload(compatibility_payload)

    def _load_json_artifact(self, artifact_path: object, label: str) -> dict[str, Any]:
        if not isinstance(artifact_path, str) or not artifact_path:
            msg = f"Invalid style mapping: artifact path '{label}' is missing"
            raise ValueError(msg)
        path = Path(artifact_path)
        if not path.is_absolute():
            path = (self._mapping_dir / path).resolve()
        if not path.exists():
            msg = f"style mapping artifact not found: {label} -> {path}"
            raise FileNotFoundError(msg)
        with path.open("r", encoding="utf-8") as file:
            payload = json.load(file)
        if not isinstance(payload, dict):
            msg = f"Invalid style mapping artifact: {label} must be JSON object"
            raise ValueError(msg)
        return payload

    def _load_compatibility_payload(self, payload: dict[str, Any]) -> None:
        style_ids_raw = payload.get("style_ids")
        if not isinstance(style_ids_raw, list):
            msg = "Invalid compatibility artifact: 'style_ids' must be list"
            raise ValueError(msg)
        compat_style_ids = sorted({int(style_id) for style_id in style_ids_raw})
        if compat_style_ids != self._style_ids:
            msg = "Invalid compatibility artifact: style_ids mismatch with style_mapping.json"
            raise ValueError(msg)

        feature_names = payload.get("feature_names")
        if not isinstance(feature_names, list) or not feature_names:
            msg = "Invalid compatibility artifact: 'feature_names' must be non-empty list"
            raise ValueError(msg)
        if not all(isinstance(name, str) and name for name in feature_names):
            msg = "Invalid compatibility artifact: feature_names must be non-empty strings"
            raise ValueError(msg)
        self._compat_feature_names = [str(name) for name in feature_names]

        control_names = payload.get("control_names")
        if not isinstance(control_names, list):
            msg = "Invalid compatibility artifact: 'control_names' must be list"
            raise ValueError(msg)
        if [str(name) for name in control_names] != CONTROL_PARAM_NAMES:
            msg = (
                "Invalid compatibility artifact: control_names must match CONTROL_PARAM_NAMES "
                f"({CONTROL_PARAM_NAMES})"
            )
            raise ValueError(msg)

        centroids = payload.get("jvnv_emotion_centroids")
        if not isinstance(centroids, dict):
            msg = "Invalid compatibility artifact: 'jvnv_emotion_centroids' must be dict"
            raise ValueError(msg)
        centroid_rows: list[np.ndarray] = []
        for emotion in JVNV_EMOTION_LABELS:
            row = centroids.get(emotion)
            if not isinstance(row, dict):
                msg = f"Invalid compatibility artifact: centroid row missing for emotion '{emotion}'"
                raise ValueError(msg)
            values: list[float] = []
            for feature in self._compat_feature_names:
                value = row.get(feature)
                if not isinstance(value, (int, float)):
                    msg = (
                        "Invalid compatibility artifact: missing centroid value for "
                        f"emotion={emotion}, feature={feature}"
                    )
                    raise ValueError(msg)
                values.append(float(value))
            centroid_rows.append(np.array(values, dtype=np.float64))
        self._jvnv_centroid_matrix = np.vstack(centroid_rows)

        style_models = payload.get("style_models")
        if not isinstance(style_models, dict):
            msg = "Invalid compatibility artifact: 'style_models' must be dict"
            raise ValueError(msg)
        self._compat_models = {}
        for style_id in self._style_ids:
            model_payload = style_models.get(str(style_id))
            if not isinstance(model_payload, dict):
                msg = f"Invalid compatibility artifact: missing model for style_id={style_id}"
                raise ValueError(msg)
            coef = np.asarray(model_payload.get("coef"), dtype=np.float64)
            intercept = np.asarray(model_payload.get("intercept"), dtype=np.float64).reshape(-1)
            expected_coef_shape = (len(self._compat_feature_names), len(CONTROL_PARAM_NAMES))
            if coef.shape != expected_coef_shape:
                msg = (
                    "Invalid compatibility artifact: coef shape mismatch for style_id="
                    f"{style_id}. expected={expected_coef_shape}, got={coef.shape}"
                )
                raise ValueError(msg)
            if intercept.shape != (len(self._compat_feature_names),):
                msg = (
                    "Invalid compatibility artifact: intercept shape mismatch for style_id="
                    f"{style_id}. expected={(len(self._compat_feature_names),)}, got={intercept.shape}"
                )
                raise ValueError(msg)
            self._compat_models[style_id] = (coef, intercept)

        metric = payload.get("selection_metric")
        if not isinstance(metric, dict):
            msg = "Invalid compatibility artifact: 'selection_metric' must be dict"
            raise ValueError(msg)
        best_lambda = metric.get("best_lambda")
        if not isinstance(best_lambda, (int, float)):
            msg = "Invalid compatibility artifact: 'selection_metric.best_lambda' must be numeric"
            raise ValueError(msg)
        self._compat_lambda = float(best_lambda)

    @staticmethod
    def _normalize_emotion_probs(emotion_probs: dict[str, float]) -> np.ndarray:
        values = np.array(
            [max(0.0, float(emotion_probs.get(emotion, 0.0))) for emotion in JVNV_EMOTION_LABELS],
            dtype=np.float64,
        )
        total = float(np.sum(values))
        if total <= 0.0:
            return np.full(len(JVNV_EMOTION_LABELS), 1.0 / float(len(JVNV_EMOTION_LABELS)))
        return values / total

    @staticmethod
    def _zscore(values: np.ndarray) -> np.ndarray:
        mean = float(np.mean(values))
        std = float(np.std(values, ddof=0))
        if std <= 1.0e-12:
            return np.zeros_like(values)
        return (values - mean) / std

    def select(
        self,
        emotion_probs: dict[str, float],
        character: str,
        control_params: dict[str, float] | None = None,
    ) -> tuple[int, str]:
        if character not in self._characters:
            msg = f"character '{character}' is not found in style mapping"
            raise KeyError(msg)

        candidate_style_ids = self._character_style_ids.get(character, self._style_ids)
        candidate_indices = np.array(
            [self._style_index_by_id[style_id] for style_id in candidate_style_ids],
            dtype=np.int64,
        )

        probs = self._normalize_emotion_probs(emotion_probs)
        base_raw_all = -(probs @ self._distance_matrix)
        base_raw = base_raw_all[candidate_indices]
        final_scores = self._zscore(base_raw)

        if self._selection_policy == "prob_distance_with_control_compat":
            if self._jvnv_centroid_matrix is None:
                msg = "compatibility centroid matrix is not initialized"
                raise RuntimeError(msg)
            if control_params is None:
                msg = (
                    "control_params is required for "
                    "selection_policy=prob_distance_with_control_compat"
                )
                raise ValueError(msg)

            try:
                control_vec = np.array(
                    [float(control_params[name]) for name in CONTROL_PARAM_NAMES],
                    dtype=np.float64,
                )
            except KeyError as exc:
                msg = f"control_params is missing required key: {exc.args[0]}"
                raise ValueError(msg) from exc

            target_feature = probs @ self._jvnv_centroid_matrix
            compat_raw = np.zeros(len(candidate_style_ids), dtype=np.float64)
            for idx, style_id in enumerate(candidate_style_ids):
                model = self._compat_models.get(style_id)
                if model is None:
                    msg = f"compatibility model is missing for style_id={style_id}"
                    raise RuntimeError(msg)
                coef, intercept = model
                pred_feature = coef @ control_vec + intercept
                compat_raw[idx] = -float(np.linalg.norm(target_feature - pred_feature))
            final_scores = final_scores + (self._compat_lambda * self._zscore(compat_raw))

        selected_idx = int(np.argmax(final_scores))
        style_id = int(candidate_style_ids[selected_idx])
        style_name = self._style_name_by_id.get(style_id, f"style_{style_id}")
        return style_id, style_name

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
            control = self._predict_control(probs_common6)
            control_dict = {
                name: float(value)
                for name, value in zip(
                    CONTROL_PARAM_NAMES,
                    control.to_numpy(),
                    strict=True,
                )
            }
            style_id, style_name = self._style_selector.select(
                emotion_probs,
                self._character,
                control_dict,
            )

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
