from __future__ import annotations

import unittest
from typing import Any, cast

import numpy as np
import torch

from emotionbridge.constants import JVNV_EMOTION_LABELS
from emotionbridge.dsp.types import DSPControlVector
from emotionbridge.inference.bridge_pipeline import EmotionBridgePipeline


class _FakeClassifier:
    def __init__(self, probs: np.ndarray) -> None:
        self._probs = probs.astype(np.float32)
        self._labels = list(JVNV_EMOTION_LABELS)

    @property
    def is_classifier(self) -> bool:
        return True

    @property
    def label_names(self) -> list[str]:
        return self._labels

    def encode(self, text: str) -> np.ndarray:
        del text
        return self._probs


class _FakeGenerator(torch.nn.Module):
    def forward(self, emotion_probs: torch.Tensor) -> torch.Tensor:
        batch = int(emotion_probs.shape[0])
        return torch.zeros((batch, 5), dtype=torch.float32, device=emotion_probs.device)


class _FakeStyleSelector:
    def __init__(self) -> None:
        self.last_control_params: dict[str, float] | None = None

    def select(
        self,
        emotion_probs: dict[str, float],
        character: str,
        control_params: dict[str, float] | None = None,
    ) -> tuple[int, str]:
        del emotion_probs, character
        self.last_control_params = control_params
        return 7, "ツンツン"

    def default_style(self, character: str) -> tuple[int, str]:
        del character
        return 3, "ノーマル"


class _FakeVoicevoxClient:
    async def audio_query(self, text: str, speaker_id: int) -> dict[str, object]:
        return {"text": text, "speaker_id": speaker_id}

    async def synthesis(self, audio_query: dict[str, object], speaker_id: int) -> bytes:
        del audio_query, speaker_id
        return b"raw-audio"

    async def close(self) -> None:
        return None


class _FakeAdapter:
    def apply(self, audio_query: dict[str, object], control: object) -> dict[str, object]:
        del control
        return audio_query


class _FakeDSPMapper:
    def generate(self, probs_common6: np.ndarray) -> DSPControlVector:
        del probs_common6
        return DSPControlVector(
            jitter_amount=0.2,
            shimmer_amount=0.1,
            aperiodicity_shift=0.1,
            spectral_tilt_shift=-0.1,
        )


class _FakeDSPProcessor:
    def __init__(self, *, should_raise: bool = False) -> None:
        self.should_raise = should_raise
        self.called = False

    def process_bytes(
        self,
        audio_bytes: bytes,
        dsp_params: DSPControlVector,
        seed: int,
    ) -> bytes:
        del audio_bytes, dsp_params, seed
        self.called = True
        if self.should_raise:
            msg = "boom"
            raise ValueError(msg)
        return b"processed-audio"


class TestBridgePipelineDSP(unittest.IsolatedAsyncioTestCase):
    async def test_dsp_applied_when_enabled_and_not_fallback(self) -> None:
        probs = np.array([0.9, 0.02, 0.02, 0.02, 0.02, 0.02], dtype=np.float32)
        dsp_processor = _FakeDSPProcessor()
        style_selector = _FakeStyleSelector()
        pipeline = EmotionBridgePipeline(
            classifier=cast("Any", _FakeClassifier(probs)),
            generator=_FakeGenerator(),
            generator_device=torch.device("cpu"),
            style_selector=cast("Any", style_selector),
            voicevox_client=cast("Any", _FakeVoicevoxClient()),
            adapter=cast("Any", _FakeAdapter()),
            character="zundamon",
            fallback_threshold=0.3,
            dsp_enabled=True,
            dsp_mapper=cast("Any", _FakeDSPMapper()),
            dsp_processor=cast("Any", dsp_processor),
        )

        result = await pipeline.synthesize("怒っています")
        assert dsp_processor.called
        assert result.dsp_applied
        assert result.dsp_params is not None
        assert isinstance(result.dsp_seed, int)
        assert result.audio_bytes == b"processed-audio"
        assert style_selector.last_control_params is not None
        assert set(style_selector.last_control_params.keys()) == {
            "pitch_shift",
            "pitch_range",
            "speed",
            "energy",
            "pause_weight",
        }

    async def test_dsp_skipped_on_fallback(self) -> None:
        probs = np.array([0.2, 0.18, 0.17, 0.16, 0.15, 0.14], dtype=np.float32)
        dsp_processor = _FakeDSPProcessor()
        style_selector = _FakeStyleSelector()
        pipeline = EmotionBridgePipeline(
            classifier=cast("Any", _FakeClassifier(probs)),
            generator=_FakeGenerator(),
            generator_device=torch.device("cpu"),
            style_selector=cast("Any", style_selector),
            voicevox_client=cast("Any", _FakeVoicevoxClient()),
            adapter=cast("Any", _FakeAdapter()),
            character="zundamon",
            fallback_threshold=0.3,
            dsp_enabled=True,
            dsp_mapper=cast("Any", _FakeDSPMapper()),
            dsp_processor=cast("Any", dsp_processor),
        )

        result = await pipeline.synthesize("不明な感情")
        assert not dsp_processor.called
        assert not result.dsp_applied
        assert result.dsp_params is None
        assert result.dsp_seed is None
        assert result.is_fallback
        assert style_selector.last_control_params is None

    async def test_dsp_failure_is_raised(self) -> None:
        probs = np.array([0.9, 0.02, 0.02, 0.02, 0.02, 0.02], dtype=np.float32)
        pipeline = EmotionBridgePipeline(
            classifier=cast("Any", _FakeClassifier(probs)),
            generator=_FakeGenerator(),
            generator_device=torch.device("cpu"),
            style_selector=cast("Any", _FakeStyleSelector()),
            voicevox_client=cast("Any", _FakeVoicevoxClient()),
            adapter=cast("Any", _FakeAdapter()),
            character="zundamon",
            fallback_threshold=0.3,
            dsp_enabled=True,
            dsp_mapper=cast("Any", _FakeDSPMapper()),
            dsp_processor=cast("Any", _FakeDSPProcessor(should_raise=True)),
        )

        caught: RuntimeError | None = None
        try:
            await pipeline.synthesize("怒っています")
        except RuntimeError as exc:
            caught = exc
        assert caught is not None
        assert "DSP processing failed" in str(caught)


if __name__ == "__main__":
    unittest.main()
