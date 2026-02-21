from __future__ import annotations

import io
from typing import Any, cast
import unittest

import numpy as np
import soundfile as sf

from emotionbridge.dsp.config import DSPProcessorConfig
from emotionbridge.dsp.processor import EmotionDSPProcessor
from emotionbridge.dsp.types import DSPControlVector


def _make_wave_bytes(*, seconds: float = 1.0, sr: int = 24000, stereo: bool = False) -> bytes:
    t = np.linspace(0.0, seconds, int(sr * seconds), endpoint=False, dtype=np.float64)
    mono = (0.2 * np.sin(2.0 * np.pi * 220.0 * t)).astype(np.float64)
    if stereo:
        wav = np.stack([mono, mono * 0.9], axis=1)
    else:
        wav = mono

    buf = io.BytesIO()
    sf.write(buf, wav, sr, format="WAV", subtype="PCM_16")
    return buf.getvalue()


class TestEmotionDSPProcessor(unittest.TestCase):
    def test_zero_params_returns_original_bytes(self) -> None:
        processor = EmotionDSPProcessor()
        source = _make_wave_bytes(stereo=True)
        out = processor.process_bytes(source, DSPControlVector(), seed=123)
        assert out == source

    def test_same_seed_produces_identical_output(self) -> None:
        processor = EmotionDSPProcessor()
        source = _make_wave_bytes()
        params = DSPControlVector(
            jitter_amount=0.2,
            shimmer_amount=0.2,
            aperiodicity_shift=0.1,
            spectral_tilt_shift=-0.1,
        )
        out1 = processor.process_bytes(source, params, seed=42)
        out2 = processor.process_bytes(source, params, seed=42)
        assert out1 == out2

    def test_nonzero_params_change_waveform_and_preserve_format(self) -> None:
        processor = EmotionDSPProcessor()
        source = _make_wave_bytes(seconds=0.8, sr=24000, stereo=False)
        params = DSPControlVector(
            jitter_amount=0.2,
            shimmer_amount=0.2,
            aperiodicity_shift=0.15,
            spectral_tilt_shift=0.2,
        )
        out = processor.process_bytes(source, params, seed=7)

        assert out != source

        in_wav, in_sr = sf.read(io.BytesIO(source), dtype="float64")
        out_wav, out_sr = sf.read(io.BytesIO(out), dtype="float64")
        assert in_sr == out_sr
        assert len(in_wav) == len(out_wav)

    def test_harvest_extractor_processes_audio(self) -> None:
        processor = EmotionDSPProcessor(
            DSPProcessorConfig(f0_extractor="harvest"),
        )
        source = _make_wave_bytes(seconds=0.8, sr=24000, stereo=False)
        params = DSPControlVector(
            jitter_amount=0.2,
            shimmer_amount=0.2,
            aperiodicity_shift=0.15,
            spectral_tilt_shift=0.2,
        )
        out1 = processor.process_bytes(source, params, seed=77)
        out2 = processor.process_bytes(source, params, seed=77)
        assert out1 == out2
        assert out1 != source

    def test_invalid_f0_extractor_raises(self) -> None:
        processor = EmotionDSPProcessor(
            DSPProcessorConfig(f0_extractor=cast("Any", "invalid")),
        )
        source = _make_wave_bytes()
        params = DSPControlVector(jitter_amount=0.2)
        caught: ValueError | None = None
        try:
            processor.process_bytes(source, params, seed=1)
        except ValueError as exc:
            caught = exc
        else:
            msg = "Expected ValueError for unsupported f0_extractor"
            raise AssertionError(msg)
        assert caught is not None
        assert "Unsupported f0_extractor" in str(caught)

    def test_invalid_wav_bytes_raise(self) -> None:
        processor = EmotionDSPProcessor()
        try:
            processor.process_bytes(b"not-a-wav", DSPControlVector(jitter_amount=0.2), seed=1)
        except (RuntimeError, ValueError):
            pass
        else:
            msg = "Expected failure for invalid WAV bytes"
            raise AssertionError(msg)


if __name__ == "__main__":
    unittest.main()
