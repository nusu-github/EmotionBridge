import io
from typing import Any, cast

import numpy as np
import pyworld as pw
import soundfile as sf

from emotionbridge.dsp.config import DSPProcessorConfig
from emotionbridge.dsp.manipulator import (
    add_jitter,
    add_shimmer,
    modify_aperiodicity,
    modify_spectral_tilt,
)
from emotionbridge.dsp.types import DSPControlVector


def _is_effectively_zero(params: DSPControlVector) -> bool:
    arr = params.to_numpy()
    return bool(np.all(np.abs(arr) <= 1e-8))


class EmotionDSPProcessor:
    """WORLD解析/再合成を用いたDSP後処理。"""

    def __init__(self, config: DSPProcessorConfig | None = None) -> None:
        self._config = config or DSPProcessorConfig()

    def process_bytes(
        self,
        audio_bytes: bytes,
        dsp_params: DSPControlVector | dict[str, Any],
        seed: int,
    ) -> bytes:
        params = (
            dsp_params if isinstance(dsp_params, DSPControlVector) else DSPControlVector.from_dict(dsp_params)
        )
        if _is_effectively_zero(params):
            return audio_bytes

        wav, sr, channels, subtype = self._read_wav_bytes(audio_bytes)
        mono = wav.mean(axis=1) if channels > 1 else wav

        f0, sp, ap = self._analyze(mono, sr)
        rng = np.random.default_rng(seed)

        f0_mod = add_jitter(f0, params.jitter_amount, rng, self._config)
        ap_mod = modify_aperiodicity(ap, params.aperiodicity_shift, f0_mod, sr, self._config)
        sp_mod = modify_spectral_tilt(sp, params.spectral_tilt_shift, sr, self._config)

        world = cast("Any", pw)
        wav_out = world.synthesize(
            f0_mod,
            sp_mod,
            ap_mod,
            sr,
            frame_period=self._config.frame_period_ms,
        )
        wav_out = add_shimmer(wav_out, f0_mod, params.shimmer_amount, sr, rng, self._config)
        wav_out = self._match_length(wav_out, target_len=mono.size)

        if channels > 1:
            wav_out = np.repeat(wav_out[:, None], channels, axis=1)

        return self._write_wav_bytes(wav_out, sr, subtype)

    def _analyze(self, wav: np.ndarray, sr: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        world = cast("Any", pw)
        wav64 = np.asarray(wav, dtype=np.float64)
        f0, time_axis = world.dio(
            wav64,
            sr,
            frame_period=self._config.frame_period_ms,
        )
        f0 = world.stonemask(wav64, f0, time_axis, sr)
        sp = world.cheaptrick(wav64, f0, time_axis, sr)
        ap = world.d4c(wav64, f0, time_axis, sr)
        return f0, sp, ap

    @staticmethod
    def _match_length(wav: np.ndarray, target_len: int) -> np.ndarray:
        if wav.size == target_len:
            return wav
        if wav.size > target_len:
            return wav[:target_len]
        pad_width = target_len - wav.size
        return np.pad(wav, (0, pad_width), mode="constant")

    @staticmethod
    def _read_wav_bytes(audio_bytes: bytes) -> tuple[np.ndarray, int, int, str | None]:
        buffer = io.BytesIO(audio_bytes)
        with sf.SoundFile(buffer) as wav_file:
            sr = int(wav_file.samplerate)
            channels = int(wav_file.channels)
            subtype = wav_file.subtype
            data = wav_file.read(dtype="float64", always_2d=True)

        if channels == 1:
            return data[:, 0], sr, channels, subtype
        return data, sr, channels, subtype

    @staticmethod
    def _write_wav_bytes(
        wav: np.ndarray,
        sr: int,
        subtype: str | None,
    ) -> bytes:
        buffer = io.BytesIO()
        try:
            sf.write(
                file=buffer,
                data=wav,
                samplerate=sr,
                format="WAV",
                subtype=subtype,
            )
        except RuntimeError:
            sf.write(
                file=buffer,
                data=wav,
                samplerate=sr,
                format="WAV",
                subtype="PCM_16",
            )
        return buffer.getvalue()
