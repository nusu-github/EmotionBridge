import numpy as np

from emotionbridge.dsp.config import DSPProcessorConfig


def _voiced_segments(mask: np.ndarray) -> list[tuple[int, int]]:
    segments: list[tuple[int, int]] = []
    start: int | None = None
    for idx, value in enumerate(mask.tolist()):
        if value and start is None:
            start = idx
        elif not value and start is not None:
            segments.append((start, idx))
            start = None
    if start is not None:
        segments.append((start, len(mask)))
    return segments


def _boundary_envelope(length: int, fade_frames: int) -> np.ndarray:
    if length <= 0:
        return np.zeros(0, dtype=np.float64)
    if fade_frames <= 0 or length <= 2:
        return np.ones(length, dtype=np.float64)

    fade = min(fade_frames, length // 2)
    if fade == 0:
        return np.ones(length, dtype=np.float64)

    envelope = np.ones(length, dtype=np.float64)
    ramp = np.linspace(0.0, 1.0, fade, endpoint=True, dtype=np.float64)
    envelope[:fade] = ramp
    envelope[-fade:] = ramp[::-1]
    return envelope


def add_jitter(
    f0: np.ndarray,
    amount: float,
    rng: np.random.Generator,
    cfg: DSPProcessorConfig,
) -> np.ndarray:
    f0_in = np.asarray(f0, dtype=np.float64)
    if amount <= 1e-8:
        return f0_in.copy()

    f0_out = f0_in.copy()
    voiced = f0_out > 0.0
    if not np.any(voiced):
        return f0_out

    for start, end in _voiced_segments(voiced):
        segment = f0_out[start:end]
        if segment.size == 0:
            continue

        local_std = float(np.std(segment, ddof=0))
        if local_std <= 1e-6:
            local_std = max(float(np.mean(segment)) * 0.01, 1.0)

        noise_std = amount * local_std * cfg.jitter_max_ratio
        noise = rng.normal(0.0, noise_std, size=segment.size)
        envelope = _boundary_envelope(segment.size, cfg.boundary_fade_frames)
        f0_out[start:end] = segment + (noise * envelope)

    lo, hi = cfg.f0_clip
    f0_out[voiced] = np.clip(f0_out[voiced], lo, hi)
    f0_out[~voiced] = 0.0
    return f0_out


def modify_aperiodicity(
    ap: np.ndarray,
    shift: float,
    f0: np.ndarray,
    sr: int,
    cfg: DSPProcessorConfig,
) -> np.ndarray:
    ap_in = np.asarray(ap, dtype=np.float64)
    if abs(shift) <= 1e-8:
        return ap_in.copy()

    ap_out = ap_in.copy()
    voiced = np.asarray(f0, dtype=np.float64) > 0.0
    frame_count = min(ap_out.shape[0], voiced.size)
    if frame_count == 0:
        return ap_out

    voiced = voiced[:frame_count]
    freqs = np.linspace(0.0, sr / 2.0, ap_out.shape[1], dtype=np.float64)
    delta = np.full(ap_out.shape[1], shift, dtype=np.float64)
    delta[freqs <= 500.0] *= 0.5

    voiced_indices = np.where(voiced)[0]
    if voiced_indices.size > 0:
        ap_out[voiced_indices, :] += delta[None, :]
    lo, hi = cfg.aperiodicity_clip
    return np.clip(ap_out, lo, hi)


def modify_spectral_tilt(
    sp: np.ndarray,
    tilt_shift: float,
    sr: int,
    cfg: DSPProcessorConfig,
) -> np.ndarray:
    sp_in = np.asarray(sp, dtype=np.float64)
    if abs(tilt_shift) <= 1e-8:
        return sp_in.copy()

    sp_out = sp_in.copy()
    nyquist = sr / 2.0
    freqs = np.linspace(0.0, nyquist, sp_out.shape[1], dtype=np.float64)
    gain = 1.0 + tilt_shift * cfg.tilt_scale * ((freqs / max(nyquist, 1.0)) - 0.5)
    gain = np.clip(gain, cfg.spectral_gain_clip[0], cfg.spectral_gain_clip[1])

    gain[freqs <= 200.0] = 1.0
    high_mask = freqs >= 5000.0
    gain[high_mask] = 1.0 + (gain[high_mask] - 1.0) * 0.5

    sp_out *= gain[None, :]
    return np.clip(sp_out, 1e-12, None)


def add_shimmer(
    wav: np.ndarray,
    f0: np.ndarray,
    amount: float,
    sr: int,
    rng: np.random.Generator,
    cfg: DSPProcessorConfig,
) -> np.ndarray:
    del sr  # frame系列をそのまま補間するため明示的には使わない
    wav_in = np.asarray(wav, dtype=np.float64)
    if amount <= 1e-8:
        return wav_in.copy()
    if wav_in.size == 0:
        return wav_in.copy()

    f0_arr = np.asarray(f0, dtype=np.float64)
    if f0_arr.size == 0:
        return wav_in.copy()

    frame_gain = np.ones(f0_arr.size, dtype=np.float64)
    voiced = f0_arr > 0.0
    if np.any(voiced):
        noise_db = rng.normal(0.0, amount * cfg.shimmer_max_db, size=int(np.sum(voiced)))
        voiced_gain = np.power(10.0, noise_db / 20.0)
        frame_gain[voiced] = np.clip(
            voiced_gain,
            cfg.shimmer_gain_clip[0],
            cfg.shimmer_gain_clip[1],
        )

    frame_positions = np.linspace(0, wav_in.size - 1, num=f0_arr.size, dtype=np.float64)
    sample_positions = np.arange(wav_in.size, dtype=np.float64)
    sample_gain = np.interp(sample_positions, frame_positions, frame_gain)
    return wav_in * sample_gain
