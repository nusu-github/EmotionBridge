from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class DSPMapperConfig:
    """EmotionDSPMapper のスケーリング設定。"""

    jitter_cap: float = 0.30
    shimmer_cap: float = 0.25
    aperiodicity_cap: float = 0.20
    spectral_tilt_cap: float = 0.30


@dataclass(frozen=True, slots=True)
class DSPProcessorConfig:
    """WORLDベースDSP処理の安全制約設定。"""

    f0_clip: tuple[float, float] = (50.0, 800.0)
    aperiodicity_clip: tuple[float, float] = (0.0, 0.95)
    boundary_fade_frames: int = 3
    frame_period_ms: float = 5.0
    tilt_scale: float = 0.3

    # 実装側の内部上限（安全側）
    jitter_max_ratio: float = 0.05
    shimmer_max_db: float = 1.0
    shimmer_gain_clip: tuple[float, float] = (0.5, 1.5)
    spectral_gain_clip: tuple[float, float] = (0.5, 2.0)
