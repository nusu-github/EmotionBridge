from dataclasses import dataclass
from typing import Any

import numpy as np

from emotionbridge.constants import DSP_PARAM_NAMES


@dataclass(frozen=True, slots=True)
class DSPControlVector:
    """DSP後処理制御の4次元ベクトル。"""

    jitter_amount: float = 0.0
    shimmer_amount: float = 0.0
    aperiodicity_shift: float = 0.0
    spectral_tilt_shift: float = 0.0

    def __post_init__(self) -> None:
        ranges = {
            "jitter_amount": (0.0, 1.0),
            "shimmer_amount": (0.0, 1.0),
            "aperiodicity_shift": (-1.0, 1.0),
            "spectral_tilt_shift": (-1.0, 1.0),
        }
        for name in DSP_PARAM_NAMES:
            value = getattr(self, name)
            if not isinstance(value, (int, float)):
                msg = f"{name} must be a number, got {type(value).__name__}"
                raise TypeError(msg)
            lo, hi = ranges[name]
            if not (lo <= value <= hi):
                msg = f"{name} must be in [{lo}, {hi}], got {value}"
                raise ValueError(msg)

    def to_numpy(self) -> np.ndarray:
        return np.array(
            [
                self.jitter_amount,
                self.shimmer_amount,
                self.aperiodicity_shift,
                self.spectral_tilt_shift,
            ],
            dtype=np.float32,
        )

    @classmethod
    def from_numpy(cls, arr: np.ndarray) -> "DSPControlVector":
        if arr.shape != (4,):
            msg = f"Expected shape (4,), got {arr.shape}"
            raise ValueError(msg)
        return cls(
            jitter_amount=float(arr[0]),
            shimmer_amount=float(arr[1]),
            aperiodicity_shift=float(arr[2]),
            spectral_tilt_shift=float(arr[3]),
        )

    def to_dict(self) -> dict[str, float]:
        return {
            "jitter_amount": float(self.jitter_amount),
            "shimmer_amount": float(self.shimmer_amount),
            "aperiodicity_shift": float(self.aperiodicity_shift),
            "spectral_tilt_shift": float(self.spectral_tilt_shift),
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "DSPControlVector":
        return cls(
            jitter_amount=float(payload.get("jitter_amount", 0.0)),
            shimmer_amount=float(payload.get("shimmer_amount", 0.0)),
            aperiodicity_shift=float(payload.get("aperiodicity_shift", 0.0)),
            spectral_tilt_shift=float(payload.get("spectral_tilt_shift", 0.0)),
        )
