from pathlib import Path

import numpy as np
import pandas as pd

from emotionbridge.constants import DSP_PARAM_NAMES, JVNV_EMOTION_LABELS
from emotionbridge.dsp.config import DSPMapperConfig
from emotionbridge.dsp.types import DSPControlVector

_TARGET_FEATURES: tuple[str, ...] = (
    "egemaps__jitterLocal_sma3nz_amean",
    "egemaps__shimmerLocaldB_sma3nz_amean",
    "egemaps__HNRdBACF_sma3nz_amean",
    "egemaps__spectralFluxV_sma3nz_amean",
    "egemaps__slopeV0-500_sma3nz_amean",
    "egemaps__slopeV500-1500_sma3nz_amean",
)


def _canon_common6(label: str) -> str | None:
    lowered = str(label).strip().lower()
    mapping = {
        "anger": "anger",
        "angry": "anger",
        "disgust": "disgust",
        "fear": "fear",
        "happy": "happy",
        "joy": "happy",
        "sad": "sad",
        "sadness": "sad",
        "surprise": "surprise",
    }
    return mapping.get(lowered)


def _scale_positive(values: list[float], cap: float) -> float:
    max_value = max(values) if values else 0.0
    if max_value <= 0.0:
        return 0.0
    return cap / max_value


def _scale_signed(values: list[float], cap: float) -> float:
    max_abs = max((abs(value) for value in values), default=0.0)
    if max_abs <= 0.0:
        return 0.0
    return cap / max_abs


class EmotionDSPMapper:
    """感情確率ベクトルをDSP制御量に変換するマッパー。"""

    def __init__(
        self,
        features_path: str | Path = "artifacts/prosody/v01/jvnv_egemaps_normalized.parquet",
        config: DSPMapperConfig | None = None,
    ) -> None:
        self._config = config or DSPMapperConfig()
        self._features_path = Path(features_path)
        if not self._features_path.exists():
            msg = f"DSP features file not found: {self._features_path}"
            raise FileNotFoundError(msg)

        df = pd.read_parquet(self._features_path).copy()
        if "emotion" not in df.columns:
            msg = "DSP features parquet must contain 'emotion' column"
            raise ValueError(msg)

        missing_features = [name for name in _TARGET_FEATURES if name not in df.columns]
        if missing_features:
            msg = f"DSP features missing required columns: {missing_features}"
            raise ValueError(msg)

        df["emotion_common6"] = df["emotion"].map(_canon_common6)
        df = df[df["emotion_common6"].notna()].reset_index(drop=True)
        if df.empty:
            msg = "No common6 emotion rows found in DSP features parquet"
            raise ValueError(msg)

        self._teacher_matrix = self._build_teacher_matrix(df)

    @property
    def teacher_matrix(self) -> np.ndarray:
        return self._teacher_matrix.copy()

    def generate(self, probs_common6: np.ndarray) -> DSPControlVector:
        probs = np.asarray(probs_common6, dtype=np.float32)
        if probs.shape != (6,):
            msg = f"Expected emotion probabilities shape (6,), got {probs.shape}"
            raise ValueError(msg)

        raw = probs @ self._teacher_matrix
        clipped = np.array(
            [
                np.clip(raw[0], 0.0, 1.0),
                np.clip(raw[1], 0.0, 1.0),
                np.clip(raw[2], -1.0, 1.0),
                np.clip(raw[3], -1.0, 1.0),
            ],
            dtype=np.float32,
        )
        return DSPControlVector.from_numpy(clipped)

    def _build_teacher_matrix(self, df: pd.DataFrame) -> np.ndarray:
        grouped = df.groupby("emotion_common6", dropna=False)[list(_TARGET_FEATURES)].mean()

        missing_emotions = [emo for emo in JVNV_EMOTION_LABELS if emo not in grouped.index]
        if missing_emotions:
            msg = f"Missing common6 emotions in DSP features: {missing_emotions}"
            raise ValueError(msg)

        global_mean = grouped.loc[JVNV_EMOTION_LABELS].mean(axis=0)

        raw_by_emotion: dict[str, dict[str, float]] = {}
        for emotion in JVNV_EMOTION_LABELS:
            row = grouped.loc[emotion]
            delta = row - global_mean
            raw_by_emotion[emotion] = {
                "jitter_amount": float(max(delta["egemaps__jitterLocal_sma3nz_amean"], 0.0)),
                "shimmer_amount": float(max(delta["egemaps__shimmerLocaldB_sma3nz_amean"], 0.0)),
                "aperiodicity_shift": float(-delta["egemaps__HNRdBACF_sma3nz_amean"]),
                "spectral_tilt_shift": float(delta["egemaps__slopeV0-500_sma3nz_amean"]),
            }

        jitter_scale = _scale_positive(
            [raw_by_emotion[emo]["jitter_amount"] for emo in JVNV_EMOTION_LABELS],
            self._config.jitter_cap,
        )
        shimmer_scale = _scale_positive(
            [raw_by_emotion[emo]["shimmer_amount"] for emo in JVNV_EMOTION_LABELS],
            self._config.shimmer_cap,
        )
        aperiodicity_scale = _scale_signed(
            [raw_by_emotion[emo]["aperiodicity_shift"] for emo in JVNV_EMOTION_LABELS],
            self._config.aperiodicity_cap,
        )
        tilt_scale = _scale_signed(
            [raw_by_emotion[emo]["spectral_tilt_shift"] for emo in JVNV_EMOTION_LABELS],
            self._config.spectral_tilt_cap,
        )

        matrix_rows: list[list[float]] = []
        for emotion in JVNV_EMOTION_LABELS:
            raw = raw_by_emotion[emotion]
            vector = DSPControlVector(
                jitter_amount=float(np.clip(raw["jitter_amount"] * jitter_scale, 0.0, 1.0)),
                shimmer_amount=float(np.clip(raw["shimmer_amount"] * shimmer_scale, 0.0, 1.0)),
                aperiodicity_shift=float(
                    np.clip(raw["aperiodicity_shift"] * aperiodicity_scale, -1.0, 1.0),
                ),
                spectral_tilt_shift=float(
                    np.clip(raw["spectral_tilt_shift"] * tilt_scale, -1.0, 1.0),
                ),
            )
            row_dict = vector.to_dict()
            matrix_rows.append([row_dict[name] for name in DSP_PARAM_NAMES])

        return np.array(matrix_rows, dtype=np.float32)
