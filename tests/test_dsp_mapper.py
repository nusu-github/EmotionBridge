from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from emotionbridge.dsp.mapper import EmotionDSPMapper


class TestEmotionDSPMapper(unittest.TestCase):
    def _build_valid_df(self) -> pd.DataFrame:
        rows = [
            {
                "emotion": "anger",
                "egemaps__jitterLocal_sma3nz_amean": 2.0,
                "egemaps__shimmerLocaldB_sma3nz_amean": 1.8,
                "egemaps__HNRdBACF_sma3nz_amean": -1.5,
                "egemaps__spectralFluxV_sma3nz_amean": 1.2,
                "egemaps__slopeV0-500_sma3nz_amean": -0.6,
                "egemaps__slopeV500-1500_sma3nz_amean": -0.4,
            },
            {
                "emotion": "disgust",
                "egemaps__jitterLocal_sma3nz_amean": 0.3,
                "egemaps__shimmerLocaldB_sma3nz_amean": 0.1,
                "egemaps__HNRdBACF_sma3nz_amean": -0.2,
                "egemaps__spectralFluxV_sma3nz_amean": 0.2,
                "egemaps__slopeV0-500_sma3nz_amean": -0.2,
                "egemaps__slopeV500-1500_sma3nz_amean": -0.1,
            },
            {
                "emotion": "fear",
                "egemaps__jitterLocal_sma3nz_amean": -0.4,
                "egemaps__shimmerLocaldB_sma3nz_amean": -0.2,
                "egemaps__HNRdBACF_sma3nz_amean": 0.7,
                "egemaps__spectralFluxV_sma3nz_amean": -0.2,
                "egemaps__slopeV0-500_sma3nz_amean": 0.3,
                "egemaps__slopeV500-1500_sma3nz_amean": 0.3,
            },
            {
                "emotion": "happy",
                "egemaps__jitterLocal_sma3nz_amean": -0.2,
                "egemaps__shimmerLocaldB_sma3nz_amean": 0.0,
                "egemaps__HNRdBACF_sma3nz_amean": 0.2,
                "egemaps__spectralFluxV_sma3nz_amean": -0.1,
                "egemaps__slopeV0-500_sma3nz_amean": 0.2,
                "egemaps__slopeV500-1500_sma3nz_amean": 0.2,
            },
            {
                "emotion": "sad",
                "egemaps__jitterLocal_sma3nz_amean": -0.3,
                "egemaps__shimmerLocaldB_sma3nz_amean": -0.1,
                "egemaps__HNRdBACF_sma3nz_amean": 0.3,
                "egemaps__spectralFluxV_sma3nz_amean": -0.6,
                "egemaps__slopeV0-500_sma3nz_amean": -0.2,
                "egemaps__slopeV500-1500_sma3nz_amean": -0.1,
            },
            {
                "emotion": "surprise",
                "egemaps__jitterLocal_sma3nz_amean": -0.1,
                "egemaps__shimmerLocaldB_sma3nz_amean": -0.4,
                "egemaps__HNRdBACF_sma3nz_amean": 0.4,
                "egemaps__spectralFluxV_sma3nz_amean": 0.0,
                "egemaps__slopeV0-500_sma3nz_amean": 0.1,
                "egemaps__slopeV500-1500_sma3nz_amean": -0.2,
            },
        ]
        return pd.DataFrame(rows)

    def test_missing_required_feature_raises(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "features.parquet"
            pd.DataFrame(
                [
                    {"emotion": "anger", "egemaps__jitterLocal_sma3nz_amean": 1.0},
                ],
            ).to_parquet(path, index=False)

            try:
                EmotionDSPMapper(features_path=path)
            except ValueError:
                pass
            else:
                msg = "Expected ValueError for missing required features"
                raise AssertionError(msg)

    def test_anger_has_positive_jitter_shimmer_and_aperiodicity(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "features.parquet"
            self._build_valid_df().to_parquet(path, index=False)
            mapper = EmotionDSPMapper(features_path=path)

            probs = np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
            params = mapper.generate(probs)

            assert params.jitter_amount > 0.0
            assert params.shimmer_amount > 0.0
            assert params.aperiodicity_shift > 0.0

    def test_linear_mixing_for_probability_blend(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "features.parquet"
            self._build_valid_df().to_parquet(path, index=False)
            mapper = EmotionDSPMapper(features_path=path)

            anger = mapper.generate(np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32))
            happy = mapper.generate(np.array([0.0, 0.0, 0.0, 1.0, 0.0, 0.0], dtype=np.float32))
            mixed = mapper.generate(np.array([0.5, 0.0, 0.0, 0.5, 0.0, 0.0], dtype=np.float32))

            expected = 0.5 * anger.to_numpy() + 0.5 * happy.to_numpy()
            np.testing.assert_allclose(mixed.to_numpy(), expected, rtol=1e-6, atol=1e-6)


if __name__ == "__main__":
    unittest.main()
