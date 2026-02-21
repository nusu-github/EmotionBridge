from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import pytest
import torch

from emotionbridge.model import DeterministicMixer


class TestGeneratorModelSerialization(unittest.TestCase):
    def test_deterministic_mixer_roundtrip(self) -> None:
        matrix = np.arange(30, dtype=np.float32).reshape(6, 5) / 30.0
        model = DeterministicMixer(matrix.tolist())
        model.eval()

        inputs = torch.tensor(
            [
                [0.70, 0.10, 0.05, 0.05, 0.05, 0.05],
                [0.10, 0.10, 0.10, 0.20, 0.30, 0.20],
            ],
            dtype=torch.float32,
        )
        with torch.no_grad():
            expected = model(inputs)

        with tempfile.TemporaryDirectory() as tmp_dir:
            model_dir = Path(tmp_dir) / "deterministic_mixer"
            model.save_pretrained(str(model_dir), safe_serialization=True)

            loaded = DeterministicMixer.from_pretrained(str(model_dir))
            loaded.eval()
            with torch.no_grad():
                actual = loaded(inputs)

            np.testing.assert_allclose(actual.numpy(), expected.numpy(), rtol=1e-6, atol=1e-6)
            assert (model_dir / "config.json").exists()
            assert (model_dir / "model.safetensors").exists()

    def test_deterministic_mixer_rejects_invalid_shape(self) -> None:
        with pytest.raises(ValueError, match="must have 6 rows"):
            DeterministicMixer([[0.0, 0.0, 0.0, 0.0, 0.0]])


if __name__ == "__main__":
    unittest.main()
