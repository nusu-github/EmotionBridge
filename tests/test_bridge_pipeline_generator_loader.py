from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

import numpy as np
import pytest

from emotionbridge.inference.bridge_pipeline import _load_generator_model_dir
from emotionbridge.model import DeterministicMixer


class TestBridgePipelineGeneratorLoader(unittest.TestCase):
    def test_loads_deterministic_mixer_from_model_dir(self) -> None:
        matrix = np.arange(30, dtype=np.float32).reshape(6, 5) / 10.0
        model = DeterministicMixer(matrix.tolist())

        with tempfile.TemporaryDirectory() as tmp_dir:
            model_dir = Path(tmp_dir) / "deterministic"
            model.save_pretrained(str(model_dir), safe_serialization=True)

            loaded, device = _load_generator_model_dir(model_dir, device="cpu")
            assert isinstance(loaded, DeterministicMixer)
            assert str(device) == "cpu"

    def test_rejects_legacy_pt_checkpoint(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            pt_path = Path(tmp_dir) / "best_generator.pt"
            pt_path.write_bytes(b"legacy")

            with pytest.raises(
                ValueError, match="Legacy \\.pt generator checkpoints are not supported"
            ):
                _load_generator_model_dir(pt_path, device="cpu")

    def test_raises_for_undetectable_config(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            model_dir = Path(tmp_dir) / "unknown"
            model_dir.mkdir(parents=True, exist_ok=True)
            (model_dir / "config.json").write_text("{}", encoding="utf-8")

            with pytest.raises(
                ValueError,
                match="Invalid generator config: expected DeterministicMixer checkpoint",
            ):
                _load_generator_model_dir(model_dir, device="cpu")


if __name__ == "__main__":
    unittest.main()
