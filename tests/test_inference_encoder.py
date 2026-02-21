from __future__ import annotations

from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import pytest

from emotionbridge.inference.encoder import EmotionEncoder


class _FakePipeline:
    def __init__(self, responses: list[object], id2label: dict[object, str] | None = None) -> None:
        self._responses = responses
        self.model = SimpleNamespace(
            config=SimpleNamespace(
                id2label=id2label or {},
                max_position_embeddings=64,
            ),
        )
        self.tokenizer = object()

    def __call__(self, *args, **kwargs):
        del args, kwargs
        return self._responses


class TestEmotionEncoder(unittest.TestCase):
    def test_non_directory_checkpoint_path_raises(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            checkpoint_file = Path(tmp_dir) / "checkpoint.bin"
            checkpoint_file.write_text("dummy", encoding="utf-8")
            with pytest.raises(ValueError, match="Checkpoint path must be a directory"):
                EmotionEncoder(str(checkpoint_file), device="cpu")

    def test_encode_batch_maps_scores_by_id2label_order(self) -> None:
        responses = [
            [
                {"label": "happy", "score": 0.7},
                {"label": "sad", "score": 0.2},
                {"label": "anger", "score": 0.1},
            ],
            [
                {"label": "happy", "score": 0.1},
                {"label": "sad", "score": 0.3},
                {"label": "anger", "score": 0.6},
            ],
        ]
        fake = _FakePipeline(responses=responses, id2label={"1": "happy", "0": "anger", "2": "sad"})

        with tempfile.TemporaryDirectory() as tmp_dir:
            checkpoint = Path(tmp_dir)
            with patch("emotionbridge.inference.encoder.pipeline", return_value=fake):
                encoder = EmotionEncoder(str(checkpoint), device="cpu")
                result = encoder.encode_batch(["a", "b"], batch_size=2)

        expected = np.array(
            [
                [0.1, 0.7, 0.2],
                [0.6, 0.1, 0.3],
            ],
            dtype=np.float32,
        )
        np.testing.assert_allclose(result, expected)
        assert encoder.label_names == ["anger", "happy", "sad"]

    def test_generic_label_fallback_uses_label_index_order(self) -> None:
        responses = [
            [
                {"label": "LABEL_5", "score": 0.6},
                {"label": "LABEL_0", "score": 0.1},
                {"label": "LABEL_1", "score": 0.2},
                {"label": "LABEL_2", "score": 0.3},
                {"label": "LABEL_3", "score": 0.4},
                {"label": "LABEL_4", "score": 0.5},
            ]
        ]
        fake = _FakePipeline(responses=responses, id2label={})

        with tempfile.TemporaryDirectory() as tmp_dir:
            checkpoint = Path(tmp_dir)
            with patch("emotionbridge.inference.encoder.pipeline", return_value=fake):
                encoder = EmotionEncoder(str(checkpoint), device="cpu")
                result = encoder.encode("テスト")

        np.testing.assert_allclose(
            result,
            np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6], dtype=np.float32),
        )
        assert len(encoder.label_names) == 6

    def test_missing_labels_raises(self) -> None:
        responses = [[{"label": "happy", "score": 1.0}]]
        fake = _FakePipeline(responses=responses, id2label={0: "anger", 1: "happy"})

        with tempfile.TemporaryDirectory() as tmp_dir:
            checkpoint = Path(tmp_dir)
            with patch("emotionbridge.inference.encoder.pipeline", return_value=fake):
                encoder = EmotionEncoder(str(checkpoint), device="cpu")
                try:
                    encoder.encode("テスト")
                except ValueError:
                    pass
                else:
                    msg = "Expected ValueError for missing label score"
                    raise AssertionError(msg)


if __name__ == "__main__":
    unittest.main()
