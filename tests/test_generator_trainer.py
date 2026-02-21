from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest
from textwrap import dedent

import pytest

from emotionbridge.constants import CONTROL_PARAM_NAMES, JVNV_EMOTION_LABELS
from emotionbridge.training.generator_trainer import (
    GeneratorConfig,
    GeneratorDataConfig,
    GeneratorEvalConfig,
    GeneratorTrainConfig,
    load_generator_config,
    train_generator,
)


def _write_teacher_table(path: Path) -> None:
    table = []
    for emotion_index, emotion in enumerate(JVNV_EMOTION_LABELS):
        row = {"emotion": emotion}
        for param_index, param_name in enumerate(CONTROL_PARAM_NAMES):
            row[f"ctrl_{param_name}"] = float((emotion_index + param_index) / 10.0)
        table.append(row)
    payload = {"table": table}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


class TestGeneratorTrainer(unittest.TestCase):
    def test_train_generator_deterministic_saves_pretrained_dir(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            teacher_path = root / "teacher" / "recommended_params.json"
            _write_teacher_table(teacher_path)

            config = GeneratorConfig(
                data=GeneratorDataConfig(
                    teacher_table_path=str(teacher_path),
                ),
                train=GeneratorTrainConfig(
                    output_dir=str(root / "output"),
                ),
                eval=GeneratorEvalConfig(mae_axis_max=0.2),
            )

            result = train_generator(config)
            checkpoint_dir = Path(result["checkpoint"])

            assert checkpoint_dir.is_dir()
            assert (checkpoint_dir / "config.json").exists()
            assert (checkpoint_dir / "model.safetensors").exists()
            assert (checkpoint_dir / "metadata.json").exists()
            assert not (checkpoint_dir.parent / "best_generator.pt").exists()

    def test_load_generator_config_rejects_legacy_keys(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            config_path = root / "generator_legacy.yaml"
            config_path.write_text(
                dedent(
                    """\
                    data:
                      teacher_table_path: artifacts/generator/teacher_table/recommended_params.json
                      strategy: lookup_table_dirichlet
                    train:
                      output_dir: artifacts/generator
                    eval:
                      mae_axis_max: 0.2
                    """
                ),
                encoding="utf-8",
            )

            with pytest.raises(ValueError, match="Unknown keys in 'data' section: strategy"):
                load_generator_config(config_path)


if __name__ == "__main__":
    unittest.main()
