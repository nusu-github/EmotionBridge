from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from emotionbridge.constants import CONTROL_PARAM_NAMES, JVNV_EMOTION_LABELS
from emotionbridge.training.generator_trainer import (
    GeneratorConfig,
    GeneratorDataConfig,
    GeneratorEvalConfig,
    GeneratorModelConfig,
    GeneratorTrainConfig,
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
                    strategy="deterministic",
                    samples_per_emotion=0,
                    val_ratio=0.2,
                    random_seed=7,
                ),
                model=GeneratorModelConfig(hidden_dim=16, dropout=0.0),
                train=GeneratorTrainConfig(
                    output_dir=str(root / "output"),
                    batch_size=8,
                    num_epochs=2,
                    lr=1e-3,
                    early_stopping_patience=1,
                    scheduler_patience=1,
                    device="cpu",
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

    def test_train_generator_nn_saves_pretrained_dir(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            teacher_path = root / "teacher" / "recommended_params.json"
            _write_teacher_table(teacher_path)

            config = GeneratorConfig(
                data=GeneratorDataConfig(
                    teacher_table_path=str(teacher_path),
                    strategy="lookup_table_dirichlet",
                    samples_per_emotion=10,
                    dirichlet_alpha_dominant=10.0,
                    dirichlet_alpha_other=1.0,
                    val_ratio=0.2,
                    random_seed=11,
                ),
                model=GeneratorModelConfig(hidden_dim=16, dropout=0.0),
                train=GeneratorTrainConfig(
                    output_dir=str(root / "output"),
                    batch_size=16,
                    num_epochs=3,
                    lr=1e-3,
                    early_stopping_patience=2,
                    scheduler_patience=1,
                    device="cpu",
                ),
                eval=GeneratorEvalConfig(mae_axis_max=0.3),
            )

            result = train_generator(config)
            checkpoint_dir = Path(result["checkpoint"])

            assert checkpoint_dir.is_dir()
            assert (checkpoint_dir / "config.json").exists()
            assert (checkpoint_dir / "model.safetensors").exists()
            assert (checkpoint_dir / "metadata.json").exists()
            assert not (checkpoint_dir.parent / "best_generator.pt").exists()


if __name__ == "__main__":
    unittest.main()
