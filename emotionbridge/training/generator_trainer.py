import json
import shutil
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml

from emotionbridge.constants import CONTROL_PARAM_NAMES, JVNV_EMOTION_LABELS
from emotionbridge.model import DeterministicMixer


@dataclass(slots=True)
class GeneratorDataConfig:
    teacher_table_path: str = "artifacts/generator/teacher_table/recommended_params.json"


@dataclass(slots=True)
class GeneratorTrainConfig:
    output_dir: str = "artifacts/generator"


@dataclass(slots=True)
class GeneratorEvalConfig:
    mae_axis_max: float = 0.2


@dataclass(slots=True)
class GeneratorConfig:
    data: GeneratorDataConfig = field(default_factory=GeneratorDataConfig)
    train: GeneratorTrainConfig = field(default_factory=GeneratorTrainConfig)
    eval: GeneratorEvalConfig = field(default_factory=GeneratorEvalConfig)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _build_section(section_type: type, payload: dict[str, Any] | None):
    payload = payload or {}
    return section_type(**payload)


def _validate_section_keys(
    section_name: str,
    payload: dict[str, Any],
    allowed_keys: set[str],
) -> None:
    unknown_keys = sorted(set(payload) - allowed_keys)
    if unknown_keys:
        msg = f"Unknown keys in '{section_name}' section: {', '.join(unknown_keys)}"
        raise ValueError(msg)


def load_generator_config(config_path: str | Path) -> GeneratorConfig:
    path = Path(config_path)
    if not path.exists():
        msg = f"Config file not found: {path}"
        raise FileNotFoundError(msg)

    with path.open("r", encoding="utf-8") as file:
        raw = yaml.safe_load(file) or {}

    if not isinstance(raw, dict):
        msg = "Generator config must be a YAML mapping"
        raise ValueError(msg)

    _validate_section_keys("root", raw, {"data", "train", "eval"})

    data_payload = raw.get("data") or {}
    train_payload = raw.get("train") or {}
    eval_payload = raw.get("eval") or {}

    if not isinstance(data_payload, dict):
        msg = "'data' section must be a mapping"
        raise ValueError(msg)

    if not isinstance(train_payload, dict):
        msg = "'train' section must be a mapping"
        raise ValueError(msg)

    if not isinstance(eval_payload, dict):
        msg = "'eval' section must be a mapping"
        raise ValueError(msg)

    _validate_section_keys("data", data_payload, {"teacher_table_path"})
    _validate_section_keys("train", train_payload, {"output_dir"})
    _validate_section_keys("eval", eval_payload, {"mae_axis_max"})

    return GeneratorConfig(
        data=_build_section(GeneratorDataConfig, data_payload),
        train=_build_section(GeneratorTrainConfig, train_payload),
        eval=_build_section(GeneratorEvalConfig, eval_payload),
    )


def save_effective_generator_config(
    config: GeneratorConfig,
    output_path: str | Path,
) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        yaml.safe_dump(config.to_dict(), file, sort_keys=False, allow_unicode=True)


def _load_teacher_table(path: Path) -> np.ndarray:
    if not path.exists():
        msg = f"teacher table not found: {path}"
        raise FileNotFoundError(msg)

    with path.open("r", encoding="utf-8") as file:
        payload = json.load(file)

    rows = payload.get("table")
    if not isinstance(rows, list):
        msg = "Invalid teacher table: 'table' must be list"
        raise ValueError(msg)

    row_by_emotion = {row["emotion"]: row for row in rows}
    matrix: list[list[float]] = []
    for emotion in JVNV_EMOTION_LABELS:
        row = row_by_emotion.get(emotion)
        if row is None:
            msg = f"Missing emotion row in teacher table: {emotion}"
            raise ValueError(msg)

        matrix.append([float(row[f"ctrl_{name}"]) for name in CONTROL_PARAM_NAMES])

    return np.asarray(matrix, dtype=np.float32)


def _reset_directory(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def _write_generator_checkpoint_metadata(
    checkpoint_dir: Path,
    *,
    config: GeneratorConfig,
) -> None:
    training_strategy = "deterministic"
    metadata = {
        "training_strategy": training_strategy,
        "emotion_labels": list(JVNV_EMOTION_LABELS),
        "control_param_names": list(CONTROL_PARAM_NAMES),
        "generator_config": config.to_dict(),
    }
    with (checkpoint_dir / "metadata.json").open("w", encoding="utf-8") as file:
        json.dump(metadata, file, ensure_ascii=False, indent=2)

    model_card = "\n".join(
        [
            "---",
            "library_name: emotionbridge",
            "tags:",
            "  - emotion-tts",
            f"  - {training_strategy}",
            "---",
            "",
            "# EmotionBridge Generator",
            "",
            "## Training",
            f"- strategy: {training_strategy}",
            f"- emotions: {', '.join(JVNV_EMOTION_LABELS)}",
            f"- control_params: {', '.join(CONTROL_PARAM_NAMES)}",
            "",
            "## Config",
            "```json",
            json.dumps(config.to_dict(), ensure_ascii=False, indent=2),
            "```",
            "",
        ],
    )
    (checkpoint_dir / "README.md").write_text(model_card, encoding="utf-8")


def _save_generator_model(
    model: DeterministicMixer,
    checkpoint_dir: Path,
    *,
    config: GeneratorConfig,
) -> None:
    _reset_directory(checkpoint_dir)
    model.save_pretrained(str(checkpoint_dir), safe_serialization=True)
    _write_generator_checkpoint_metadata(
        checkpoint_dir,
        config=config,
    )


def _train_deterministic(
    config: GeneratorConfig,
    recommended_matrix: np.ndarray,
    output_dir: Path,
) -> dict[str, Any]:
    """DeterministicMixer を教師表から直接構築して保存する。

    NNの学習ループは不要。教師表の線形混合が最適解であるため、
    学習をスキップして決定論的にチェックポイントを生成する。
    """
    checkpoints_dir = output_dir / "checkpoints"
    reports_dir = output_dir / "reports"
    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    model = DeterministicMixer.from_numpy(recommended_matrix)

    best_dir = checkpoints_dir / "best_generator"
    _save_generator_model(
        model,
        best_dir,
        config=config,
    )

    save_effective_generator_config(config, output_dir / "effective_config.yaml")

    # pure emotion入力でMAEを計算（teacher tableとの一致確認）
    num_emotions = len(JVNV_EMOTION_LABELS)
    identity = np.eye(num_emotions, dtype=np.float32)
    with torch.no_grad():
        preds = model(torch.tensor(identity)).numpy()
    expected = np.tanh(recommended_matrix)
    mae_per_axis = np.mean(np.abs(preds - expected), axis=0).tolist()

    mae_payload = {
        "axis_mae": {
            name: float(value)
            for name, value in zip(CONTROL_PARAM_NAMES, mae_per_axis, strict=True)
        },
        "macro_mae": float(np.mean(mae_per_axis)),
        "threshold": float(config.eval.mae_axis_max),
        "axis_pass": {
            name: bool(value <= config.eval.mae_axis_max)
            for name, value in zip(CONTROL_PARAM_NAMES, mae_per_axis, strict=True)
        },
    }

    go_no_go = {
        "all_axis_mae": bool(all(mae_payload["axis_pass"].values())),
        "go": bool(all(mae_payload["axis_pass"].values())),
    }

    with (reports_dir / "parameter_mae.json").open("w", encoding="utf-8") as file:
        json.dump(mae_payload, file, ensure_ascii=False, indent=2)

    with (reports_dir / "evaluation.json").open("w", encoding="utf-8") as file:
        json.dump(
            {
                "best_epoch": 0,
                "best_val_loss": 0.0,
                "final_val_loss": 0.0,
                "strategy": "deterministic",
                "mae": mae_payload,
                "go_no_go": go_no_go,
            },
            file,
            ensure_ascii=False,
            indent=2,
        )

    return {
        "output_dir": str(output_dir),
        "checkpoint": str(best_dir),
        "best_epoch": 0,
        "best_val_loss": 0.0,
        "final_val_loss": 0.0,
        "mae": mae_payload,
        "go_no_go": go_no_go,
        "num_train_samples": 0,
        "num_val_samples": 0,
        "strategy": "deterministic",
    }


def train_generator(config: GeneratorConfig) -> dict[str, Any]:
    """教師表から DeterministicMixer チェックポイントを生成する。"""
    output_dir = Path(config.train.output_dir)
    recommended_matrix = _load_teacher_table(Path(config.data.teacher_table_path))
    return _train_deterministic(config, recommended_matrix, output_dir)
