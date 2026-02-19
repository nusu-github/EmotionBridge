import json
import random
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from sklearn.model_selection import train_test_split
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, TensorDataset

from emotionbridge.constants import CONTROL_PARAM_NAMES, JVNV_EMOTION_LABELS
from emotionbridge.model import DeterministicMixer, ParameterGenerator


@dataclass(slots=True)
class GeneratorDataConfig:
    teacher_table_path: str = "artifacts/generator/teacher_table/recommended_params.json"
    strategy: str = "lookup_table_dirichlet"
    samples_per_emotion: int = 300
    dirichlet_alpha_dominant: float = 10.0
    dirichlet_alpha_other: float = 1.0
    val_ratio: float = 0.2
    random_seed: int = 42


@dataclass(slots=True)
class GeneratorModelConfig:
    hidden_dim: int = 64
    dropout: float = 0.3


@dataclass(slots=True)
class GeneratorTrainConfig:
    output_dir: str = "artifacts/generator"
    batch_size: int = 32
    num_epochs: int = 500
    lr: float = 1e-3
    weight_decay: float = 0.0
    early_stopping_patience: int = 30
    scheduler_patience: int = 10
    scheduler_factor: float = 0.5
    device: str = "cuda"


@dataclass(slots=True)
class GeneratorEvalConfig:
    mae_axis_max: float = 0.2


@dataclass(slots=True)
class GeneratorConfig:
    data: GeneratorDataConfig = field(default_factory=GeneratorDataConfig)
    model: GeneratorModelConfig = field(default_factory=GeneratorModelConfig)
    train: GeneratorTrainConfig = field(default_factory=GeneratorTrainConfig)
    eval: GeneratorEvalConfig = field(default_factory=GeneratorEvalConfig)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _build_section(section_type: type, payload: dict[str, Any] | None):
    payload = payload or {}
    return section_type(**payload)


def load_generator_config(config_path: str | Path) -> GeneratorConfig:
    path = Path(config_path)
    if not path.exists():
        msg = f"Config file not found: {path}"
        raise FileNotFoundError(msg)

    with path.open("r", encoding="utf-8") as file:
        raw = yaml.safe_load(file) or {}

    config = GeneratorConfig(
        data=_build_section(GeneratorDataConfig, raw.get("data")),
        model=_build_section(GeneratorModelConfig, raw.get("model")),
        train=_build_section(GeneratorTrainConfig, raw.get("train")),
        eval=_build_section(GeneratorEvalConfig, raw.get("eval")),
    )

    if not 0.0 < config.data.val_ratio < 1.0:
        msg = "data.val_ratio must be in (0, 1)"
        raise ValueError(msg)

    if config.data.samples_per_emotion < 0:
        msg = "data.samples_per_emotion must be >= 0"
        raise ValueError(msg)

    return config


def save_effective_generator_config(
    config: GeneratorConfig,
    output_path: str | Path,
) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        yaml.safe_dump(config.to_dict(), file, sort_keys=False, allow_unicode=True)


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _resolve_device(device_name: str) -> torch.device:
    if device_name == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


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


def _build_augmented_dataset(
    recommended_matrix: np.ndarray,
    *,
    samples_per_emotion: int,
    alpha_dominant: float,
    alpha_other: float,
    random_seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(random_seed)

    emotion_probs: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    dominant_labels: list[int] = []

    num_emotions = len(JVNV_EMOTION_LABELS)
    for index in range(num_emotions):
        one_hot = np.zeros((num_emotions,), dtype=np.float32)
        one_hot[index] = 1.0
        emotion_probs.append(one_hot)
        targets.append(recommended_matrix[index])
        dominant_labels.append(index)

        if samples_per_emotion > 0:
            alpha = np.full((num_emotions,), max(alpha_other, 1e-6), dtype=np.float64)
            alpha[index] = max(alpha_dominant, 1e-6)
            sampled = rng.dirichlet(alpha, size=samples_per_emotion).astype(np.float32)
            sampled_targets = sampled @ recommended_matrix

            emotion_probs.extend(sampled)
            targets.extend(sampled_targets)
            dominant_labels.extend(np.argmax(sampled, axis=1).astype(np.int64).tolist())

    return (
        np.asarray(emotion_probs, dtype=np.float32),
        np.asarray(targets, dtype=np.float32),
        np.asarray(dominant_labels, dtype=np.int64),
    )


def weighted_mse_loss(
    predicted: torch.Tensor,
    emotion_probs: torch.Tensor,
    recommended: torch.Tensor,
) -> torch.Tensor:
    diff = predicted.unsqueeze(1) - recommended.unsqueeze(0)
    mse_per_emotion = (diff**2).mean(dim=2)
    weighted = (emotion_probs.detach() * mse_per_emotion).sum(dim=1)
    return weighted.mean()


def _mae_by_axis(predictions: np.ndarray, targets: np.ndarray) -> list[float]:
    return np.mean(np.abs(predictions - targets), axis=0).astype(np.float64).tolist()


def _run_eval(
    model: ParameterGenerator,
    loader: DataLoader,
    device: torch.device,
    recommended: torch.Tensor,
) -> tuple[float, list[float]]:
    model.eval()
    losses: list[float] = []
    preds_all: list[np.ndarray] = []
    targets_all: list[np.ndarray] = []

    with torch.no_grad():
        for probs, targets in loader:
            probs = probs.to(device)
            targets = targets.to(device)

            preds = model(probs)
            loss = weighted_mse_loss(preds, probs, recommended)
            losses.append(float(loss.detach().cpu().item()))

            preds_all.append(preds.detach().cpu().numpy())
            targets_all.append(targets.detach().cpu().numpy())

    val_loss = float(np.mean(losses)) if losses else 0.0
    mae_per_axis = (
        [0.0 for _ in CONTROL_PARAM_NAMES]
        if not preds_all
        else _mae_by_axis(np.vstack(preds_all), np.vstack(targets_all))
    )

    return val_loss, mae_per_axis


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

    best_path = checkpoints_dir / "best_generator.pt"
    torch.save(
        {
            "model_type": "deterministic_mixer",
            "model_state_dict": model.state_dict(),
            "recommended_params": recommended_matrix.tolist(),
            "emotion_labels": JVNV_EMOTION_LABELS,
            "control_param_names": CONTROL_PARAM_NAMES,
            "training_strategy": "deterministic",
            "config": config.to_dict(),
        },
        best_path,
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
        "checkpoint": str(best_path),
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
    """パラメータ生成器の学習エントリポイント。

    config.data.strategy に応じて処理を分岐する:
    - "deterministic": 教師表から DeterministicMixer を直接構築（学習ループなし）
    - "lookup_table_dirichlet": Dirichlet 拡張データで ParameterGenerator を学習
    """
    _set_seed(config.data.random_seed)
    device = _resolve_device(config.train.device)

    output_dir = Path(config.train.output_dir)
    checkpoints_dir = output_dir / "checkpoints"
    reports_dir = output_dir / "reports"
    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    teacher_path = Path(config.data.teacher_table_path)
    recommended_matrix = _load_teacher_table(teacher_path)

    if config.data.strategy == "deterministic":
        return _train_deterministic(config, recommended_matrix, output_dir)

    probs_np, targets_np, dominant = _build_augmented_dataset(
        recommended_matrix,
        samples_per_emotion=config.data.samples_per_emotion,
        alpha_dominant=config.data.dirichlet_alpha_dominant,
        alpha_other=config.data.dirichlet_alpha_other,
        random_seed=config.data.random_seed,
    )

    train_probs, val_probs, train_targets, val_targets, _train_dom, _val_dom = train_test_split(
        probs_np,
        targets_np,
        dominant,
        test_size=config.data.val_ratio,
        random_state=config.data.random_seed,
        stratify=dominant,
    )

    train_ds = TensorDataset(
        torch.tensor(train_probs, dtype=torch.float32),
        torch.tensor(train_targets, dtype=torch.float32),
    )
    val_ds = TensorDataset(
        torch.tensor(val_probs, dtype=torch.float32),
        torch.tensor(val_targets, dtype=torch.float32),
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=config.train.batch_size,
        shuffle=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=config.train.batch_size,
        shuffle=False,
    )

    model = ParameterGenerator(
        hidden_dim=config.model.hidden_dim,
        dropout=config.model.dropout,
    ).to(device)

    recommended_tensor = torch.tensor(
        recommended_matrix,
        dtype=torch.float32,
        device=device,
    )
    model.register_buffer("recommended_params", recommended_tensor)

    optimizer = Adam(
        model.parameters(),
        lr=config.train.lr,
        weight_decay=config.train.weight_decay,
    )
    scheduler = ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=config.train.scheduler_factor,
        patience=config.train.scheduler_patience,
    )

    best_val_loss = float("inf")
    best_epoch = 0
    early_stop = 0
    best_path = checkpoints_dir / "best_generator.pt"
    history: list[dict[str, Any]] = []

    for epoch in range(1, config.train.num_epochs + 1):
        model.train()
        epoch_losses: list[float] = []
        train_preds: list[np.ndarray] = []
        train_targets_list: list[np.ndarray] = []

        for probs, targets in train_loader:
            probs = probs.to(device)
            targets = targets.to(device)

            optimizer.zero_grad(set_to_none=True)
            preds = model(probs)
            loss = weighted_mse_loss(preds, probs, recommended_tensor)
            loss.backward()
            optimizer.step()

            epoch_losses.append(float(loss.detach().cpu().item()))
            train_preds.append(preds.detach().cpu().numpy())
            train_targets_list.append(targets.detach().cpu().numpy())

        train_loss = float(np.mean(epoch_losses)) if epoch_losses else 0.0
        train_mae_axis = _mae_by_axis(
            np.vstack(train_preds),
            np.vstack(train_targets_list),
        )

        val_loss, val_mae_axis = _run_eval(
            model,
            val_loader,
            device,
            recommended_tensor,
        )
        scheduler.step(val_loss)

        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "train_mae_macro": float(np.mean(train_mae_axis)),
                "val_mae_macro": float(np.mean(val_mae_axis)),
                "train_mae_axis": {
                    name: float(value)
                    for name, value in zip(
                        CONTROL_PARAM_NAMES,
                        train_mae_axis,
                        strict=True,
                    )
                },
                "val_mae_axis": {
                    name: float(value)
                    for name, value in zip(
                        CONTROL_PARAM_NAMES,
                        val_mae_axis,
                        strict=True,
                    )
                },
            },
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            early_stop = 0
            torch.save(
                {
                    "model_type": "parameter_generator",
                    "model_state_dict": model.state_dict(),
                    "recommended_params": recommended_matrix.tolist(),
                    "emotion_labels": JVNV_EMOTION_LABELS,
                    "control_param_names": CONTROL_PARAM_NAMES,
                    "training_strategy": config.data.strategy,
                    "nearest_k": 25,
                    "config": config.to_dict(),
                },
                best_path,
            )
        else:
            early_stop += 1

        if early_stop >= config.train.early_stopping_patience:
            break

    checkpoint = torch.load(best_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    final_val_loss, final_mae_axis = _run_eval(
        model,
        val_loader,
        device,
        recommended_tensor,
    )

    mae_payload = {
        "axis_mae": {
            name: float(value)
            for name, value in zip(CONTROL_PARAM_NAMES, final_mae_axis, strict=True)
        },
        "macro_mae": float(np.mean(final_mae_axis)),
        "threshold": float(config.eval.mae_axis_max),
        "axis_pass": {
            name: bool(value <= config.eval.mae_axis_max)
            for name, value in zip(CONTROL_PARAM_NAMES, final_mae_axis, strict=True)
        },
    }

    go_no_go = {
        "all_axis_mae": bool(all(mae_payload["axis_pass"].values())),
        "go": bool(all(mae_payload["axis_pass"].values())),
    }

    save_effective_generator_config(config, output_dir / "effective_config.yaml")

    with (reports_dir / "training_history.json").open("w", encoding="utf-8") as file:
        json.dump(history, file, ensure_ascii=False, indent=2)

    with (reports_dir / "parameter_mae.json").open("w", encoding="utf-8") as file:
        json.dump(mae_payload, file, ensure_ascii=False, indent=2)

    with (reports_dir / "evaluation.json").open("w", encoding="utf-8") as file:
        json.dump(
            {
                "best_epoch": best_epoch,
                "best_val_loss": best_val_loss,
                "final_val_loss": final_val_loss,
                "mae": mae_payload,
                "go_no_go": go_no_go,
            },
            file,
            ensure_ascii=False,
            indent=2,
        )

    return {
        "output_dir": str(output_dir),
        "checkpoint": str(best_path),
        "best_epoch": int(best_epoch),
        "best_val_loss": float(best_val_loss),
        "final_val_loss": float(final_val_loss),
        "mae": mae_payload,
        "go_no_go": go_no_go,
        "num_train_samples": len(train_ds),
        "num_val_samples": len(val_ds),
    }
