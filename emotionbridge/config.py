from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass(slots=True)
class DataConfig:
    dataset_name: str = "shunk031/wrime"
    dataset_config_name: str = "ver1"
    text_field: str = "sentence"
    label_source: str = "avg_readers"
    max_length: int = 128
    use_official_split: bool = False
    filter_max_intensity_lte: int = 1
    stratify_after_filter: bool = True
    train_ratio: float = 0.8
    val_ratio: float = 0.1
    test_ratio: float = 0.1
    random_seed: int = 42


@dataclass(slots=True)
class ModelConfig:
    pretrained_model_name: str = "cl-tohoku/bert-base-japanese-whole-word-masking"
    bottleneck_dim: int = 256
    dropout: float = 0.1


@dataclass(slots=True)
class TrainConfig:
    output_dir: str = "artifacts/phase0"
    batch_size: int = 32
    num_epochs: int = 10
    bert_lr: float = 2e-5
    head_lr: float = 1e-3
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1
    early_stopping_patience: int = 3
    device: str = "cuda"
    num_workers: int = 2
    pin_memory: bool = True
    log_every_steps: int = 50
    emotion_weights: list[float] | None = None


@dataclass(slots=True)
class EvalConfig:
    go_macro_mse_max: float = 0.05
    go_min_pearson: float = 0.5
    go_top1_acc_min: float = 0.6


@dataclass(slots=True)
class Phase0Config:
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _build_section(section_type: type, payload: dict[str, Any] | None):
    payload = payload or {}
    return section_type(**payload)


def load_config(config_path: str | Path) -> Phase0Config:
    path = Path(config_path)
    if not path.exists():
        msg = f"Config file not found: {path}"
        raise FileNotFoundError(msg)

    with path.open("r", encoding="utf-8") as file:
        raw = yaml.safe_load(file) or {}

    return Phase0Config(
        data=_build_section(DataConfig, raw.get("data")),
        model=_build_section(ModelConfig, raw.get("model")),
        train=_build_section(TrainConfig, raw.get("train")),
        eval=_build_section(EvalConfig, raw.get("eval")),
    )


def save_effective_config(config: Phase0Config, output_path: str | Path) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        yaml.safe_dump(config.to_dict(), file, sort_keys=False, allow_unicode=True)
