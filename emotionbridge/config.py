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
class ClassifierDataConfig(DataConfig):
    label_conversion: str = "argmax"
    soft_label_temperature: float = 1.0


@dataclass(slots=True)
class ClassifierModelConfig(ModelConfig):
    num_classes: int = 6
    transfer_from: str | None = None


@dataclass(slots=True)
class ClassifierTrainConfig:
    output_dir: str = "artifacts/classifier"
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
    gradient_accumulation_steps: int = 1
    mixed_precision: str = "no"
    class_weight_mode: str = "inverse_frequency"
    class_weights: list[float] | None = None
    use_weighted_sampler: bool = False


@dataclass(slots=True)
class ClassifierEvalConfig:
    go_macro_f1_min: float = 0.40
    go_key_emotion_f1_min: float = 0.50
    key_emotions: list[str] = field(
        default_factory=lambda: ["anger", "happy", "sad"],
    )


@dataclass(slots=True)
class ClassifierConfig:
    data: ClassifierDataConfig = field(default_factory=ClassifierDataConfig)
    model: ClassifierModelConfig = field(default_factory=ClassifierModelConfig)
    train: ClassifierTrainConfig = field(default_factory=ClassifierTrainConfig)
    eval: ClassifierEvalConfig = field(default_factory=ClassifierEvalConfig)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# --- 音声サンプル生成設定 ---


@dataclass(slots=True)
class VoicevoxConfig:
    """VOICEVOX Engine接続設定。"""

    host: str = "127.0.0.1"
    port: int = 50021
    default_speaker_id: int = 0
    speaker_ids: list[int] | None = None
    timeout: float = 30.0
    max_retries: int = 3
    retry_delay: float = 0.5
    output_sampling_rate: int = 24000
    output_stereo: bool = False

    @property
    def base_url(self) -> str:
        return f"http://{self.host}:{self.port}"


@dataclass(slots=True)
class ControlSpaceConfig:
    """制御空間のマッピング範囲設定。

    デフォルト値はProposal Section 3.2に基づく保守的な範囲。
    """

    speed_range: tuple[float, float] = (0.5, 1.5)
    pitch_shift_range: tuple[float, float] = (-0.15, 0.15)
    pitch_range_range: tuple[float, float] = (0.5, 1.5)
    energy_range: tuple[float, float] = (0.5, 1.5)
    pause_pre_range: tuple[float, float] = (0.0, 0.2)
    pause_post_range: tuple[float, float] = (0.0, 0.2)
    pause_length_scale_range: tuple[float, float] = (0.5, 1.5)


@dataclass(slots=True)
class GridConfig:
    """パラメータグリッド設定。"""

    strategy: str = "lhs"
    lhs_samples_per_text: int = 128
    grid_steps: int = 5
    random_seed: int = 42


@dataclass(slots=True)
class TextSelectionConfig:
    """テキスト選定設定。"""

    num_texts: int = 200
    texts_per_emotion: int = 25
    intensity_bins: int = 3
    min_text_length: int = 5
    max_text_length: int = 200
    random_seed: int = 42


@dataclass(slots=True)
class ValidationConfig:
    """音声品質検証設定。"""

    min_file_size_bytes: int = 1024
    min_duration_seconds: float = 0.1
    min_rms_amplitude: float = 0.001
    expected_sample_rate: int = 24000


@dataclass(slots=True)
class GenerationConfig:
    """音声生成パイプライン設定。"""

    output_dir: str = "artifacts/audio_gen"
    audio_subdir: str = "audio"
    max_concurrent_requests: int = 4
    checkpoint_interval: int = 100
    skip_existing: bool = True


@dataclass(slots=True)
class AudioGenConfig:
    """音声サンプル生成パイプラインの設定。"""

    classifier_checkpoint: str = "artifacts/classifier/checkpoints/best_model.pt"
    device: str = "cuda"
    voicevox: VoicevoxConfig = field(default_factory=VoicevoxConfig)
    control_space: ControlSpaceConfig = field(default_factory=ControlSpaceConfig)
    grid: GridConfig = field(default_factory=GridConfig)
    text_selection: TextSelectionConfig = field(default_factory=TextSelectionConfig)
    validation: ValidationConfig = field(default_factory=ValidationConfig)
    generation: GenerationConfig = field(default_factory=GenerationConfig)
    data: DataConfig = field(default_factory=DataConfig)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _build_section(section_type: type, payload: dict[str, Any] | None):
    import dataclasses

    payload = payload or {}
    field_map = {f.name: f for f in dataclasses.fields(section_type)}
    valid = {}
    for k, v in payload.items():
        if isinstance(v, list) and len(v) == 2:
            f = field_map.get(k)
            if f and "tuple" in str(f.type):
                valid[k] = tuple(v)
                continue
        valid[k] = v
    return section_type(**valid)


def load_config(
    config_path: str | Path,
) -> ClassifierConfig | AudioGenConfig:
    """YAML設定ファイルを読み込む。

    - 'voicevox' キーが存在すれば AudioGenConfig
    - それ以外は ClassifierConfig
    """
    path = Path(config_path)
    if not path.exists():
        msg = f"Config file not found: {path}"
        raise FileNotFoundError(msg)

    with path.open("r", encoding="utf-8") as file:
        raw = yaml.safe_load(file) or {}

    if "voicevox" in raw:
        return AudioGenConfig(
            classifier_checkpoint=raw.get(
                "classifier_checkpoint",
                "artifacts/classifier/checkpoints/best_model.pt",
            ),
            device=raw.get("device", "cuda"),
            voicevox=_build_section(VoicevoxConfig, raw.get("voicevox")),
            control_space=_build_section(ControlSpaceConfig, raw.get("control_space")),
            grid=_build_section(GridConfig, raw.get("grid")),
            text_selection=_build_section(
                TextSelectionConfig,
                raw.get("text_selection"),
            ),
            validation=_build_section(ValidationConfig, raw.get("validation")),
            generation=_build_section(GenerationConfig, raw.get("generation")),
            data=_build_section(DataConfig, raw.get("data")),
        )

    model_raw = raw.get("model") or {}
    data_raw = raw.get("data") or {}
    train_raw = raw.get("train") or {}
    if (
        "num_classes" in model_raw
        or "label_conversion" in data_raw
        or "class_weight_mode" in train_raw
    ):
        return ClassifierConfig(
            data=_build_section(ClassifierDataConfig, data_raw),
            model=_build_section(ClassifierModelConfig, model_raw),
            train=_build_section(ClassifierTrainConfig, train_raw),
            eval=_build_section(ClassifierEvalConfig, raw.get("eval")),
        )

    msg = (
        "Unknown config format. Expected AudioGenConfig (voicevox key) "
        "or ClassifierConfig (num_classes/label_conversion/class_weight_mode key)."
    )
    raise ValueError(msg)


def save_effective_config(
    config: ClassifierConfig | AudioGenConfig,
    output_path: str | Path,
) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        yaml.safe_dump(config.to_dict(), file, sort_keys=False, allow_unicode=True)
