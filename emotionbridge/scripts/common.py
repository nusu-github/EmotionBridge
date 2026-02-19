from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd
import yaml


@dataclass(slots=True)
class PathsConfig:
    jvnv_root: str = "data/jvnv_v1"
    jvnv_nv_label_dir: str = "data/jvnv_v1/nv_label"
    jvnv_processed_audio_dir: str = "artifacts/phase3/v01/jvnv_processed_audio"
    phase1_dataset_path: str = "artifacts/phase1/dataset/triplet_dataset.parquet"
    phase1_output_dir: str = "artifacts/phase1"
    output_root: str = "artifacts/phase3"


@dataclass(slots=True)
class V01Config:
    output_dir: str = "artifacts/phase3/v01"
    overwrite_existing_nv_masked: bool = False
    equalize_speaker_stats: bool = True
    random_seed: int = 42
    tsne_perplexities: list[float] = field(default_factory=lambda: [10.0, 30.0, 50.0])
    nv_handling: str = "excise"


@dataclass(slots=True)
class V02Config:
    output_dir: str = "artifacts/phase3/v02"
    speaker_mode: str = "style_id"
    random_seed: int = 42
    tsne_perplexities: list[float] = field(default_factory=lambda: [10.0, 30.0, 50.0])


@dataclass(slots=True)
class V03Config:
    output_dir: str = "artifacts/phase3/v03"
    emotion_labels_common6: list[str] = field(
        default_factory=lambda: [
            "anger",
            "disgust",
            "fear",
            "happy",
            "sad",
            "surprise",
        ],
    )
    nearest_k: int = 25
    random_seed: int = 42
    tsne_perplexity: float = 30.0
    cross_domain_alignment: bool = True
    distance_metric: str = "weighted_euclidean"
    feature_weight_min_corr: float = 0.05


@dataclass(slots=True)
class ExtractionConfig:
    feature_set: str = "eGeMAPSv02"
    feature_level: str = "Functionals"


@dataclass(slots=True)
class EvaluationConfig:
    silhouette_go_threshold: float = 0.15
    tsne_visual_separation_min_emotions: int = 4


@dataclass(slots=True)
class ExperimentConfig:
    paths: PathsConfig = field(default_factory=PathsConfig)
    extraction: ExtractionConfig = field(default_factory=ExtractionConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    v01: V01Config = field(default_factory=V01Config)
    v02: V02Config = field(default_factory=V02Config)
    v03: V03Config = field(default_factory=V03Config)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _build_section(section_type: type, payload: dict[str, Any] | None):
    payload = payload or {}
    return section_type(**payload)


def load_experiment_config(config_path: str | Path) -> ExperimentConfig:
    path = Path(config_path)
    if not path.exists():
        msg = f"Config file not found: {path}"
        raise FileNotFoundError(msg)

    with path.open("r", encoding="utf-8") as file:
        raw = yaml.safe_load(file) or {}

    return ExperimentConfig(
        paths=_build_section(PathsConfig, raw.get("paths")),
        extraction=_build_section(ExtractionConfig, raw.get("extraction")),
        evaluation=_build_section(EvaluationConfig, raw.get("evaluation")),
        v01=_build_section(V01Config, raw.get("v01")),
        v02=_build_section(V02Config, raw.get("v02")),
        v03=_build_section(V03Config, raw.get("v03")),
    )


def save_json(payload: dict[str, Any], output_path: str | Path) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, ensure_ascii=False, indent=2)


def save_markdown(content: str, output_path: str | Path) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def read_parquet(path: str | Path) -> pd.DataFrame:
    return pd.read_parquet(Path(path))


def write_parquet(df: pd.DataFrame, path: str | Path) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out, index=False)


def resolve_path(path_like: str | Path, base_dir: str | Path | None = None) -> Path:
    path = Path(path_like)
    if path.is_absolute():
        return path
    if base_dir is None:
        return (Path.cwd() / path).resolve()
    return (Path(base_dir) / path).resolve()


def ensure_columns(df: pd.DataFrame, required: list[str], *, where: str) -> None:
    missing = [name for name in required if name not in df.columns]
    if missing:
        msg = f"Missing required columns in {where}: {missing}"
        raise ValueError(msg)
