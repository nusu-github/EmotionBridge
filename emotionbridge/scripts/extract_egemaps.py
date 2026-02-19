from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Any

import pandas as pd

from emotionbridge.scripts.common import (
    ensure_columns,
    load_experiment_config,
    read_parquet,
    resolve_path,
    save_json,
    write_parquet,
)

logger = logging.getLogger(__name__)


def _configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="eGeMAPS 一括抽出スクリプト")
    parser.add_argument(
        "--config",
        default="configs/experiment_config.yaml",
        help="実験設定ファイル",
    )
    parser.add_argument(
        "--source",
        required=True,
        choices=["jvnv", "voicevox"],
        help="抽出対象",
    )
    parser.add_argument(
        "--jvnv-audio-key",
        choices=["processed", "raw"],
        default="processed",
        help="JVNV抽出時の入力音声種別",
    )
    return parser


def _resolve_opensmile_enums(feature_set: str, feature_level: str):
    try:
        import opensmile
    except ImportError as exc:
        msg = "opensmile がインストールされていません。`uv add opensmile` を実行してください。"
        raise ImportError(msg) from exc

    feature_set_map = {
        "eGeMAPSv02": opensmile.FeatureSet.eGeMAPSv02,
    }
    feature_level_map = {
        "Functionals": opensmile.FeatureLevel.Functionals,
    }
    if feature_set not in feature_set_map:
        msg = f"Unsupported feature_set: {feature_set}"
        raise ValueError(msg)
    if feature_level not in feature_level_map:
        msg = f"Unsupported feature_level: {feature_level}"
        raise ValueError(msg)
    return feature_set_map[feature_set], feature_level_map[feature_level]


def _build_extractor(config_path: str):
    config = load_experiment_config(config_path)
    feature_set, feature_level = _resolve_opensmile_enums(
        config.extraction.feature_set,
        config.extraction.feature_level,
    )
    import opensmile

    return opensmile.Smile(feature_set=feature_set, feature_level=feature_level)


def _extract_row(smile, audio_path: Path) -> dict[str, float]:
    feature_df = smile.process_file(str(audio_path))
    if feature_df.empty:
        msg = f"openSMILE returned empty output for {audio_path}"
        raise RuntimeError(msg)
    row = feature_df.iloc[0].to_dict()
    return {f"egemaps__{key}": float(value) for key, value in row.items()}


def _resolve_audio_path(
    audio_path: str,
    *,
    audio_gen_output_dir: Path,
    dataset_dir: Path,
) -> Path | None:
    candidate = Path(audio_path)
    if candidate.is_absolute():
        return candidate if candidate.exists() else None

    for path in [
        audio_gen_output_dir / candidate,
        dataset_dir / candidate,
        Path.cwd() / candidate,
    ]:
        if path.exists():
            return path
    return None


def _extract_jvnv(config_path: str, jvnv_audio_key: str) -> dict[str, Any]:
    config = load_experiment_config(config_path)
    v01_dir = resolve_path(config.v01.output_dir)
    manifest_path = v01_dir / "jvnv_manifest.parquet"
    if not manifest_path.exists():
        msg = f"JVNV manifest not found: {manifest_path}. 先に `python -m emotionbridge.scripts.prepare_jvnv` を実行してください。"
        raise FileNotFoundError(msg)

    manifest_df = read_parquet(manifest_path)
    required = [
        "utterance_id",
        "speaker",
        "emotion",
        "session",
        "audio_path_raw",
        "audio_path_processed",
    ]
    ensure_columns(manifest_df, required, where="JVNV manifest")

    smile = _build_extractor(config_path)

    rows: list[dict[str, Any]] = []
    missing_files = 0
    failures = 0
    audio_col = "audio_path_processed" if jvnv_audio_key == "processed" else "audio_path_raw"

    for _, sample in manifest_df.iterrows():
        audio_path = Path(str(sample[audio_col]))
        if not audio_path.exists():
            missing_files += 1
            continue

        try:
            feature_dict = _extract_row(smile, audio_path)
        except Exception:
            failures += 1
            continue

        rows.append(
            {
                "source_domain": "jvnv",
                "variant": jvnv_audio_key,
                "utterance_id": str(sample["utterance_id"]),
                "speaker": str(sample["speaker"]),
                "emotion": str(sample["emotion"]),
                "session": str(sample["session"]),
                "audio_path": str(audio_path),
                **feature_dict,
            },
        )

    if not rows:
        msg = "No JVNV features were extracted."
        raise RuntimeError(msg)

    feature_df = pd.DataFrame(rows)
    output_name = (
        "jvnv_egemaps_raw.parquet"
        if jvnv_audio_key == "processed"
        else "jvnv_egemaps_with_nv_raw.parquet"
    )
    output_path = v01_dir / output_name
    write_parquet(feature_df, output_path)

    summary = {
        "source": "jvnv",
        "variant": jvnv_audio_key,
        "input_manifest": str(manifest_path),
        "output_path": str(output_path),
        "num_rows": len(feature_df),
        "num_features": len(
            [c for c in feature_df.columns if c.startswith("egemaps__")],
        ),
        "missing_audio_files": int(missing_files),
        "failed_extractions": int(failures),
    }
    save_json(summary, v01_dir / f"extract_egemaps_jvnv_{jvnv_audio_key}_summary.json")
    return summary


def _extract_voicevox(config_path: str) -> dict[str, Any]:
    config = load_experiment_config(config_path)
    triplet_dataset_path = resolve_path(config.paths.triplet_dataset_path)
    audio_gen_output_dir = resolve_path(config.paths.audio_gen_output_dir)
    v02_dir = resolve_path(config.v02.output_dir)
    v02_dir.mkdir(parents=True, exist_ok=True)

    if not triplet_dataset_path.exists():
        msg = f"Triplet dataset not found: {triplet_dataset_path}"
        raise FileNotFoundError(msg)

    dataset_df = read_parquet(triplet_dataset_path)
    ensure_columns(dataset_df, ["audio_path"], where="triplet dataset")

    smile = _build_extractor(config_path)

    dataset_dir = triplet_dataset_path.parent.parent
    rows: list[dict[str, Any]] = []
    missing_files = 0
    failures = 0

    metadata_columns = [
        "text_id",
        "text",
        "dominant_emotion",
        "style_id",
        "audio_path",
        "generation_timestamp",
    ]
    passthrough_columns = [
        name for name in dataset_df.columns if name.startswith(("ctrl_", "emotion_", "vv_"))
    ]

    for _, sample in dataset_df.iterrows():
        resolved = _resolve_audio_path(
            str(sample["audio_path"]),
            audio_gen_output_dir=audio_gen_output_dir,
            dataset_dir=dataset_dir,
        )
        if resolved is None:
            missing_files += 1
            continue

        try:
            feature_dict = _extract_row(smile, resolved)
        except Exception:
            failures += 1
            continue

        row: dict[str, Any] = {
            "source_domain": "voicevox",
            "audio_path": str(resolved),
        }
        for column in metadata_columns:
            if column in sample.index:
                row[column] = sample[column]
        for column in passthrough_columns:
            row[column] = sample[column]
        row.update(feature_dict)
        rows.append(row)

    if not rows:
        msg = "No VOICEVOX features were extracted."
        raise RuntimeError(msg)

    feature_df = pd.DataFrame(rows)
    output_path = v02_dir / "voicevox_egemaps_raw.parquet"
    write_parquet(feature_df, output_path)

    summary = {
        "source": "voicevox",
        "input_dataset": str(triplet_dataset_path),
        "output_path": str(output_path),
        "num_rows": len(feature_df),
        "num_features": len(
            [c for c in feature_df.columns if c.startswith("egemaps__")],
        ),
        "missing_audio_files": int(missing_files),
        "failed_extractions": int(failures),
    }
    save_json(summary, v02_dir / "extract_egemaps_voicevox_summary.json")
    return summary


def main() -> None:
    _configure_logging()
    parser = _build_parser()
    args = parser.parse_args()

    summary = (
        _extract_jvnv(args.config, args.jvnv_audio_key)
        if args.source == "jvnv"
        else _extract_voicevox(args.config)
    )

    logger.info("eGeMAPS抽出完了: %s", summary["output_path"])


if __name__ == "__main__":
    main()
