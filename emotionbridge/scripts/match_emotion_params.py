"""感情-制御パラメータマッチングスクリプト。

JVNV の感情クラスタ重心に近い VOICEVOX サンプルを探索し、
感情ごとの推奨制御パラメータ（中央値）を算出する。
cross_domain_alignment 有効時は aligned 特徴量を、
distance_metric=weighted_euclidean 時は偏相関ベースの特徴量重みを使用する。
"""

from __future__ import annotations

import argparse
import logging
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

from emotionbridge.scripts.common import (
    load_experiment_config,
    read_parquet,
    resolve_path,
    save_json,
    save_markdown,
    write_parquet,
)

if TYPE_CHECKING:
    from pathlib import Path

logger = logging.getLogger(__name__)


def _configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="感情クラスタ重心→VOICEVOX制御パラメータ探索",
    )
    parser.add_argument(
        "--config",
        default="configs/experiment_config.yaml",
        help="実験設定ファイル",
    )
    parser.add_argument("--jvnv-normalized", default=None, help="JVNV正規化特徴量")
    parser.add_argument(
        "--voicevox-normalized",
        default=None,
        help="VOICEVOX正規化特徴量",
    )
    parser.add_argument("--nearest-k", type=int, default=None, help="近傍件数")
    return parser


def _feature_columns(df: pd.DataFrame) -> list[str]:
    return sorted([name for name in df.columns if name.startswith("egemaps__")])


def _load_feature_weights(
    v02_dir: Path,
    feature_cols: list[str],
) -> np.ndarray | None:
    """V-02の偏相関から算出した特徴量重みを読み込む。

    VOICEVOXの5D制御で動かせない次元（jitter/shimmer等）の重みを低くし、
    距離計算が制御不能な方向に引きずられるのを防ぐ。
    """
    import json

    weights_path = v02_dir / "feature_weights.json"
    if not weights_path.exists():
        return None

    with weights_path.open("r", encoding="utf-8") as f:
        payload = json.load(f)

    weights_dict = payload.get("weights", {})
    weights = np.array(
        [float(weights_dict.get(col, 0.0)) for col in feature_cols],
        dtype=np.float32,
    )

    total = weights.sum()
    if total > 0:
        weights = weights * len(feature_cols) / total
    return weights


def _control_columns(df: pd.DataFrame) -> list[str]:
    return sorted([name for name in df.columns if name.startswith("ctrl_")])


def _feature_mapping(feature_names: list[str], values: np.ndarray) -> dict[str, float]:
    return {
        feature: float(value)
        for feature, value in zip(feature_names, values, strict=False)
    }


def _build_jvnv_profiles_payload(
    *,
    jvnv_df: pd.DataFrame,
    feature_cols: list[str],
    target_emotions: list[str],
    jvnv_path: Path,
    v01_dir: Path,
) -> dict[str, Any]:
    covariance_dir = v01_dir / "covariance"
    covariance_dir.mkdir(parents=True, exist_ok=True)

    profiles: dict[str, Any] = {}
    for emotion in target_emotions:
        subset = jvnv_df[jvnv_df["emotion_common6"] == emotion]
        if subset.empty:
            continue

        feature_matrix = subset[feature_cols].to_numpy(dtype=np.float64, copy=False)
        centroid = feature_matrix.mean(axis=0)
        stddev = feature_matrix.std(axis=0, ddof=0)
        if len(feature_matrix) > 1:
            covariance = np.cov(feature_matrix, rowvar=False, bias=True)
        else:
            covariance = np.zeros(
                (len(feature_cols), len(feature_cols)),
                dtype=np.float64,
            )

        covariance_path = covariance_dir / f"{emotion}_cov.npy"
        np.save(covariance_path, covariance)

        profiles[emotion] = {
            "num_samples": len(subset),
            "centroid": _feature_mapping(feature_cols, centroid),
            "stddev": _feature_mapping(feature_cols, stddev),
            "covariance_diag": [float(value) for value in np.diag(covariance)],
            "covariance_full_path": str(covariance_path),
        }

    return {
        "version": "1.0",
        "feature_set": "eGeMAPSv02",
        "feature_count": len(feature_cols),
        "normalization": "speaker_zscore",
        "source_normalized_path": str(jvnv_path),
        "profiles": profiles,
    }


def _build_recommended_params_payload(
    *,
    summary: dict[str, Any],
    target_emotions: list[str],
    nearest_k: int,
) -> dict[str, Any]:
    emotions: dict[str, dict[str, float]] = {}
    for emotion in target_emotions:
        emotion_summary = summary.get(emotion)
        if not emotion_summary:
            continue

        control_summary = emotion_summary["control_summary"]
        emotions[emotion] = {
            control: float(values["median"])
            for control, values in control_summary.items()
        }

    return {
        "version": "1.0",
        "nearest_k": int(nearest_k),
        "aggregation": "median",
        "emotions": emotions,
    }


def _canon_jvnv(label: str) -> str | None:
    lowered = str(label).strip().lower()
    mapping = {
        "anger": "anger",
        "angry": "anger",
        "ang": "anger",
        "disgust": "disgust",
        "disgusted": "disgust",
        "dis": "disgust",
        "fear": "fear",
        "fearful": "fear",
        "fea": "fear",
        "happy": "happy",
        "joy": "happy",
        "hap": "happy",
        "sad": "sad",
        "sadness": "sad",
        "surprise": "surprise",
        "surprised": "surprise",
        "sur": "surprise",
    }
    return mapping.get(lowered)


def _canon_voicevox(label: str) -> str | None:
    lowered = str(label).strip().lower()
    mapping = {
        "joy": "happy",
        "happy": "happy",
        "sadness": "sad",
        "sad": "sad",
        "surprise": "surprise",
        "anger": "anger",
        "fear": "fear",
        "disgust": "disgust",
    }
    return mapping.get(lowered)


def run_matching(
    *,
    config_path: str,
    jvnv_normalized: str | None,
    voicevox_normalized: str | None,
    nearest_k: int | None,
) -> dict[str, Any]:
    """感情ごとの JVNV 重心に最近傍の VOICEVOX サンプルを探索する。

    cross_domain_alignment=True の場合は aligned parquet を入力に使う。
    distance_metric=weighted_euclidean の場合は feature_weights.json の
    偏相関ベース重みで距離計算を行い、制御不能な方向の影響を抑制する。
    """
    config = load_experiment_config(config_path)
    v01_dir = resolve_path(config.v01.output_dir)
    v02_dir = resolve_path(config.v02.output_dir)
    v03_dir = resolve_path(config.v03.output_dir)
    v03_dir.mkdir(parents=True, exist_ok=True)

    if jvnv_normalized:
        jvnv_path = resolve_path(jvnv_normalized)
    elif config.v03.cross_domain_alignment:
        jvnv_path = v03_dir / "jvnv_egemaps_aligned.parquet"
    else:
        jvnv_path = v01_dir / "jvnv_egemaps_normalized.parquet"

    if voicevox_normalized:
        voice_path = resolve_path(voicevox_normalized)
    elif config.v03.cross_domain_alignment:
        voice_path = v03_dir / "voicevox_egemaps_aligned.parquet"
    else:
        voice_path = v02_dir / "voicevox_egemaps_normalized.parquet"
    if not jvnv_path.exists() or not voice_path.exists():
        msg = f"Input file missing: jvnv={jvnv_path.exists()}, voicevox={voice_path.exists()}"
        raise FileNotFoundError(msg)

    jvnv_df = read_parquet(jvnv_path).copy()
    voice_df = read_parquet(voice_path).copy()

    jvnv_df["emotion_common6"] = jvnv_df["emotion"].map(_canon_jvnv)
    voice_df["emotion_common6"] = voice_df["dominant_emotion"].map(_canon_voicevox)
    jvnv_df = jvnv_df[jvnv_df["emotion_common6"].notna()].reset_index(drop=True)
    voice_df = voice_df[voice_df["emotion_common6"].notna()].reset_index(drop=True)

    feature_cols = sorted(
        set(_feature_columns(jvnv_df)).intersection(_feature_columns(voice_df)),
    )
    if not feature_cols:
        msg = "No common eGeMAPS feature columns found"
        raise ValueError(msg)
    control_cols = _control_columns(voice_df)

    target_emotions = config.v03.emotion_labels_common6
    top_k = nearest_k if nearest_k is not None else config.v03.nearest_k

    feature_weights: np.ndarray | None = None
    distance_metric = config.v03.distance_metric
    if distance_metric == "weighted_euclidean":
        feature_weights = _load_feature_weights(v02_dir, feature_cols)
        if feature_weights is None:
            logger.warning(
                "重み付き距離を要求されたが feature_weights.json が見つからない: %s"
                " → unweighted L2にフォールバック",
                v02_dir,
            )
            distance_metric = "euclidean"

    matched_rows: list[dict[str, Any]] = []
    summary: dict[str, Any] = {}

    voice_features = voice_df[feature_cols].to_numpy(dtype=np.float32, copy=False)

    for emotion in target_emotions:
        j_subset = jvnv_df[jvnv_df["emotion_common6"] == emotion]
        if j_subset.empty:
            continue

        centroid = (
            j_subset[feature_cols].to_numpy(dtype=np.float32, copy=False).mean(axis=0)
        )
        diff = voice_features - centroid[None, :]
        if distance_metric == "weighted_euclidean" and feature_weights is not None:
            distances = np.sqrt(np.sum(feature_weights[None, :] * diff**2, axis=1))
        else:
            distances = np.linalg.norm(diff, axis=1)
        nearest_indices = np.argsort(distances)[:top_k]
        nearest_df = voice_df.iloc[nearest_indices].copy()
        nearest_df["target_emotion"] = emotion
        nearest_df["distance_to_centroid"] = distances[nearest_indices]
        matched_rows.extend(nearest_df.to_dict(orient="records"))

        controls_summary: dict[str, dict[str, float]] = {}
        for control in control_cols:
            values = nearest_df[control].to_numpy(dtype=np.float64)
            controls_summary[control] = {
                "mean": float(np.mean(values)),
                "median": float(np.median(values)),
                "std": float(np.std(values, ddof=0)),
                "min": float(np.min(values)),
                "max": float(np.max(values)),
            }

        summary[emotion] = {
            "num_jvnv_samples": len(j_subset),
            "nearest_k": int(top_k),
            "distance_mean": float(np.mean(distances[nearest_indices])),
            "distance_std": float(np.std(distances[nearest_indices], ddof=0)),
            "control_summary": controls_summary,
        }

    if not matched_rows:
        msg = "No matching results generated"
        raise RuntimeError(msg)

    matched_df = pd.DataFrame(matched_rows)
    output_matches = v03_dir / "emotion_param_matches.parquet"
    write_parquet(matched_df, output_matches)

    payload = {
        "jvnv_path": str(jvnv_path),
        "voicevox_path": str(voice_path),
        "feature_count": len(feature_cols),
        "nearest_k": int(top_k),
        "distance_metric": distance_metric,
        "cross_domain_alignment": config.v03.cross_domain_alignment,
        "summary": summary,
        "output_matches": str(output_matches),
    }
    output_json = v03_dir / "emotion_param_matches.json"
    save_json(payload, output_json)

    profiles_payload = _build_jvnv_profiles_payload(
        jvnv_df=jvnv_df,
        feature_cols=feature_cols,
        target_emotions=target_emotions,
        jvnv_path=jvnv_path,
        v01_dir=v01_dir,
    )
    profiles_path = v01_dir / "jvnv_emotion_profiles.json"
    save_json(profiles_payload, profiles_path)

    recommended_params_payload = _build_recommended_params_payload(
        summary=summary,
        target_emotions=target_emotions,
        nearest_k=top_k,
    )
    recommended_params_path = v03_dir / "recommended_params.json"
    save_json(recommended_params_payload, recommended_params_path)

    payload["output_summary"] = str(output_json)
    payload["jvnv_profiles"] = str(profiles_path)
    payload["recommended_params"] = str(recommended_params_path)

    report_lines = [
        "# Emotion-Parameter Matching Report",
        "",
        f"- JVNV: {jvnv_path}",
        f"- VOICEVOX: {voice_path}",
        f"- Common features: {len(feature_cols)}",
        f"- Nearest k: {top_k}",
        "",
        "## Per-emotion Summary",
    ]
    for emotion, values in summary.items():
        report_lines.extend(
            (
                f"### {emotion}",
                f"- distance mean: {values['distance_mean']:.6f}",
                f"- distance std: {values['distance_std']:.6f}",
            ),
        )
        for control, control_values in values["control_summary"].items():
            report_lines.append(
                "- "
                f"{control}: mean={control_values['mean']:.4f}, "
                f"median={control_values['median']:.4f}, "
                f"std={control_values['std']:.4f}, "
                f"range=[{control_values['min']:.4f}, {control_values['max']:.4f}]",
            )
        report_lines.append("")

    save_markdown("\n".join(report_lines), v03_dir / "emotion_param_matching_report.md")
    return payload


def main() -> None:
    _configure_logging()
    parser = _build_parser()
    args = parser.parse_args()

    summary = run_matching(
        config_path=args.config,
        jvnv_normalized=args.jvnv_normalized,
        voicevox_normalized=args.voicevox_normalized,
        nearest_k=args.nearest_k,
    )
    logger.info("感情マッチング完了: %s", summary["output_matches"])


if __name__ == "__main__":
    main()
