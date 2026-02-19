"""クロスドメイン整合スクリプト。

JVNV（実音声）と VOICEVOX（合成音声）の per-speaker 正規化済み eGeMAPS 特徴量を
pooled z-score で再正規化し、ドメイン間の距離比較を有意にする。
"""

from __future__ import annotations

import argparse
import logging
from typing import TYPE_CHECKING, Any

import numpy as np

from emotionbridge.scripts.common import (
    load_experiment_config,
    read_parquet,
    resolve_path,
    save_json,
    write_parquet,
)

if TYPE_CHECKING:
    import pandas as pd

logger = logging.getLogger(__name__)


def _configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="クロスドメイン正規化: JVNV/VOICEVOXのpooled z-score整合",
    )
    parser.add_argument(
        "--config",
        default="configs/experiment_config.yaml",
        help="実験設定ファイル",
    )
    parser.add_argument(
        "--jvnv-normalized",
        default=None,
        help="JVNV正規化済み特徴量parquet",
    )
    parser.add_argument(
        "--voicevox-normalized",
        default=None,
        help="VOICEVOX正規化済み特徴量parquet",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="出力先ディレクトリ",
    )
    return parser


def _collect_feature_columns(df: pd.DataFrame) -> list[str]:
    return sorted([name for name in df.columns if name.startswith("egemaps__")])


def _cross_domain_normalize(
    jvnv_df: pd.DataFrame,
    voicevox_df: pd.DataFrame,
    feature_cols: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Stage 2: pooled z-score でクロスドメイン整合を行う。

    Stage 1（per-speaker/style z-score）で話者・スタイル効果は除去済み。
    ここでは両ドメインをプールし、共通の μ,σ で再正規化することで
    「1σ」の意味を統一し、ドメイン間の距離比較を有意にする。
    """
    jvnv_features = jvnv_df[feature_cols].to_numpy(dtype=np.float64)
    voicevox_features = voicevox_df[feature_cols].to_numpy(dtype=np.float64)

    pooled = np.vstack([jvnv_features, voicevox_features])
    pooled_means = pooled.mean(axis=0)
    pooled_stds = pooled.std(axis=0, ddof=0)

    valid_mask = pooled_stds > 0.0
    valid_features = [
        col for col, is_valid in zip(feature_cols, valid_mask, strict=True) if is_valid
    ]
    dropped_features = [
        col for col, is_valid in zip(feature_cols, valid_mask, strict=True) if not is_valid
    ]

    jvnv_aligned = jvnv_df.copy()
    voicevox_aligned = voicevox_df.copy()

    for df_aligned, features_raw in [
        (jvnv_aligned, jvnv_features),
        (voicevox_aligned, voicevox_features),
    ]:
        for i, col in enumerate(feature_cols):
            if valid_mask[i]:
                df_aligned[col] = (features_raw[:, i] - pooled_means[i]) / pooled_stds[i]

    jvnv_aligned = jvnv_aligned.drop(columns=dropped_features, errors="ignore")
    voicevox_aligned = voicevox_aligned.drop(columns=dropped_features, errors="ignore")

    params = {
        "valid_features": valid_features,
        "dropped_zero_variance_features": dropped_features,
        "num_jvnv_rows": len(jvnv_df),
        "num_voicevox_rows": len(voicevox_df),
        "pooled_means": {
            col: float(pooled_means[i]) for i, col in enumerate(feature_cols) if valid_mask[i]
        },
        "pooled_stds": {
            col: float(pooled_stds[i]) for i, col in enumerate(feature_cols) if valid_mask[i]
        },
    }

    return jvnv_aligned, voicevox_aligned, params


def run_alignment(
    *,
    config_path: str,
    jvnv_normalized: str | None,
    voicevox_normalized: str | None,
    output_dir: str | None,
) -> dict[str, Any]:
    """JVNV/VOICEVOX の正規化済み特徴量を pooled z-score で整合する。

    入力: per-speaker z-score 済み parquet（v01, v02）
    出力: aligned parquet（v03）と整合パラメータ JSON
    """
    config = load_experiment_config(config_path)
    v01_dir = resolve_path(config.v01.output_dir)
    v02_dir = resolve_path(config.v02.output_dir)
    v03_dir = resolve_path(output_dir) if output_dir else resolve_path(config.v03.output_dir)
    v03_dir.mkdir(parents=True, exist_ok=True)

    jvnv_path = (
        resolve_path(jvnv_normalized)
        if jvnv_normalized
        else v01_dir / "jvnv_egemaps_normalized.parquet"
    )
    voicevox_path = (
        resolve_path(voicevox_normalized)
        if voicevox_normalized
        else v02_dir / "voicevox_egemaps_normalized.parquet"
    )

    if not jvnv_path.exists():
        msg = f"JVNV normalized features not found: {jvnv_path}"
        raise FileNotFoundError(msg)
    if not voicevox_path.exists():
        msg = f"VOICEVOX normalized features not found: {voicevox_path}"
        raise FileNotFoundError(msg)

    jvnv_df = read_parquet(jvnv_path)
    voicevox_df = read_parquet(voicevox_path)

    jvnv_features = _collect_feature_columns(jvnv_df)
    voicevox_features = _collect_feature_columns(voicevox_df)
    common_features = sorted(set(jvnv_features).intersection(voicevox_features))

    if not common_features:
        msg = "No common eGeMAPS feature columns found between JVNV and VOICEVOX"
        raise ValueError(msg)

    logger.info(
        "クロスドメイン整合: JVNV=%d行, VOICEVOX=%d行, 共通特徴量=%d次元",
        len(jvnv_df),
        len(voicevox_df),
        len(common_features),
    )

    jvnv_aligned, voicevox_aligned, params = _cross_domain_normalize(
        jvnv_df,
        voicevox_df,
        common_features,
    )

    jvnv_out = v03_dir / "jvnv_egemaps_aligned.parquet"
    voicevox_out = v03_dir / "voicevox_egemaps_aligned.parquet"
    params_out = v03_dir / "cross_domain_alignment_params.json"

    write_parquet(jvnv_aligned, jvnv_out)
    write_parquet(voicevox_aligned, voicevox_out)

    params_payload = {
        "version": "1.0",
        "jvnv_input": str(jvnv_path),
        "voicevox_input": str(voicevox_path),
        "jvnv_output": str(jvnv_out),
        "voicevox_output": str(voicevox_out),
        "num_input_features": len(common_features),
        "num_valid_features": len(params["valid_features"]),
        **params,
    }
    save_json(params_payload, params_out)

    logger.info(
        "整合済み特徴量を保存: %d有効次元 (%d除外)",
        len(params["valid_features"]),
        len(params["dropped_zero_variance_features"]),
    )

    return {
        "jvnv_output": str(jvnv_out),
        "voicevox_output": str(voicevox_out),
        "params_output": str(params_out),
        "num_valid_features": len(params["valid_features"]),
    }


def main() -> None:
    _configure_logging()
    parser = _build_parser()
    args = parser.parse_args()
    summary = run_alignment(
        config_path=args.config,
        jvnv_normalized=args.jvnv_normalized,
        voicevox_normalized=args.voicevox_normalized,
        output_dir=args.output_dir,
    )
    logger.info("クロスドメイン整合完了: %s", summary["jvnv_output"])


if __name__ == "__main__":
    main()
