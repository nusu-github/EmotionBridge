import argparse
import logging
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
from pathlib import Path


logger = logging.getLogger(__name__)


def _configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="eGeMAPS特徴のグループ内z-score正規化")
    parser.add_argument(
        "--config",
        default="configs/experiment_config.yaml",
        help="実験設定ファイル",
    )
    parser.add_argument(
        "--source",
        required=True,
        choices=["jvnv", "voicevox"],
        help="正規化対象",
    )
    parser.add_argument("--input-path", default=None, help="入力parquetを明示指定")
    parser.add_argument("--output-path", default=None, help="出力parquetを明示指定")
    parser.add_argument(
        "--params-path",
        default=None,
        help="正規化パラメータJSON出力先",
    )
    return parser


def _collect_feature_columns(df: pd.DataFrame) -> list[str]:
    return sorted([name for name in df.columns if name.startswith("egemaps__")])


def _balanced_subset_by_label(
    df: pd.DataFrame,
    *,
    label_col: str,
    random_seed: int,
) -> pd.DataFrame:
    counts = df[label_col].value_counts()
    min_count = int(counts.min())
    sampled_groups: list[pd.DataFrame] = []

    for index, (label, _) in enumerate(sorted(counts.to_dict().items())):
        subset = df[df[label_col] == label]
        sampled = subset.sample(
            n=min_count,
            random_state=random_seed + index,
            replace=False,
        )
        sampled_groups.append(sampled)
    return pd.concat(sampled_groups, ignore_index=True)


def _normalize(
    df: pd.DataFrame,
    *,
    group_col: str,
    feature_cols: list[str],
    equalize_stats: bool,
    label_col_for_equalize: str | None,
    random_seed: int,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    normalized = df.copy()

    group_params: dict[str, Any] = {}
    valid_feature_set = set(feature_cols)

    for group_name, group_df in df.groupby(group_col):
        stats_source = group_df
        if (
            equalize_stats
            and label_col_for_equalize is not None
            and label_col_for_equalize in group_df.columns
            and group_df[label_col_for_equalize].nunique() > 1
        ):
            stats_source = _balanced_subset_by_label(
                group_df,
                label_col=label_col_for_equalize,
                random_seed=random_seed,
            )

        means = stats_source[feature_cols].mean(axis=0)
        stds = stats_source[feature_cols].std(axis=0, ddof=0)

        group_valid = set(stds[stds > 0.0].index.tolist())
        valid_feature_set &= group_valid

        group_params[str(group_name)] = {
            "num_rows": len(group_df),
            "stats_rows": len(stats_source),
            "means": means.to_dict(),
            "stds": stds.to_dict(),
        }

    valid_features = sorted(valid_feature_set)
    dropped_features = sorted(set(feature_cols) - valid_feature_set)

    for group_name, group_df in df.groupby(group_col):
        index = group_df.index
        means = pd.Series(group_params[str(group_name)]["means"])
        stds = pd.Series(group_params[str(group_name)]["stds"])
        normalized.loc[index, valid_features] = (
            df.loc[index, valid_features] - means[valid_features]
        ) / stds[valid_features]

    normalized = normalized.drop(columns=dropped_features, errors="ignore")

    params = {
        "group_col": group_col,
        "valid_features": valid_features,
        "dropped_zero_variance_features": dropped_features,
        "group_params": {
            group_name: {
                "num_rows": payload["num_rows"],
                "stats_rows": payload["stats_rows"],
                "means": {name: payload["means"][name] for name in valid_features},
                "stds": {name: payload["stds"][name] for name in valid_features},
            }
            for group_name, payload in group_params.items()
        },
    }

    return normalized, params


def _default_paths(source: str, config_path: str) -> tuple[Path, Path, Path]:
    config = load_experiment_config(config_path)
    if source == "jvnv":
        base = resolve_path(config.v01.output_dir)
        return (
            base / "jvnv_egemaps_raw.parquet",
            base / "jvnv_egemaps_normalized.parquet",
            base / "jvnv_normalization_params.json",
        )

    base = resolve_path(config.v02.output_dir)
    return (
        base / "voicevox_egemaps_raw.parquet",
        base / "voicevox_egemaps_normalized.parquet",
        base / "voicevox_normalization_params.json",
    )


def run_normalize(
    *,
    config_path: str,
    source: str,
    input_path: str | None,
    output_path: str | None,
    params_path: str | None,
) -> dict[str, Any]:
    config = load_experiment_config(config_path)
    default_input, default_output, default_params = _default_paths(source, config_path)

    source_path = resolve_path(input_path) if input_path else default_input
    target_path = resolve_path(output_path) if output_path else default_output
    parameter_path = resolve_path(params_path) if params_path else default_params

    if not source_path.exists():
        msg = f"Input parquet not found: {source_path}"
        raise FileNotFoundError(msg)

    df = read_parquet(source_path)
    feature_cols = _collect_feature_columns(df)
    if not feature_cols:
        msg = f"No eGeMAPS columns found in {source_path}"
        raise ValueError(msg)

    if source == "jvnv":
        ensure_columns(df, ["speaker", "emotion"], where="JVNV raw features")
        group_col = "speaker"
        normalized_df, params = _normalize(
            df,
            group_col=group_col,
            feature_cols=feature_cols,
            equalize_stats=config.v01.equalize_speaker_stats,
            label_col_for_equalize="emotion",
            random_seed=config.v01.random_seed,
        )
    else:
        group_col = config.v02.speaker_mode
        if group_col not in df.columns:
            if "style_id" in df.columns:
                group_col = "style_id"
            else:
                df["_global_group"] = "all"
                group_col = "_global_group"

        normalized_df, params = _normalize(
            df,
            group_col=group_col,
            feature_cols=feature_cols,
            equalize_stats=False,
            label_col_for_equalize=None,
            random_seed=config.v02.random_seed,
        )

    write_parquet(normalized_df, target_path)

    params_payload = {
        "source": source,
        "input_path": str(source_path),
        "output_path": str(target_path),
        "num_rows": len(normalized_df),
        "num_original_features": len(feature_cols),
        "num_normalized_features": len(params["valid_features"]),
        **params,
    }
    save_json(params_payload, parameter_path)

    return {
        "source": source,
        "input_path": str(source_path),
        "output_path": str(target_path),
        "params_path": str(parameter_path),
        "num_rows": len(normalized_df),
        "num_features": len(params["valid_features"]),
    }


def main() -> None:
    _configure_logging()
    parser = _build_parser()
    args = parser.parse_args()

    summary = run_normalize(
        config_path=args.config,
        source=args.source,
        input_path=args.input_path,
        output_path=args.output_path,
        params_path=args.params_path,
    )
    logger.info("正規化完了: %s", summary["output_path"])


if __name__ == "__main__":
    main()
