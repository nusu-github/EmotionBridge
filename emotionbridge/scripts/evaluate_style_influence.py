import argparse
import logging
from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

from emotionbridge.eval import (
    boolean_check,
    build_evaluation_manifest,
    build_gate_decision,
    threshold_check,
    write_evaluation_manifest,
)
from emotionbridge.scripts.common import (
    ensure_columns,
    load_experiment_config,
    read_parquet,
    resolve_path,
    save_json,
    save_markdown,
    write_parquet,
)

logger = logging.getLogger(__name__)

CONTROL_COLUMNS = [
    "ctrl_pitch_shift",
    "ctrl_pitch_range",
    "ctrl_speed",
    "ctrl_energy",
    "ctrl_pause_weight",
]


def _configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="V-03: style_id主効果/交互作用の影響評価",
    )
    parser.add_argument(
        "--config",
        default="configs/experiment_config.yaml",
        help="実験設定ファイル",
    )
    parser.add_argument(
        "--voicevox-raw",
        default=None,
        help="VOICEVOX生特徴量parquet（未指定時はv02出力）",
    )
    parser.add_argument(
        "--jvnv-normalized",
        default=None,
        help="JVNV正規化特徴量parquet（未指定時はv01出力）",
    )
    parser.add_argument(
        "--target-style-ids",
        default=None,
        help="対象style_idをカンマ区切りで指定（未指定時は入力内の全style）",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="出力先ディレクトリ（未指定時はv03出力）",
    )
    return parser


def _feature_cols(df: pd.DataFrame) -> list[str]:
    return sorted([name for name in df.columns if name.startswith("egemaps__")])


def _parse_target_style_ids(raw: str | None) -> list[int] | None:
    if raw is None or not raw.strip():
        return None
    values = [chunk.strip() for chunk in raw.split(",")]
    parsed = [int(value) for value in values if value]
    if not parsed:
        return None
    return list(dict.fromkeys(parsed))


def _eta2_per_feature(matrix: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """One-way ANOVAのeta^2を特徴量ごとに算出する。"""
    n_features = matrix.shape[1]
    if matrix.shape[0] == 0 or n_features == 0:
        return np.array([], dtype=np.float64)

    grand_mean = matrix.mean(axis=0, keepdims=True)
    ss_total = np.square(matrix - grand_mean).sum(axis=0)

    ss_between = np.zeros(n_features, dtype=np.float64)
    for label in np.unique(labels):
        idx = labels == label
        if not np.any(idx):
            continue
        group = matrix[idx]
        diff = group.mean(axis=0) - grand_mean.ravel()
        ss_between += group.shape[0] * np.square(diff)

    return np.divide(
        ss_between,
        ss_total,
        out=np.full(n_features, np.nan, dtype=np.float64),
        where=ss_total > 0.0,
    )


def _r2_per_feature(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    y_mean = y_true.mean(axis=0, keepdims=True)
    ss_total = np.square(y_true - y_mean).sum(axis=0)
    ss_error = np.square(y_true - y_pred).sum(axis=0)
    return np.divide(
        ss_total - ss_error,
        ss_total,
        out=np.full(y_true.shape[1], np.nan, dtype=np.float64),
        where=ss_total > 0.0,
    )


def _safe_summary(values: np.ndarray) -> dict[str, float]:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return {"mean": float("nan"), "median": float("nan"), "max": float("nan")}
    return {
        "mean": float(np.mean(finite)),
        "median": float(np.median(finite)),
        "max": float(np.max(finite)),
    }


def run_evaluation(
    *,
    config_path: str,
    voicevox_raw: str | None,
    jvnv_normalized: str | None,
    target_style_ids_raw: str | None,
    output_dir: str | None,
) -> dict[str, Any]:
    config = load_experiment_config(config_path)
    v01_dir = resolve_path(config.v01.output_dir)
    v02_dir = resolve_path(config.v02.output_dir)
    v03_dir = resolve_path(output_dir) if output_dir else resolve_path(config.v03.output_dir)
    v03_dir.mkdir(parents=True, exist_ok=True)

    raw_path = (
        resolve_path(voicevox_raw)
        if voicevox_raw
        else v02_dir / "voicevox_egemaps_raw.parquet"
    )
    jvnv_path = (
        resolve_path(jvnv_normalized)
        if jvnv_normalized
        else v01_dir / "jvnv_egemaps_normalized.parquet"
    )
    if not raw_path.exists() or not jvnv_path.exists():
        msg = f"Input file missing: raw={raw_path.exists()}, jvnv={jvnv_path.exists()}"
        raise FileNotFoundError(msg)

    raw_df = read_parquet(raw_path).copy()
    jvnv_df = read_parquet(jvnv_path).copy()

    ensure_columns(raw_df, ["style_id", *CONTROL_COLUMNS], where="voicevox raw")

    target_style_ids = _parse_target_style_ids(target_style_ids_raw)
    if target_style_ids is None:
        target_style_ids = sorted(int(style_id) for style_id in raw_df["style_id"].unique().tolist())
    raw_df = raw_df[raw_df["style_id"].isin(target_style_ids)].reset_index(drop=True)
    if raw_df.empty:
        msg = f"No rows remain after style_id filter: {target_style_ids}"
        raise RuntimeError(msg)

    feature_cols = sorted(set(_feature_cols(raw_df)).intersection(_feature_cols(jvnv_df)))
    if not feature_cols:
        msg = "No overlapping eGeMAPS feature columns between voicevox raw and jvnv normalized"
        raise ValueError(msg)

    raw_feature_mat = raw_df[feature_cols].to_numpy(dtype=np.float64)
    means = raw_feature_mat.mean(axis=0)
    stds = raw_feature_mat.std(axis=0, ddof=0)
    valid_idx = stds > 0.0
    valid_feature_cols = [feature for feature, ok in zip(feature_cols, valid_idx, strict=True) if ok]
    if not valid_feature_cols:
        msg = "No valid feature columns after global z-score standardization"
        raise RuntimeError(msg)

    valid_raw_mat = raw_df[valid_feature_cols].to_numpy(dtype=np.float64)
    valid_means = means[valid_idx]
    valid_stds = stds[valid_idx]
    global_z_mat = (valid_raw_mat - valid_means) / valid_stds

    labels = raw_df["style_id"].to_numpy()
    num_styles = len(np.unique(labels))

    eta2_raw = _eta2_per_feature(valid_raw_mat, labels)
    eta2_global_z = _eta2_per_feature(global_z_mat, labels)

    control_mat = raw_df[CONTROL_COLUMNS].to_numpy(dtype=np.float64)
    style_dummies = pd.get_dummies(pd.Categorical(labels), drop_first=True, dtype=np.float64).to_numpy()

    r2_m0 = np.full(len(valid_feature_cols), np.nan, dtype=np.float64)
    r2_m1 = np.full(len(valid_feature_cols), np.nan, dtype=np.float64)
    r2_m2 = np.full(len(valid_feature_cols), np.nan, dtype=np.float64)
    delta_style_main = np.full(len(valid_feature_cols), np.nan, dtype=np.float64)
    delta_interaction = np.full(len(valid_feature_cols), np.nan, dtype=np.float64)
    partial_eta2_style_main = np.full(len(valid_feature_cols), np.nan, dtype=np.float64)
    partial_eta2_interaction = np.full(len(valid_feature_cols), np.nan, dtype=np.float64)

    style_signal_status = "unavailable"
    unavailable_reason: str | None = None

    if num_styles >= 2 and style_dummies.shape[1] >= 1:
        interaction_terms = np.concatenate(
            [control_mat[:, [idx]] * style_dummies for idx in range(control_mat.shape[1])],
            axis=1,
        )
        x_m0 = control_mat
        x_m1 = np.concatenate([control_mat, style_dummies], axis=1)
        x_m2 = np.concatenate([control_mat, style_dummies, interaction_terms], axis=1)

        model_m0 = LinearRegression().fit(x_m0, global_z_mat)
        model_m1 = LinearRegression().fit(x_m1, global_z_mat)
        model_m2 = LinearRegression().fit(x_m2, global_z_mat)

        pred_m0 = model_m0.predict(x_m0)
        pred_m1 = model_m1.predict(x_m1)
        pred_m2 = model_m2.predict(x_m2)

        r2_m0 = _r2_per_feature(global_z_mat, pred_m0)
        r2_m1 = _r2_per_feature(global_z_mat, pred_m1)
        r2_m2 = _r2_per_feature(global_z_mat, pred_m2)

        delta_style_main = np.maximum(0.0, r2_m1 - r2_m0)
        delta_interaction = np.maximum(0.0, r2_m2 - r2_m1)

        residual_m1 = np.maximum(1e-12, 1.0 - r2_m1)
        residual_m2 = np.maximum(1e-12, 1.0 - r2_m2)
        partial_eta2_style_main = delta_style_main / (delta_style_main + residual_m1)
        partial_eta2_interaction = delta_interaction / (delta_interaction + residual_m2)

        style_main_ratio = float(
            np.mean(partial_eta2_style_main >= config.v03.style_effect_min_partial_eta2),
        )
        interaction_ratio = float(
            np.mean(
                partial_eta2_interaction >= config.v03.style_interaction_min_partial_eta2,
            ),
        )
        if (
            style_main_ratio >= config.v03.style_effect_min_feature_ratio
            or interaction_ratio >= config.v03.style_interaction_min_feature_ratio
        ):
            style_signal_status = "retain_style"
        else:
            style_signal_status = "deprioritize_style"
    else:
        unavailable_reason = (
            "style effect metrics require at least two styles with non-empty style dummies"
        )
        style_main_ratio = float("nan")
        interaction_ratio = float("nan")

    feature_table = pd.DataFrame(
        {
            "feature": valid_feature_cols,
            "eta2_style_raw": eta2_raw,
            "eta2_style_global_z": eta2_global_z,
            "r2_controls": r2_m0,
            "r2_controls_plus_style": r2_m1,
            "r2_controls_plus_style_interaction": r2_m2,
            "delta_r2_style_main": delta_style_main,
            "delta_r2_interaction": delta_interaction,
            "partial_eta2_style_main": partial_eta2_style_main,
            "partial_eta2_interaction": partial_eta2_interaction,
            "style_main_effective": partial_eta2_style_main
            >= config.v03.style_effect_min_partial_eta2,
            "interaction_effective": partial_eta2_interaction
            >= config.v03.style_interaction_min_partial_eta2,
        },
    )
    feature_table_path = v03_dir / "style_influence_feature_table.parquet"
    write_parquet(feature_table, feature_table_path)

    top_style_main = (
        feature_table[["feature", "partial_eta2_style_main"]]
        .sort_values("partial_eta2_style_main", ascending=False)
        .head(10)
        .to_dict(orient="records")
    )
    top_interaction = (
        feature_table[["feature", "partial_eta2_interaction"]]
        .sort_values("partial_eta2_interaction", ascending=False)
        .head(10)
        .to_dict(orient="records")
    )

    if style_signal_status == "unavailable":
        gate_checks = [
            boolean_check(
                "style_influence_available",
                False,
                detail=unavailable_reason,
            ),
        ]
        gate_pass = False
    else:
        gate_checks = [
            threshold_check(
                "style_main_feature_ratio",
                style_main_ratio,
                float(config.v03.style_effect_min_feature_ratio),
                ">=",
            ),
            threshold_check(
                "interaction_feature_ratio",
                interaction_ratio,
                float(config.v03.style_interaction_min_feature_ratio),
                ">=",
            ),
        ]
        gate_pass = style_signal_status == "retain_style"
    gate_decision = build_gate_decision(
        gate_checks,
        overall_pass=gate_pass,
        label_pass="Retain-Style",
        label_fail="Deprioritize-Style",
    )

    metrics_path = v03_dir / "style_influence_metrics.json"
    report_path = v03_dir / "style_influence_report.md"

    results = {
        "voicevox_raw_path": str(raw_path),
        "jvnv_normalized_path": str(jvnv_path),
        "num_rows": len(raw_df),
        "styles_used": target_style_ids,
        "num_styles": num_styles,
        "feature_count": len(valid_feature_cols),
        "dropped_zero_std_features": int(len(feature_cols) - len(valid_feature_cols)),
        "style_effect_thresholds": {
            "style_main_min_partial_eta2": float(config.v03.style_effect_min_partial_eta2),
            "style_main_min_feature_ratio": float(config.v03.style_effect_min_feature_ratio),
            "interaction_min_partial_eta2": float(config.v03.style_interaction_min_partial_eta2),
            "interaction_min_feature_ratio": float(
                config.v03.style_interaction_min_feature_ratio,
            ),
        },
        "summary": {
            "eta2_style_raw": _safe_summary(eta2_raw),
            "eta2_style_global_z": _safe_summary(eta2_global_z),
            "delta_r2_style_main": _safe_summary(delta_style_main),
            "delta_r2_interaction": _safe_summary(delta_interaction),
            "partial_eta2_style_main": _safe_summary(partial_eta2_style_main),
            "partial_eta2_interaction": _safe_summary(partial_eta2_interaction),
        },
        "style_main_feature_ratio": style_main_ratio,
        "interaction_feature_ratio": interaction_ratio,
        "style_signal_status": style_signal_status,
        "unavailable_reason": unavailable_reason,
        "top_style_main_features": top_style_main,
        "top_interaction_features": top_interaction,
        "decision": gate_decision["label"],
        "gate_decision": gate_decision,
        "feature_table_path": str(feature_table_path),
    }

    report_lines = [
        "# Style Influence Report",
        "",
        "## Summary",
        f"- VOICEVOX raw: {raw_path}",
        f"- JVNV normalized: {jvnv_path}",
        f"- Rows: {len(raw_df)}",
        f"- Styles: {target_style_ids}",
        f"- Features: {len(valid_feature_cols)}",
        f"- Dropped zero-std features: {len(feature_cols) - len(valid_feature_cols)}",
        f"- Style signal status: {style_signal_status}",
        f"- Decision: {gate_decision['label']}",
        "",
        "## Thresholds",
        f"- style_main partial eta2 threshold: {config.v03.style_effect_min_partial_eta2:.4f}",
        f"- style_main feature ratio threshold: {config.v03.style_effect_min_feature_ratio:.4f}",
        f"- interaction partial eta2 threshold: {config.v03.style_interaction_min_partial_eta2:.4f}",
        f"- interaction feature ratio threshold: {config.v03.style_interaction_min_feature_ratio:.4f}",
        "",
        "## Ratios",
        f"- style_main_feature_ratio: {style_main_ratio}",
        f"- interaction_feature_ratio: {interaction_ratio}",
        "",
        "## Top Style Main Features (partial eta2)",
        "| feature | partial_eta2_style_main |",
        "|---|---:|",
    ]
    report_lines.extend(
        [
            f"| {row['feature']} | {float(row['partial_eta2_style_main']):.6f} |"
            for row in top_style_main
        ],
    )

    report_lines.extend(
        [
            "",
            "## Top Interaction Features (partial eta2)",
            "| feature | partial_eta2_interaction |",
            "|---|---:|",
        ],
    )
    report_lines.extend(
        [
            f"| {row['feature']} | {float(row['partial_eta2_interaction']):.6f} |"
            for row in top_interaction
        ],
    )

    save_markdown("\n".join(report_lines), report_path)

    manifest = build_evaluation_manifest(
        task="v03_style_influence",
        gate=gate_decision,
        summary={
            "num_rows": len(raw_df),
            "num_styles": num_styles,
            "feature_count": len(valid_feature_cols),
            "style_signal_status": style_signal_status,
            "style_main_feature_ratio": style_main_ratio,
            "interaction_feature_ratio": interaction_ratio,
        },
        inputs={
            "voicevox_raw_path": str(raw_path),
            "jvnv_normalized_path": str(jvnv_path),
            "styles_used": [int(style_id) for style_id in target_style_ids],
        },
        artifacts={
            "metrics_json": str(metrics_path),
            "feature_table_parquet": str(feature_table_path),
            "report_markdown": str(report_path),
        },
        metadata={
            "control_columns": CONTROL_COLUMNS,
            "dropped_zero_std_features": int(len(feature_cols) - len(valid_feature_cols)),
        },
    )
    manifest_path = write_evaluation_manifest(manifest, v03_dir / "style_influence_manifest.json")
    results["evaluation_manifest_path"] = str(manifest_path)

    save_json(results, metrics_path)
    return results


def main() -> None:
    _configure_logging()
    parser = _build_parser()
    args = parser.parse_args()
    summary = run_evaluation(
        config_path=args.config,
        voicevox_raw=args.voicevox_raw,
        jvnv_normalized=args.jvnv_normalized,
        target_style_ids_raw=args.target_style_ids,
        output_dir=args.output_dir,
    )
    logger.info("Style influence evaluation completed: %s", summary["style_signal_status"])


if __name__ == "__main__":
    main()
