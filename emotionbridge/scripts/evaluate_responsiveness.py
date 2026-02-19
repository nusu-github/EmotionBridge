import argparse
import logging
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from emotionbridge.constants import COMMON6_CIRCUMPLEX_COORDS
from emotionbridge.scripts.common import (
    ensure_columns,
    load_experiment_config,
    read_parquet,
    resolve_path,
    save_json,
    save_markdown,
)
from emotionbridge.scripts.visualize import (
    plot_correlation_heatmap,
    plot_pca_scatter,
    plot_tsne_scatter,
)

logger = logging.getLogger(__name__)


def _configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="V-02: VOICEVOX 韻律応答性評価")
    parser.add_argument(
        "--config",
        default="configs/experiment_config.yaml",
        help="実験設定ファイル",
    )
    parser.add_argument(
        "--input-path",
        default=None,
        help="正規化済みVOICEVOX特徴量parquet",
    )
    parser.add_argument(
        "--gate-policy",
        choices=["feature_only", "feature_and_av"],
        default="feature_only",
        help="Go/No-Go 判定ポリシー",
    )
    return parser


def _feature_columns(df: pd.DataFrame) -> list[str]:
    return sorted([name for name in df.columns if name.startswith("egemaps__")])


def _control_columns(df: pd.DataFrame) -> list[str]:
    preferred = [
        "ctrl_pitch_shift",
        "ctrl_pitch_range",
        "ctrl_speed",
        "ctrl_energy",
        "ctrl_pause_weight",
    ]
    available = [name for name in preferred if name in df.columns]
    if available:
        return available
    return sorted([name for name in df.columns if name.startswith("ctrl_")])


def _canon_common6(label: str) -> str | None:
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


def _compute_corr_matrices(
    df: pd.DataFrame,
    control_cols: list[str],
    feature_cols: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    pearson_matrix = pd.DataFrame(
        index=control_cols,
        columns=feature_cols,
        dtype=np.float64,
    )
    spearman_matrix = pd.DataFrame(
        index=control_cols,
        columns=feature_cols,
        dtype=np.float64,
    )

    for control in control_cols:
        control_values = df[control].to_numpy(dtype=np.float64)
        for feature in feature_cols:
            feature_values = df[feature].to_numpy(dtype=np.float64)
            pearson_matrix.loc[control, feature] = pearsonr(
                control_values,
                feature_values,
            ).statistic
            spearman_matrix.loc[control, feature] = spearmanr(
                control_values,
                feature_values,
            ).statistic
    return pearson_matrix, spearman_matrix


def _residualize(target: np.ndarray, covariates: np.ndarray) -> np.ndarray:
    if covariates.size == 0:
        return target - target.mean()
    model = LinearRegression()
    model.fit(covariates, target)
    predicted = model.predict(covariates)
    return target - predicted


def _partial_correlations(
    df: pd.DataFrame,
    control_cols: list[str],
    feature_cols: list[str],
) -> pd.DataFrame:
    result = pd.DataFrame(index=control_cols, columns=feature_cols, dtype=np.float64)
    control_values = df[control_cols].to_numpy(dtype=np.float64)

    for control_index, control in enumerate(control_cols):
        target_control = control_values[:, control_index]
        other_indices = [index for index in range(len(control_cols)) if index != control_index]
        covariates = control_values[:, other_indices] if other_indices else np.empty((len(df), 0))
        control_residual = _residualize(target_control, covariates)

        for feature in feature_cols:
            feature_values = df[feature].to_numpy(dtype=np.float64)
            feature_residual = _residualize(feature_values, covariates)
            corr = pearsonr(control_residual, feature_residual).statistic
            result.loc[control, feature] = corr

    return result


def _pick_expected_features(feature_cols: list[str]) -> dict[str, list[str]]:
    keyword_rules = {
        "ctrl_pitch_shift": ["f0semitone", "f0"],
        "ctrl_pitch_range": ["f0semitone", "stddev", "range"],
        "ctrl_speed": [
            "voicedsegmentspersec",
            "meanvoicedsegmentlength",
            "meanunvoicedsegmentlength",
        ],
        "ctrl_energy": ["loudness", "equivalentsoundlevel"],
        "ctrl_pause_weight": ["unvoiced", "pause"],
    }
    picked: dict[str, list[str]] = {}
    for control, keywords in keyword_rules.items():
        matched: list[str] = []
        for column in feature_cols:
            lowered = column.lower()
            if any(keyword in lowered for keyword in keywords):
                matched.append(column)
        picked[control] = matched[:12]
    return picked


def _compute_feature_weights(
    partial_corr: pd.DataFrame,
    *,
    min_corr_threshold: float = 0.05,
) -> dict[str, float]:
    """各特徴量のmax|偏相関|を制御可能性スコアとして算出する。

    VOICEVOXの5D制御で動かせない特徴量次元にweight=0を付与し、
    重み付き距離計算で「動かせない方向に引きずられる」のを防ぐ。
    """
    abs_corr = partial_corr.abs()
    max_per_feature = abs_corr.max(axis=0)

    weights: dict[str, float] = {}
    for feature in max_per_feature.index:
        score = float(max_per_feature[feature])
        weights[str(feature)] = score if score >= min_corr_threshold else 0.0
    return weights


def _axis_responsiveness(
    partial_corr: pd.DataFrame,
    control_cols: list[str],
) -> dict[str, Any]:
    axis: dict[str, Any] = {}
    for control in control_cols:
        abs_values = partial_corr.loc[control].abs()
        top_feature = abs_values.idxmax()
        top_value = float(abs_values.max())
        axis[control] = {
            "top_feature": top_feature,
            "top_abs_partial_corr": top_value,
            "responsive": top_value >= 0.15,
        }
    return axis


def _compute_av_alignment(
    *,
    config,
    df: pd.DataFrame,
    control_cols: list[str],
) -> tuple[dict[str, Any] | None, pd.DataFrame | None]:
    jvnv_path = resolve_path(config.v01.output_dir) / "jvnv_egemaps_normalized.parquet"
    if not jvnv_path.exists():
        return None, None

    jvnv_df = read_parquet(jvnv_path).copy()
    if "emotion" not in jvnv_df.columns:
        return None, None

    jvnv_df["emotion_common6"] = jvnv_df["emotion"].map(_canon_common6)
    jvnv_df = jvnv_df[jvnv_df["emotion_common6"].notna()].reset_index(drop=True)
    if jvnv_df.empty:
        return None, None

    common_features = sorted(
        set(_feature_columns(jvnv_df)).intersection(_feature_columns(df)),
    )
    if len(common_features) < 5:
        return None, None

    x_jvnv = jvnv_df[common_features].to_numpy(dtype=np.float32, copy=False)
    y_arousal = np.array(
        [COMMON6_CIRCUMPLEX_COORDS[label][0] for label in jvnv_df["emotion_common6"]],
        dtype=np.float32,
    )
    y_valence = np.array(
        [COMMON6_CIRCUMPLEX_COORDS[label][1] for label in jvnv_df["emotion_common6"]],
        dtype=np.float32,
    )

    arousal_model = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("ridge", Ridge(alpha=1.0)),
        ],
    )
    valence_model = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("ridge", Ridge(alpha=1.0)),
        ],
    )
    arousal_model.fit(x_jvnv, y_arousal)
    valence_model.fit(x_jvnv, y_valence)

    train_arousal_r2 = float(r2_score(y_arousal, arousal_model.predict(x_jvnv)))
    train_valence_r2 = float(r2_score(y_valence, valence_model.predict(x_jvnv)))

    x_voicevox = df[common_features].to_numpy(dtype=np.float32, copy=False)
    av = np.column_stack(
        [
            arousal_model.predict(x_voicevox),
            valence_model.predict(x_voicevox),
        ],
    ).astype(np.float32)
    av = np.clip(av, -1.0, 1.0)

    corr_matrix = pd.DataFrame(
        index=control_cols,
        columns=["arousal", "valence"],
        dtype=np.float64,
    )
    by_control: dict[str, Any] = {}

    for control in control_cols:
        control_values = df[control].to_numpy(dtype=np.float64)
        a_pearson = float(pearsonr(control_values, av[:, 0]).statistic)
        a_spearman = float(spearmanr(control_values, av[:, 0]).statistic)
        v_pearson = float(pearsonr(control_values, av[:, 1]).statistic)
        v_spearman = float(spearmanr(control_values, av[:, 1]).statistic)

        corr_matrix.loc[control, "arousal"] = a_pearson
        corr_matrix.loc[control, "valence"] = v_pearson
        by_control[control] = {
            "arousal": {
                "pearson": a_pearson,
                "spearman": a_spearman,
            },
            "valence": {
                "pearson": v_pearson,
                "spearman": v_spearman,
            },
        }

    direction_rules: dict[str, tuple[str, str, float]] = {
        "ctrl_pitch_shift": ("arousal", "positive", 0.15),
        "ctrl_speed": ("arousal", "positive", 0.15),
        "ctrl_energy": ("arousal", "positive", 0.15),
        "ctrl_pause_weight": ("arousal", "negative", 0.10),
    }

    checks: dict[str, Any] = {}
    required_controls: list[str] = []
    for control, (axis, sign, threshold) in direction_rules.items():
        if control not in by_control:
            continue
        required_controls.append(control)
        score = float(by_control[control][axis]["pearson"])
        passes = score >= threshold if sign == "positive" else score <= -threshold
        checks[control] = {
            "axis": axis,
            "expected_sign": sign,
            "threshold": threshold,
            "score": score,
            "pass": bool(passes),
        }

    direction_gate_pass = bool(required_controls) and all(
        checks[name]["pass"] for name in required_controls
    )

    payload = {
        "jvnv_reference_path": str(jvnv_path),
        "common_features": common_features,
        "jvnv_fit": {
            "arousal_train_r2": train_arousal_r2,
            "valence_train_r2": train_valence_r2,
        },
        "control_axis_correlation": by_control,
        "direction_checks": checks,
        "required_direction_controls": required_controls,
        "direction_gate_pass": direction_gate_pass,
    }
    return payload, corr_matrix


def run_evaluation(
    config_path: str,
    input_path: str | None,
    gate_policy: str,
) -> dict[str, Any]:
    config = load_experiment_config(config_path)
    v02_dir = resolve_path(config.v02.output_dir)
    v02_dir.mkdir(parents=True, exist_ok=True)

    source_path = (
        resolve_path(input_path) if input_path else v02_dir / "voicevox_egemaps_normalized.parquet"
    )
    if not source_path.exists():
        msg = f"Input not found: {source_path}"
        raise FileNotFoundError(msg)

    df = read_parquet(source_path)
    control_cols = _control_columns(df)
    feature_cols = _feature_columns(df)
    ensure_columns(df, [*control_cols, *feature_cols], where="VOICEVOX normalized")

    pearson_matrix, spearman_matrix = _compute_corr_matrices(
        df,
        control_cols,
        feature_cols,
    )
    partial_corr = _partial_correlations(df, control_cols, feature_cols)
    expected_features = _pick_expected_features(feature_cols)
    axis_responsiveness = _axis_responsiveness(partial_corr, control_cols)

    required_axes = ["ctrl_pitch_shift", "ctrl_speed", "ctrl_energy"]
    responsive_required = [
        axis
        for axis in required_axes
        if axis in axis_responsiveness and axis_responsiveness[axis]["responsive"]
    ]
    feature_gate_pass = len(responsive_required) >= 3

    av_alignment, av_corr_matrix = _compute_av_alignment(
        config=config,
        df=df,
        control_cols=control_cols,
    )
    av_gate_pass = bool(av_alignment and av_alignment["direction_gate_pass"])
    gate_pass = (
        (feature_gate_pass and av_gate_pass if av_alignment is not None else feature_gate_pass)
        if gate_policy == "feature_and_av"
        else feature_gate_pass
    )

    top_features = (
        pearson_matrix.abs().max(axis=0).sort_values(ascending=False).head(25).index.tolist()
    )
    heatmap_matrix = pearson_matrix[top_features]
    plots_dir = v02_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    plot_correlation_heatmap(
        heatmap_matrix,
        output_path=plots_dir / "v02_control_feature_pearson_heatmap.png",
        title="V-02 Control vs eGeMAPS Pearson Correlation",
    )
    if av_corr_matrix is not None:
        plot_correlation_heatmap(
            av_corr_matrix,
            output_path=plots_dir / "v02_control_av_pearson_heatmap.png",
            title="V-02 Control vs Arousal/Valence Pearson Correlation",
        )

    marker_column = "style_id" if "style_id" in df.columns else None
    for control in ["ctrl_pitch_shift", "ctrl_speed", "ctrl_energy"]:
        if control not in control_cols:
            continue
        plot_pca_scatter(
            df,
            feature_cols,
            color_col=control,
            style_col=marker_column,
            output_path=plots_dir / f"v02_pca_{control}.png",
            title=f"V-02 PCA colored by {control}",
            random_seed=config.v02.random_seed,
        )
        plot_tsne_scatter(
            df,
            feature_cols,
            color_col=control,
            style_col=marker_column,
            output_path=plots_dir / f"v02_tsne_{control}.png",
            title=f"V-02 t-SNE colored by {control}",
            perplexity=config.v02.tsne_perplexities[1]
            if len(config.v02.tsne_perplexities) > 1
            else 30.0,
            random_seed=config.v02.random_seed,
        )

    feature_weights = _compute_feature_weights(
        partial_corr,
        min_corr_threshold=0.05,
    )
    num_nonzero = sum(1 for w in feature_weights.values() if w > 0)
    save_json(
        {
            "version": "1.0",
            "source": str(source_path),
            "min_corr_threshold": 0.05,
            "num_nonzero": num_nonzero,
            "num_zero": len(feature_weights) - num_nonzero,
            "weights": feature_weights,
        },
        v02_dir / "feature_weights.json",
    )

    results = {
        "input_path": str(source_path),
        "num_rows": len(df),
        "num_features": len(feature_cols),
        "control_columns": control_cols,
        "axis_responsiveness": axis_responsiveness,
        "expected_feature_candidates": expected_features,
        "required_axes": required_axes,
        "responsive_required_axes": responsive_required,
        "gate_policy": gate_policy,
        "feature_gate_pass": feature_gate_pass,
        "av_alignment": av_alignment,
        "av_gate_pass": av_gate_pass,
        "gate_pass": gate_pass,
        "feature_weights_path": str(v02_dir / "feature_weights.json"),
    }
    save_json(results, v02_dir / "v02_responsiveness_metrics.json")

    pearson_matrix.to_parquet(v02_dir / "v02_corr_pearson.parquet")
    spearman_matrix.to_parquet(v02_dir / "v02_corr_spearman.parquet")
    partial_corr.to_parquet(v02_dir / "v02_corr_partial.parquet")

    report_lines = [
        "# V-02 Responsiveness Report",
        "",
        "## Summary",
        f"- Input: {source_path}",
        f"- Samples: {len(df)}",
        f"- Features: {len(feature_cols)}",
        f"- Required responsive axes: {required_axes}",
        f"- Responsive required axes: {responsive_required}",
        f"- Gate policy: {gate_policy}",
        f"- Feature gate: {'Go' if feature_gate_pass else 'No-Go'}",
        f"- A/V direction gate (diagnostic): {'Go' if av_gate_pass else 'No-Go'}",
        f"- Gate result: {'Go' if gate_pass else 'No-Go'}",
        "",
        "## Axis Responsiveness",
        "| Control | Top Feature | Top |partial corr| | Responsive |",
        "|---|---|---:|---:|",
    ]
    for control in control_cols:
        payload = axis_responsiveness[control]
        report_lines.append(
            f"| {control} | {payload['top_feature']} | {payload['top_abs_partial_corr']:.4f} | {payload['responsive']} |",
        )

    if av_alignment is not None:
        report_lines.extend(
            [
                "",
                "## A/V Direction Checks",
                "| Control | Axis | Expected | Threshold | Score | Pass |",
                "|---|---|---|---:|---:|---:|",
            ],
        )
        for control in av_alignment["required_direction_controls"]:
            check = av_alignment["direction_checks"][control]
            report_lines.append(
                f"| {control} | {check['axis']} | {check['expected_sign']} | {check['threshold']:.2f} | {check['score']:.4f} | {check['pass']} |",
            )

    save_markdown("\n".join(report_lines), v02_dir / "v02_responsiveness_report.md")
    return results


def main() -> None:
    _configure_logging()
    parser = _build_parser()
    args = parser.parse_args()
    summary = run_evaluation(
        args.config,
        args.input_path,
        args.gate_policy,
    )
    logger.info(
        "V-02評価完了: responsive axes=%d",
        len(summary["responsive_required_axes"]),
    )


if __name__ == "__main__":
    main()
