import argparse
import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, r2_score, silhouette_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from emotionbridge.constants import COMMON6_CIRCUMPLEX_COORDS
from emotionbridge.eval import (
    build_evaluation_manifest,
    build_gate_decision,
    threshold_check,
    write_evaluation_manifest,
)
from emotionbridge.scripts.common import (
    load_experiment_config,
    read_parquet,
    resolve_path,
)
import pandas as pd


logger = logging.getLogger(__name__)


DEFAULT_ANCHORS: dict[str, dict[str, float]] = {
    emotion: {
        "arousal": coords[0],
        "valence": coords[1],
    }
    for emotion, coords in COMMON6_CIRCUMPLEX_COORDS.items()
}


def _configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="V-01 連続軸（Arousal/Valence）評価")
    parser.add_argument(
        "--config",
        default="configs/experiment_config.yaml",
        help="実験設定ファイル",
    )
    parser.add_argument(
        "--input-path",
        default=None,
        help="JVNV 正規化特徴量 parquet",
    )
    parser.add_argument(
        "--anchors-json",
        default=None,
        help="感情アンカー値を上書きする JSON ファイル",
    )
    parser.add_argument(
        "--arousal-r2-threshold",
        type=float,
        default=0.30,
        help="Conditional Go 判定の arousal R2 下限",
    )
    parser.add_argument(
        "--valence-r2-threshold",
        type=float,
        default=0.15,
        help="Conditional Go 判定の valence R2 下限",
    )
    return parser


def _canonicalize_emotion(raw_label: str) -> str | None:
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
    return mapping.get(str(raw_label).strip().lower())


def _load_anchors(path: str | None) -> dict[str, dict[str, float]]:
    if path is None:
        return DEFAULT_ANCHORS
    anchor_path = Path(path)
    with anchor_path.open("r", encoding="utf-8") as file:
        raw = json.load(file)
    return {
        str(emotion): {
            "arousal": float(values["arousal"]),
            "valence": float(values["valence"]),
        }
        for emotion, values in raw.items()
    }


def _feature_columns(df: pd.DataFrame) -> list[str]:
    return sorted([name for name in df.columns if name.startswith("egemaps__")])


def _cross_validated_regression(
    x: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
) -> dict[str, float]:
    splitter = GroupKFold(n_splits=4)
    y_true: list[np.ndarray] = []
    y_pred: list[np.ndarray] = []

    for train_index, test_index in splitter.split(x, y, groups=groups):
        model = Pipeline(
            [
                ("scaler", StandardScaler()),
                ("ridge", Ridge(alpha=1.0)),
            ],
        )
        model.fit(x[train_index], y[train_index])
        predicted = model.predict(x[test_index])
        y_true.append(y[test_index])
        y_pred.append(predicted)

    truth = np.concatenate(y_true)
    pred = np.concatenate(y_pred)
    return {
        "r2": float(r2_score(truth, pred)),
        "mae": float(mean_absolute_error(truth, pred)),
    }


def run_evaluation(
    *,
    config_path: str,
    input_path: str | None,
    anchors_json: str | None,
    arousal_r2_threshold: float,
    valence_r2_threshold: float,
) -> dict[str, Any]:
    config = load_experiment_config(config_path)
    v01_dir = resolve_path(config.v01.output_dir)
    v01_dir.mkdir(parents=True, exist_ok=True)

    source_path = (
        resolve_path(input_path)
        if input_path is not None
        else v01_dir / "jvnv_egemaps_normalized.parquet"
    )
    if not source_path.exists():
        msg = f"Input not found: {source_path}"
        raise FileNotFoundError(msg)

    anchors = _load_anchors(anchors_json)
    df = read_parquet(source_path).copy()
    df["emotion_common6"] = df["emotion"].map(_canonicalize_emotion)
    df = df[df["emotion_common6"].notna()].reset_index(drop=True)
    if df.empty:
        msg = "No valid common6 labels remained"
        raise ValueError(msg)

    feature_cols = _feature_columns(df)
    x = df[feature_cols].to_numpy(dtype=np.float32, copy=False)
    groups = df["speaker"].to_numpy()
    labels = df["emotion_common6"].to_numpy()

    df["arousal"] = df["emotion_common6"].map(lambda emo: anchors[emo]["arousal"])
    df["valence"] = df["emotion_common6"].map(lambda emo: anchors[emo]["valence"])

    categorical_silhouette = float(silhouette_score(x, labels))
    arousal_metrics = _cross_validated_regression(
        x,
        df["arousal"].to_numpy(dtype=np.float32),
        groups,
    )
    valence_metrics = _cross_validated_regression(
        x,
        df["valence"].to_numpy(dtype=np.float32),
        groups,
    )

    pca = PCA(n_components=2, random_state=42)
    z = pca.fit_transform(x)
    pca_df = df[["emotion_common6", "arousal", "valence"]].copy()
    pca_df["pc1"] = z[:, 0]
    pca_df["pc2"] = z[:, 1]
    centroids = (
        pca_df.groupby("emotion_common6")[["pc1", "pc2", "arousal", "valence"]]
        .mean()
        .sort_values("arousal")
    )

    conditional_go = bool(
        categorical_silhouette < config.evaluation.silhouette_go_threshold
        and arousal_metrics["r2"] >= arousal_r2_threshold
        and valence_metrics["r2"] >= valence_r2_threshold,
    )
    gate_decision = build_gate_decision(
        [
            threshold_check(
                "categorical_silhouette",
                categorical_silhouette,
                config.evaluation.silhouette_go_threshold,
                "<=",
            ),
            threshold_check("arousal_r2", arousal_metrics["r2"], arousal_r2_threshold, ">="),
            threshold_check("valence_r2", valence_metrics["r2"], valence_r2_threshold, ">="),
        ],
        overall_pass=conditional_go,
        label_pass="Conditional-Go",
        label_fail="No-Go",
    )

    result = {
        "input_path": str(source_path),
        "samples": len(df),
        "num_features": len(feature_cols),
        "categorical": {
            "silhouette": categorical_silhouette,
            "threshold": float(config.evaluation.silhouette_go_threshold),
            "go": bool(
                categorical_silhouette > config.evaluation.silhouette_go_threshold,
            ),
        },
        "continuous_axes": {
            "anchors": anchors,
            "arousal": arousal_metrics,
            "valence": valence_metrics,
            "r2_thresholds": {
                "arousal": arousal_r2_threshold,
                "valence": valence_r2_threshold,
            },
        },
        "pca_centroids": {
            emotion: {
                "pc1": float(values["pc1"]),
                "pc2": float(values["pc2"]),
                "arousal": float(values["arousal"]),
                "valence": float(values["valence"]),
            }
            for emotion, values in centroids.to_dict(orient="index").items()
        },
        "decision": {
            "label": gate_decision["label"],
            "continuous_axes_go": conditional_go,
            "gate_decision": gate_decision,
            "message": (
                "6感情クラスタリングはNo-Goだが連続軸回帰が閾値を満たすため、連続軸ベース設計として進行可能。"
                if conditional_go
                else "連続軸回帰も閾値を満たさないため、再設計が必要。"
            ),
        },
    }

    output_json = v01_dir / "v01_continuous_axes_metrics.json"
    output_md = v01_dir / "v01_continuous_axes_report.md"

    lines = [
        "# V-01 Continuous Axes Report",
        "",
        "## Decision",
        f"- Label: {result['decision']['label']}",
        f"- Message: {result['decision']['message']}",
        "",
        "## Categorical (Reference)",
        f"- Silhouette: {categorical_silhouette:.4f}",
        f"- Threshold: {config.evaluation.silhouette_go_threshold:.3f}",
        f"- Result: {'Go' if result['categorical']['go'] else 'No-Go'}",
        "",
        "## Continuous Axes",
        f"- Arousal R2: {arousal_metrics['r2']:.4f}",
        f"- Arousal MAE: {arousal_metrics['mae']:.4f}",
        f"- Valence R2: {valence_metrics['r2']:.4f}",
        f"- Valence MAE: {valence_metrics['mae']:.4f}",
        f"- Thresholds: arousal>={arousal_r2_threshold:.2f}, valence>={valence_r2_threshold:.2f}",
        "",
        "## PCA Centroids (sorted by arousal)",
        "| Emotion | PC1 | PC2 | Arousal | Valence |",
        "|---|---:|---:|---:|---:|",
    ]

    for emotion, payload in result["pca_centroids"].items():
        lines.append(
            f"| {emotion} | {payload['pc1']:.4f} | {payload['pc2']:.4f} | {payload['arousal']:.2f} | {payload['valence']:.2f} |",
        )

    with output_md.open("w", encoding="utf-8") as file:
        file.write("\n".join(lines))

    manifest = build_evaluation_manifest(
        task="v01_continuous_axes",
        gate=gate_decision,
        summary={
            "samples": len(df),
            "num_features": len(feature_cols),
            "categorical_silhouette": categorical_silhouette,
            "arousal_r2": arousal_metrics["r2"],
            "valence_r2": valence_metrics["r2"],
            "arousal_mae": arousal_metrics["mae"],
            "valence_mae": valence_metrics["mae"],
        },
        inputs={
            "input_path": str(source_path),
            "anchors_json": anchors_json,
        },
        artifacts={
            "metrics_json": str(output_json),
            "report_markdown": str(output_md),
        },
        metadata={
            "arousal_r2_threshold": arousal_r2_threshold,
            "valence_r2_threshold": valence_r2_threshold,
        },
    )
    manifest_path = write_evaluation_manifest(
        manifest,
        v01_dir / "v01_continuous_axes_manifest.json",
    )
    result["evaluation_manifest_path"] = str(manifest_path)

    with output_json.open("w", encoding="utf-8") as file:
        json.dump(result, file, ensure_ascii=False, indent=2)

    return {
        "json": str(output_json),
        "report": str(output_md),
        "manifest": str(manifest_path),
        "decision": result["decision"],
        "arousal_r2": arousal_metrics["r2"],
        "valence_r2": valence_metrics["r2"],
    }


def main() -> None:
    _configure_logging()
    args = _build_parser().parse_args()
    summary = run_evaluation(
        config_path=args.config,
        input_path=args.input_path,
        anchors_json=args.anchors_json,
        arousal_r2_threshold=args.arousal_r2_threshold,
        valence_r2_threshold=args.valence_r2_threshold,
    )
    logger.info(
        "連続軸評価完了: decision=%s, arousal_r2=%.4f, valence_r2=%.4f",
        summary["decision"]["label"],
        summary["arousal_r2"],
        summary["valence_r2"],
    )


if __name__ == "__main__":
    main()
