import argparse
import logging
import operator
from itertools import combinations
from typing import TYPE_CHECKING, Any

import numpy as np
from scipy.spatial.distance import pdist, squareform
from scipy.stats import ttest_rel
from sklearn.metrics import calinski_harabasz_score, silhouette_score

from emotionbridge.scripts.common import (
    ensure_columns,
    load_experiment_config,
    read_parquet,
    resolve_path,
    save_json,
    save_markdown,
)
from emotionbridge.scripts.visualize import (
    plot_pca_3d_scatter,
    plot_pca_scatter,
    plot_tier_boxplots,
    plot_tsne_scatter,
)
import pandas as pd

if TYPE_CHECKING:
    from pathlib import Path


logger = logging.getLogger(__name__)


def _configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="V-01: JVNV 感情分離性評価")
    parser.add_argument(
        "--config",
        default="configs/experiment_config.yaml",
        help="実験設定ファイル",
    )
    parser.add_argument(
        "--input-path",
        default=None,
        help="正規化済みJVNV特徴量parquet",
    )
    parser.add_argument(
        "--with-nv-input-path",
        default=None,
        help="NV除外なし版（比較用、任意）",
    )
    parser.add_argument(
        "--permutations",
        type=int,
        default=499,
        help="PERMANOVA permutation回数",
    )
    return parser


def _canonicalize_jvnv_emotion(raw_label: str) -> str | None:
    label = raw_label.strip().lower()
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
    return mapping.get(label)


def _feature_columns(df: pd.DataFrame) -> list[str]:
    return sorted([name for name in df.columns if name.startswith("egemaps__")])


def _permanova_statistic(features: np.ndarray, labels: np.ndarray) -> float:
    unique_labels = np.unique(labels)
    n_samples = len(labels)
    n_groups = len(unique_labels)
    if n_groups < 2 or n_samples <= n_groups:
        return float("nan")

    dist_sq = squareform(pdist(features, metric="euclidean")) ** 2
    total_ss = dist_sq.sum() / (2 * n_samples)

    within_ss = 0.0
    for label in unique_labels:
        index = np.where(labels == label)[0]
        n_group = len(index)
        if n_group <= 1:
            continue
        within_ss += dist_sq[np.ix_(index, index)].sum() / (2 * n_group)

    between_ss = total_ss - within_ss
    numerator = between_ss / (n_groups - 1)
    denominator = within_ss / (n_samples - n_groups)
    if denominator <= 0:
        return float("nan")
    return float(numerator / denominator)


def _permanova_test(
    features: np.ndarray,
    labels: np.ndarray,
    *,
    permutations: int,
    random_seed: int,
) -> dict[str, float]:
    observed_f = _permanova_statistic(features, labels)
    if np.isnan(observed_f):
        return {"pseudo_f": float("nan"), "p_value": float("nan")}

    rng = np.random.default_rng(random_seed)
    exceed_count = 0
    for _ in range(permutations):
        shuffled = rng.permutation(labels)
        permuted_f = _permanova_statistic(features, shuffled)
        if not np.isnan(permuted_f) and permuted_f >= observed_f:
            exceed_count += 1

    p_value = (exceed_count + 1) / (permutations + 1)
    return {"pseudo_f": observed_f, "p_value": float(p_value)}


def _pairwise_scores(
    features: np.ndarray,
    labels: np.ndarray,
    *,
    permutations: int,
    random_seed: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    unique_labels = sorted(np.unique(labels).tolist())

    for left, right in combinations(unique_labels, 2):
        mask = np.logical_or(labels == left, labels == right)
        subset_features = features[mask]
        subset_labels = labels[mask]
        silhouette = float(silhouette_score(subset_features, subset_labels))
        permanova = _permanova_test(
            subset_features,
            subset_labels,
            permutations=permutations,
            random_seed=random_seed,
        )
        rows.append(
            {
                "emotion_left": left,
                "emotion_right": right,
                "silhouette": silhouette,
                "permanova_pseudo_f": permanova["pseudo_f"],
                "permanova_p_value": permanova["p_value"],
            },
        )

    return rows


def _pick_tier1_features(feature_cols: list[str]) -> list[str]:
    keywords = [
        "f0",
        "loudness",
        "voicedsegmentspersec",
        "meanunvoicedsegmentlength",
        "meanvoicedsegmentlength",
        "equivalentSoundLevel",
    ]
    selected: list[str] = []
    for column in feature_cols:
        lowered = column.lower()
        if any(keyword in lowered for keyword in keywords):
            selected.append(column)
    return selected[:8]


def _evaluate_nv_impact(
    without_nv_df: pd.DataFrame,
    with_nv_df: pd.DataFrame,
    feature_cols: list[str],
) -> dict[str, Any]:
    joined = without_nv_df[["utterance_id", *feature_cols]].merge(
        with_nv_df[["utterance_id", *feature_cols]],
        on="utterance_id",
        suffixes=("_without", "_with"),
        how="inner",
    )
    if joined.empty:
        return {"paired_rows": 0}

    deltas: dict[str, float] = {}
    pvalues: dict[str, float] = {}
    for feature in feature_cols:
        without_values = joined[f"{feature}_without"].to_numpy(dtype=np.float32)
        with_values = joined[f"{feature}_with"].to_numpy(dtype=np.float32)
        deltas[feature] = float(np.mean(np.abs(without_values - with_values)))
        _, pvalue = ttest_rel(without_values, with_values, nan_policy="omit")
        pvalues[feature] = float(pvalue) if np.isfinite(pvalue) else float("nan")

    top_changed = sorted(deltas.items(), key=operator.itemgetter(1), reverse=True)[:10]
    return {
        "paired_rows": len(joined),
        "mean_abs_shift_per_feature": deltas,
        "paired_ttest_p_values": pvalues,
        "top_changed_features": top_changed,
    }


def run_evaluation(
    *,
    config_path: str,
    input_path: str | None,
    with_nv_input_path: str | None,
    permutations: int,
) -> dict[str, Any]:
    config = load_experiment_config(config_path)
    v01_dir = resolve_path(config.v01.output_dir)
    output_dir = v01_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    source_path = (
        resolve_path(input_path) if input_path else v01_dir / "jvnv_egemaps_normalized.parquet"
    )
    if not source_path.exists():
        msg = f"Input not found: {source_path}"
        raise FileNotFoundError(msg)

    df = read_parquet(source_path)
    ensure_columns(df, ["utterance_id", "speaker", "emotion"], where="JVNV normalized")
    feature_cols = _feature_columns(df)
    if not feature_cols:
        msg = "No normalized eGeMAPS feature columns found"
        raise ValueError(msg)

    df = df.copy()
    df["emotion_common6"] = df["emotion"].map(_canonicalize_jvnv_emotion)
    df = df[df["emotion_common6"].notna()].reset_index(drop=True)
    if df.empty:
        msg = "No valid JVNV emotion labels remained after canonicalization"
        raise ValueError(msg)

    features = df[feature_cols].to_numpy(dtype=np.float32, copy=False)
    labels = df["emotion_common6"].to_numpy()

    silhouette = float(silhouette_score(features, labels))
    calinski_harabasz = float(calinski_harabasz_score(features, labels))
    overall_permanova = _permanova_test(
        features,
        labels,
        permutations=permutations,
        random_seed=config.v01.random_seed,
    )
    pairwise = _pairwise_scores(
        features,
        labels,
        permutations=permutations,
        random_seed=config.v01.random_seed,
    )

    plots_dir = output_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    for perplexity in config.v01.tsne_perplexities:
        plot_tsne_scatter(
            df,
            feature_cols,
            color_col="emotion_common6",
            style_col="speaker",
            output_path=plots_dir / f"v01_tsne_perp_{int(perplexity)}.png",
            title=f"V-01 t-SNE (perplexity={perplexity})",
            perplexity=perplexity,
            random_seed=config.v01.random_seed,
        )

    plot_pca_scatter(
        df,
        feature_cols,
        color_col="emotion_common6",
        style_col="speaker",
        output_path=plots_dir / "v01_pca_2d.png",
        title="V-01 PCA 2D",
        random_seed=config.v01.random_seed,
    )
    plot_pca_3d_scatter(
        df,
        feature_cols,
        color_col="emotion_common6",
        output_path=plots_dir / "v01_pca_3d.png",
        title="V-01 PCA 3D",
        random_seed=config.v01.random_seed,
    )

    tier_features = _pick_tier1_features(feature_cols)
    plot_tier_boxplots(
        df,
        tier_features,
        group_col="emotion_common6",
        output_path=plots_dir / "v01_tier1_boxplot.png",
        title="V-01 Tier-1 Feature Distribution",
    )

    nv_impact: dict[str, Any] | None = None
    with_nv_path: Path | None = None
    if with_nv_input_path:
        with_nv_path = resolve_path(with_nv_input_path)
    else:
        candidate = v01_dir / "jvnv_egemaps_with_nv_normalized.parquet"
        if candidate.exists():
            with_nv_path = candidate

    if with_nv_path is not None and with_nv_path.exists():
        with_nv_df = read_parquet(with_nv_path)
        common_features = sorted(
            set(feature_cols).intersection(_feature_columns(with_nv_df)),
        )
        if common_features and "utterance_id" in with_nv_df.columns:
            nv_impact = _evaluate_nv_impact(df, with_nv_df, common_features)

    go_threshold = config.evaluation.silhouette_go_threshold
    go_result = silhouette > go_threshold

    results = {
        "input_path": str(source_path),
        "num_rows": len(df),
        "num_features": len(feature_cols),
        "silhouette_score": silhouette,
        "calinski_harabasz_index": calinski_harabasz,
        "overall_permanova": overall_permanova,
        "pairwise": pairwise,
        "go_threshold": go_threshold,
        "go_result": go_result,
        "nv_impact": nv_impact,
    }

    save_json(results, output_dir / "v01_metrics.json")

    report_lines = [
        "# V-01 Separation Report",
        "",
        "## Summary",
        f"- Input: {source_path}",
        f"- Samples: {len(df)}",
        f"- Features: {len(feature_cols)}",
        f"- Silhouette score: {silhouette:.4f}",
        f"- Calinski-Harabasz Index: {calinski_harabasz:.4f}",
        f"- PERMANOVA pseudo-F: {overall_permanova['pseudo_f']:.4f}",
        f"- PERMANOVA p-value: {overall_permanova['p_value']:.6f}",
        f"- Gate threshold (silhouette): {go_threshold:.3f}",
        f"- Gate result: {'Go' if go_result else 'No-Go'}",
        "",
        "## Pairwise Metrics",
        "| Emotion A | Emotion B | Silhouette | PERMANOVA pseudo-F | p-value |",
        "|---|---:|---:|---:|---:|",
    ]
    report_lines.extend(
        f"| {row['emotion_left']} | {row['emotion_right']} | {row['silhouette']:.4f} | {row['permanova_pseudo_f']:.4f} | {row['permanova_p_value']:.6f} |"
        for row in pairwise
    )

    if nv_impact is not None:
        report_lines.extend(
            [
                "",
                "## NV Impact",
                f"- Paired rows: {nv_impact.get('paired_rows', 0)}",
            ],
        )
        for feature, shift in nv_impact.get("top_changed_features", []):
            report_lines.append(f"- {feature}: mean|Δ|={shift:.6f}")

    save_markdown("\n".join(report_lines), output_dir / "v01_separation_report.md")
    return results


def main() -> None:
    _configure_logging()
    parser = _build_parser()
    args = parser.parse_args()

    summary = run_evaluation(
        config_path=args.config,
        input_path=args.input_path,
        with_nv_input_path=args.with_nv_input_path,
        permutations=args.permutations,
    )
    logger.info("V-01評価完了 silhouette=%.4f", summary["silhouette_score"])


if __name__ == "__main__":
    main()
