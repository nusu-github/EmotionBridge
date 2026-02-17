from __future__ import annotations

import argparse
import logging
import operator
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import wasserstein_distance
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.neighbors import NearestNeighbors

from emotionbridge.scripts.common import (
    load_experiment_config,
    read_parquet,
    resolve_path,
    save_json,
    save_markdown,
)
from emotionbridge.scripts.visualize import plot_pca_scatter, plot_tsne_scatter

logger = logging.getLogger(__name__)


def _configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="V-03: JVNV-VOICEVOX ドメインギャップ評価",
    )
    parser.add_argument(
        "--config",
        default="configs/experiment_config.yaml",
        help="実験設定ファイル",
    )
    parser.add_argument("--jvnv-normalized", default=None, help="JVNV 正規化特徴量")
    parser.add_argument(
        "--voicevox-normalized",
        default=None,
        help="VOICEVOX 正規化特徴量",
    )
    parser.add_argument("--jvnv-raw", default=None, help="JVNV 生特徴量")
    parser.add_argument("--voicevox-raw", default=None, help="VOICEVOX 生特徴量")
    return parser


def _feature_columns(df: pd.DataFrame) -> list[str]:
    return sorted([name for name in df.columns if name.startswith("egemaps__")])


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


def _rbf_mmd(
    x: np.ndarray,
    y: np.ndarray,
    *,
    random_seed: int,
    max_samples: int = 4000,
) -> float:
    rng = np.random.default_rng(random_seed)
    if len(x) > max_samples:
        x = x[rng.choice(len(x), size=max_samples, replace=False)]
    if len(y) > max_samples:
        y = y[rng.choice(len(y), size=max_samples, replace=False)]

    combined = np.vstack([x, y])
    norms = np.sum(combined**2, axis=1, keepdims=True)
    dist_sq = norms + norms.T - 2 * combined @ combined.T
    median = np.median(dist_sq[np.triu_indices_from(dist_sq, k=1)])
    gamma = 1.0 / (2.0 * max(median, 1e-6))

    k_xx = rbf_kernel(x, x, gamma=gamma)
    k_yy = rbf_kernel(y, y, gamma=gamma)
    k_xy = rbf_kernel(x, y, gamma=gamma)

    n_x = len(x)
    n_y = len(y)
    if n_x < 2 or n_y < 2:
        return float("nan")

    term_xx = (k_xx.sum() - np.trace(k_xx)) / (n_x * (n_x - 1))
    term_yy = (k_yy.sum() - np.trace(k_yy)) / (n_y * (n_y - 1))
    term_xy = 2.0 * k_xy.mean()
    return float(term_xx + term_yy - term_xy)


def _wasserstein_by_feature(
    left: pd.DataFrame,
    right: pd.DataFrame,
    feature_cols: list[str],
) -> dict[str, float]:
    distances: dict[str, float] = {}
    for feature in feature_cols:
        distances[feature] = float(
            wasserstein_distance(
                left[feature].to_numpy(dtype=np.float64),
                right[feature].to_numpy(dtype=np.float64),
            ),
        )
    return distances


def _overlap_ratio(
    jvnv_features: np.ndarray,
    voicevox_features: np.ndarray,
) -> float:
    if len(jvnv_features) < 2 or len(voicevox_features) < 1:
        return float("nan")

    within_model = NearestNeighbors(n_neighbors=2).fit(jvnv_features)
    within_distances, _ = within_model.kneighbors(jvnv_features)
    within_reference = within_distances[:, 1]
    threshold = np.quantile(within_reference, 0.75)

    cross_model = NearestNeighbors(n_neighbors=1).fit(voicevox_features)
    cross_distances, _ = cross_model.kneighbors(jvnv_features)
    overlap = np.mean(cross_distances[:, 0] <= threshold)
    return float(overlap)


def run_evaluation(
    *,
    config_path: str,
    jvnv_normalized: str | None,
    voicevox_normalized: str | None,
    jvnv_raw: str | None,
    voicevox_raw: str | None,
) -> dict[str, Any]:
    config = load_experiment_config(config_path)
    v01_dir = resolve_path(config.v01.output_dir)
    v02_dir = resolve_path(config.v02.output_dir)
    v03_dir = resolve_path(config.v03.output_dir)
    v03_dir.mkdir(parents=True, exist_ok=True)

    jvnv_norm_path = (
        resolve_path(jvnv_normalized)
        if jvnv_normalized
        else v01_dir / "jvnv_egemaps_normalized.parquet"
    )
    voice_norm_path = (
        resolve_path(voicevox_normalized)
        if voicevox_normalized
        else v02_dir / "voicevox_egemaps_normalized.parquet"
    )
    jvnv_raw_path = (
        resolve_path(jvnv_raw) if jvnv_raw else v01_dir / "jvnv_egemaps_raw.parquet"
    )
    voice_raw_path = (
        resolve_path(voicevox_raw)
        if voicevox_raw
        else v02_dir / "voicevox_egemaps_raw.parquet"
    )

    if not jvnv_norm_path.exists() or not voice_norm_path.exists():
        msg = (
            "Normalized feature files are required. "
            f"jvnv={jvnv_norm_path.exists()}, voicevox={voice_norm_path.exists()}"
        )
        raise FileNotFoundError(msg)

    jvnv_norm = read_parquet(jvnv_norm_path).copy()
    voice_norm = read_parquet(voice_norm_path).copy()

    jvnv_norm["emotion_common6"] = jvnv_norm["emotion"].map(_canon_jvnv)
    voice_norm["emotion_common6"] = voice_norm["dominant_emotion"].map(_canon_voicevox)
    jvnv_norm = jvnv_norm[jvnv_norm["emotion_common6"].notna()].reset_index(drop=True)
    voice_norm = voice_norm[voice_norm["emotion_common6"].notna()].reset_index(
        drop=True,
    )

    common_features = sorted(
        set(_feature_columns(jvnv_norm)).intersection(_feature_columns(voice_norm)),
    )
    if not common_features:
        msg = "No overlapping eGeMAPS features found between JVNV and VOICEVOX normalized sets"
        raise ValueError(msg)

    norm_wasserstein = _wasserstein_by_feature(jvnv_norm, voice_norm, common_features)

    raw_wasserstein: dict[str, float] | None = None
    if jvnv_raw_path.exists() and voice_raw_path.exists():
        jvnv_raw_df = read_parquet(jvnv_raw_path)
        voice_raw_df = read_parquet(voice_raw_path)
        raw_features = sorted(
            set(_feature_columns(jvnv_raw_df)).intersection(
                _feature_columns(voice_raw_df),
            ),
        )
        raw_features = [name for name in raw_features if name in common_features]
        if raw_features:
            raw_wasserstein = _wasserstein_by_feature(
                jvnv_raw_df,
                voice_raw_df,
                raw_features,
            )

    by_emotion: dict[str, dict[str, float]] = {}
    for emotion in sorted(
        set(jvnv_norm["emotion_common6"]).intersection(
            set(voice_norm["emotion_common6"]),
        ),
    ):
        left = jvnv_norm[jvnv_norm["emotion_common6"] == emotion]
        right = voice_norm[voice_norm["emotion_common6"] == emotion]
        if len(left) < 2 or len(right) < 2:
            continue
        distances = _wasserstein_by_feature(left, right, common_features)
        by_emotion[emotion] = {
            "mean_wasserstein": float(np.mean(list(distances.values()))),
            "median_wasserstein": float(np.median(list(distances.values()))),
        }

    x_norm = jvnv_norm[common_features].to_numpy(dtype=np.float32, copy=False)
    y_norm = voice_norm[common_features].to_numpy(dtype=np.float32, copy=False)
    mmd_after = _rbf_mmd(x_norm, y_norm, random_seed=config.v03.random_seed)

    mmd_before = float("nan")
    if jvnv_raw_path.exists() and voice_raw_path.exists():
        jvnv_raw_df = read_parquet(jvnv_raw_path)
        voice_raw_df = read_parquet(voice_raw_path)
        raw_features = sorted(
            set(_feature_columns(jvnv_raw_df)).intersection(
                _feature_columns(voice_raw_df),
            ),
        )
        raw_features = [name for name in raw_features if name in common_features]
        if raw_features:
            x_raw = jvnv_raw_df[raw_features].to_numpy(dtype=np.float32, copy=False)
            y_raw = voice_raw_df[raw_features].to_numpy(dtype=np.float32, copy=False)
            mmd_before = _rbf_mmd(x_raw, y_raw, random_seed=config.v03.random_seed)

    overlap_ratio = _overlap_ratio(x_norm, y_norm)

    mean_after = float(np.mean(list(norm_wasserstein.values())))
    mean_before = (
        float(np.mean(list(raw_wasserstein.values())))
        if raw_wasserstein
        else float("nan")
    )
    shrink = float(mean_before - mean_after) if raw_wasserstein else float("nan")

    gate_pass = bool(raw_wasserstein and mean_after < mean_before)

    combined = pd.concat(
        [
            jvnv_norm.assign(domain="jvnv"),
            voice_norm.assign(domain="voicevox"),
        ],
        ignore_index=True,
    )
    sample_size = min(len(combined), 5000)
    combined_sampled = combined.sample(
        n=sample_size,
        random_state=config.v03.random_seed,
        replace=False,
    )
    plots_dir = v03_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    plot_pca_scatter(
        combined_sampled,
        common_features,
        color_col="emotion_common6",
        style_col="domain",
        output_path=plots_dir / "v03_pca_integrated.png",
        title="V-03 Integrated PCA",
        random_seed=config.v03.random_seed,
    )
    plot_tsne_scatter(
        combined_sampled,
        common_features,
        color_col="emotion_common6",
        style_col="domain",
        output_path=plots_dir / "v03_tsne_integrated.png",
        title="V-03 Integrated t-SNE",
        perplexity=config.v03.tsne_perplexity,
        random_seed=config.v03.random_seed,
    )

    top_features_after = sorted(
        norm_wasserstein.items(),
        key=operator.itemgetter(1),
        reverse=True,
    )[:15]
    top_shrunk: list[tuple[str, float]] = []
    if raw_wasserstein:
        top_shrunk = sorted(
            [
                (feature, raw_wasserstein[feature] - norm_wasserstein[feature])
                for feature in norm_wasserstein
                if feature in raw_wasserstein
            ],
            key=operator.itemgetter(1),
            reverse=True,
        )[:15]

    results = {
        "jvnv_normalized": str(jvnv_norm_path),
        "voicevox_normalized": str(voice_norm_path),
        "jvnv_raw": str(jvnv_raw_path) if jvnv_raw_path.exists() else None,
        "voicevox_raw": str(voice_raw_path) if voice_raw_path.exists() else None,
        "num_common_features": len(common_features),
        "wasserstein_mean_before": mean_before,
        "wasserstein_mean_after": mean_after,
        "wasserstein_mean_shrink": shrink,
        "mmd_before": mmd_before,
        "mmd_after": mmd_after,
        "overlap_ratio": overlap_ratio,
        "gate_pass": gate_pass,
        "by_emotion": by_emotion,
        "top_features_after": top_features_after,
        "top_shrunk_features": top_shrunk,
    }

    save_json(results, v03_dir / "v03_domain_gap_metrics.json")

    report_lines = [
        "# V-03 Domain Gap Report",
        "",
        "## Summary",
        f"- JVNV normalized: {jvnv_norm_path}",
        f"- VOICEVOX normalized: {voice_norm_path}",
        f"- Common features: {len(common_features)}",
        f"- Mean Wasserstein before normalization: {mean_before:.6f}",
        f"- Mean Wasserstein after normalization: {mean_after:.6f}",
        f"- Mean Wasserstein shrink: {shrink:.6f}",
        f"- MMD before: {mmd_before:.6f}",
        f"- MMD after: {mmd_after:.6f}",
        f"- Overlap ratio: {overlap_ratio:.4f}",
        f"- Gate result: {'Go' if gate_pass else 'No-Go'}",
        "",
        "## Emotion-wise Gap",
        "| Emotion | Mean Wasserstein | Median Wasserstein |",
        "|---|---:|---:|",
    ]
    for emotion, values in by_emotion.items():
        report_lines.append(
            f"| {emotion} | {values['mean_wasserstein']:.6f} | {values['median_wasserstein']:.6f} |",
        )

    report_lines.extend(
        [
            "",
            "## Top Features (After Normalization)",
        ],
    )
    for feature, distance in top_features_after:
        report_lines.append(f"- {feature}: {distance:.6f}")

    if top_shrunk:
        report_lines.extend(
            [
                "",
                "## Top Shrunk Features",
            ],
        )
        for feature, shrink_value in top_shrunk:
            report_lines.append(f"- {feature}: {shrink_value:.6f}")

    save_markdown("\n".join(report_lines), v03_dir / "v03_domain_gap_report.md")
    return results


def main() -> None:
    _configure_logging()
    parser = _build_parser()
    args = parser.parse_args()
    summary = run_evaluation(
        config_path=args.config,
        jvnv_normalized=args.jvnv_normalized,
        voicevox_normalized=args.voicevox_normalized,
        jvnv_raw=args.jvnv_raw,
        voicevox_raw=args.voicevox_raw,
    )
    logger.info("V-03評価完了: overlap=%.4f", summary["overlap_ratio"])


if __name__ == "__main__":
    main()
