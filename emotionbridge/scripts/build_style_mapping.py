import argparse
import json
import logging
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

from emotionbridge.constants import CONTROL_PARAM_NAMES, JVNV_EMOTION_LABELS
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

CONTROL_COLUMNS = [f"ctrl_{name}" for name in CONTROL_PARAM_NAMES]
EMOTION_PROB_COLUMNS = [f"emotion_{label}" for label in JVNV_EMOTION_LABELS]


VOICEVOX_STYLE_METADATA: dict[int, dict[str, str]] = {
    3: {"character": "zundamon", "style_name": "ノーマル"},
    1: {"character": "zundamon", "style_name": "あまあま"},
    7: {"character": "zundamon", "style_name": "ツンツン"},
    5: {"character": "zundamon", "style_name": "セクシー"},
    22: {"character": "zundamon", "style_name": "ささやき"},
    38: {"character": "zundamon", "style_name": "ヒソヒソ"},
    75: {"character": "zundamon", "style_name": "ヘロヘロ"},
    76: {"character": "zundamon", "style_name": "なみだめ"},
    2: {"character": "shikoku_metan", "style_name": "ノーマル"},
    0: {"character": "shikoku_metan", "style_name": "あまあま"},
    6: {"character": "shikoku_metan", "style_name": "ツンツン"},
    4: {"character": "shikoku_metan", "style_name": "セクシー"},
    36: {"character": "shikoku_metan", "style_name": "ささやき"},
    37: {"character": "shikoku_metan", "style_name": "ヒソヒソ"},
}


def _configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="スタイル重心と感情重心の距離行列からスタイルマッピングを構築する",
    )
    parser.add_argument(
        "--config",
        default="configs/experiment_config.yaml",
        help="実験設定ファイル",
    )
    parser.add_argument(
        "--jvnv-normalized",
        default=None,
        help="JVNV正規化特徴量parquet（未指定時はv01出力）",
    )
    parser.add_argument(
        "--voicevox-normalized",
        default=None,
        help="VOICEVOX正規化特徴量parquet（未指定時はv02出力）",
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


def _parse_target_style_ids(raw: str | None) -> list[int] | None:
    if raw is None or not raw.strip():
        return None

    values = [chunk.strip() for chunk in raw.split(",")]
    parsed = [int(value) for value in values if value]
    if not parsed:
        return None

    # 順序保持の重複除去
    return list(dict.fromkeys(parsed))


def _style_meta(style_id: int) -> dict[str, str]:
    meta = VOICEVOX_STYLE_METADATA.get(style_id)
    if meta is None:
        return {
            "character": "unknown",
            "style_name": f"style_{style_id}",
        }
    return meta


def _character_from_styles(style_ids: list[int]) -> str:
    characters = {_style_meta(style_id)["character"] for style_id in style_ids}
    if len(characters) == 1:
        return next(iter(characters))
    return "custom"


def _default_style_id(character: str, style_ids: list[int]) -> int:
    if character == "zundamon" and 3 in style_ids:
        return 3
    if character == "shikoku_metan" and 2 in style_ids:
        return 2
    return style_ids[0]


def _build_style_profiles(
    *,
    voice_df: pd.DataFrame,
    feature_cols: list[str],
    style_ids: list[int],
) -> tuple[dict[int, dict[str, Any]], dict[int, np.ndarray]]:
    style_profiles: dict[int, dict[str, Any]] = {}
    style_centroids: dict[int, np.ndarray] = {}

    for style_id in style_ids:
        subset = voice_df[voice_df["style_id"] == style_id]
        if subset.empty:
            continue

        mat = subset[feature_cols].to_numpy(dtype=np.float64)
        centroid = mat.mean(axis=0)
        std = mat.std(axis=0, ddof=0)

        style_centroids[style_id] = centroid
        style_profiles[style_id] = {
            "style_id": int(style_id),
            "character": _style_meta(style_id)["character"],
            "style_name": _style_meta(style_id)["style_name"],
            "num_samples": len(subset),
            "centroid": {
                feature: float(value) for feature, value in zip(feature_cols, centroid, strict=True)
            },
            "std": {
                feature: float(value) for feature, value in zip(feature_cols, std, strict=True)
            },
        }

    return style_profiles, style_centroids


def _max_pairwise_centroid_distance(style_centroids: dict[int, np.ndarray]) -> float:
    if len(style_centroids) <= 1:
        return 0.0

    keys = sorted(style_centroids.keys())
    max_distance = 0.0
    for i, key_i in enumerate(keys):
        for key_j in keys[i + 1 :]:
            distance = float(
                np.linalg.norm(style_centroids[key_i] - style_centroids[key_j]),
            )
            max_distance = max(max_distance, distance)
    return max_distance


def _normalize_prob_rows(matrix: np.ndarray) -> np.ndarray:
    clipped = np.clip(matrix, a_min=0.0, a_max=None)
    sums = clipped.sum(axis=1, keepdims=True)
    normalized = np.divide(
        clipped,
        sums,
        out=np.zeros_like(clipped),
        where=sums > 0.0,
    )
    zero_rows = (sums[:, 0] <= 0.0).astype(bool)
    if np.any(zero_rows):
        normalized[zero_rows] = 1.0 / float(clipped.shape[1])
    return normalized


def _rowwise_zscore(matrix: np.ndarray) -> np.ndarray:
    means = matrix.mean(axis=1, keepdims=True)
    stds = matrix.std(axis=1, ddof=0, keepdims=True)
    return np.divide(
        matrix - means,
        stds,
        out=np.zeros_like(matrix),
        where=stds > 1.0e-12,
    )


def _build_style_control_compatibility(
    *,
    raw_path: str | Path,
    target_style_ids: list[int],
    feature_cols: list[str],
    jvnv_centroids: dict[str, np.ndarray],
    distance_nested: dict[str, dict[str, float]],
    random_seed: int,
) -> dict[str, Any]:
    raw_df = read_parquet(resolve_path(raw_path)).copy()
    required = ["style_id", *CONTROL_COLUMNS, *EMOTION_PROB_COLUMNS, *feature_cols]
    ensure_columns(raw_df, required, where="voicevox raw (style compatibility)")
    raw_df = raw_df[raw_df["style_id"].isin(target_style_ids)].reset_index(drop=True)
    if raw_df.empty:
        msg = f"No rows remain after style_id filter in compatibility build: {target_style_ids}"
        raise RuntimeError(msg)

    feature_mat = raw_df[feature_cols].to_numpy(dtype=np.float64)
    means = feature_mat.mean(axis=0)
    stds = feature_mat.std(axis=0, ddof=0)
    if np.any(stds <= 0.0):
        zero_cols = [feature_cols[i] for i, value in enumerate(stds) if value <= 0.0]
        msg = f"Feature std must be positive for compatibility model: {zero_cols}"
        raise RuntimeError(msg)
    z_feature_mat = (feature_mat - means) / stds

    control_mat = raw_df[CONTROL_COLUMNS].to_numpy(dtype=np.float64)
    emotion_probs = _normalize_prob_rows(
        raw_df[EMOTION_PROB_COLUMNS].to_numpy(dtype=np.float64),
    )
    style_labels = raw_df["style_id"].to_numpy(dtype=np.int64)
    candidate_style_ids = sorted(
        int(style_id) for style_id in np.unique(style_labels).tolist() if int(style_id) in target_style_ids
    )
    if len(candidate_style_ids) < 2:
        msg = "style compatibility build requires at least two styles"
        raise RuntimeError(msg)

    centroid_matrix_rows: list[np.ndarray] = []
    for emotion in JVNV_EMOTION_LABELS:
        centroid = jvnv_centroids.get(emotion)
        if centroid is None:
            msg = f"Missing JVNV centroid for emotion: {emotion}"
            raise RuntimeError(msg)
        centroid_matrix_rows.append(np.asarray(centroid, dtype=np.float64))
    centroid_matrix = np.vstack(centroid_matrix_rows)
    if centroid_matrix.shape[1] != len(feature_cols):
        msg = "JVNV centroid dimension mismatch in compatibility build"
        raise RuntimeError(msg)

    distance_matrix = np.zeros((len(JVNV_EMOTION_LABELS), len(candidate_style_ids)), dtype=np.float64)
    for e_idx, emotion in enumerate(JVNV_EMOTION_LABELS):
        row = distance_nested.get(emotion)
        if not isinstance(row, dict):
            msg = f"Distance matrix row is missing for emotion: {emotion}"
            raise RuntimeError(msg)
        for s_idx, style_id in enumerate(candidate_style_ids):
            value = row.get(str(style_id))
            if not isinstance(value, (int, float)):
                msg = f"Distance matrix is missing value for emotion={emotion}, style_id={style_id}"
                raise RuntimeError(msg)
            distance_matrix[e_idx, s_idx] = float(value)

    rng = np.random.default_rng(random_seed)
    valid_mask = np.zeros(len(raw_df), dtype=bool)
    style_models_payload: dict[str, dict[str, Any]] = {}
    trained_models: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    style_split_rows: list[dict[str, Any]] = []
    for style_id in candidate_style_ids:
        idx = np.flatnonzero(style_labels == style_id)
        if idx.size < 2:
            msg = f"style_id={style_id} has too few samples for hold-out split: {idx.size}"
            raise RuntimeError(msg)
        shuffled = rng.permutation(idx)
        valid_size = max(1, int(np.floor(float(idx.size) * 0.2)))
        valid_size = min(valid_size, idx.size - 1)
        valid_idx = shuffled[:valid_size]
        train_idx = shuffled[valid_size:]
        valid_mask[valid_idx] = True

        x_train = control_mat[train_idx]
        y_train = z_feature_mat[train_idx]
        x_valid = control_mat[valid_idx]
        y_valid = z_feature_mat[valid_idx]

        model = Ridge(alpha=1.0, fit_intercept=True)
        model.fit(x_train, y_train)

        pred_train = model.predict(x_train)
        pred_valid = model.predict(x_valid)

        train_rmse = float(np.sqrt(np.mean(np.square(y_train - pred_train))))
        valid_rmse = float(np.sqrt(np.mean(np.square(y_valid - pred_valid))))

        coef = np.asarray(model.coef_, dtype=np.float64)
        intercept = np.asarray(model.intercept_, dtype=np.float64).reshape(-1)
        trained_models[style_id] = (coef, intercept)
        style_models_payload[str(style_id)] = {
            "style_id": int(style_id),
            "character": _style_meta(style_id)["character"],
            "style_name": _style_meta(style_id)["style_name"],
            "num_samples": int(idx.size),
            "num_train": int(train_idx.size),
            "num_valid": int(valid_idx.size),
            "ridge_alpha": 1.0,
            "train_rmse": train_rmse,
            "valid_rmse": valid_rmse,
            "coef": coef.tolist(),
            "intercept": intercept.tolist(),
        }
        style_split_rows.append(
            {
                "style_id": int(style_id),
                "num_samples": int(idx.size),
                "num_train": int(train_idx.size),
                "num_valid": int(valid_idx.size),
                "train_rmse": train_rmse,
                "valid_rmse": valid_rmse,
            },
        )

    valid_indices = np.flatnonzero(valid_mask)
    if valid_indices.size == 0:
        msg = "No validation rows available for lambda optimization"
        raise RuntimeError(msg)

    x_valid_all = control_mat[valid_indices]
    probs_valid_all = emotion_probs[valid_indices]
    target_features = probs_valid_all @ centroid_matrix
    base_scores_raw = -(probs_valid_all @ distance_matrix)

    compat_scores_raw = np.zeros_like(base_scores_raw)
    for s_idx, style_id in enumerate(candidate_style_ids):
        coef, intercept = trained_models[style_id]
        pred_features = x_valid_all @ coef.T + intercept
        compat_scores_raw[:, s_idx] = -np.linalg.norm(target_features - pred_features, axis=1)

    base_scores = _rowwise_zscore(base_scores_raw)
    compat_scores = _rowwise_zscore(compat_scores_raw)

    expected_distance_by_style = probs_valid_all @ distance_matrix
    lambda_grid = [round(value, 2) for value in np.arange(0.0, 1.0001, 0.05).tolist()]
    objective_rows: list[dict[str, Any]] = []
    best_lambda = 0.0
    best_objective = float("inf")
    best_selected_style_ids: np.ndarray | None = None
    for lam in lambda_grid:
        final_scores = base_scores + (float(lam) * compat_scores)
        selected_idx = np.argmax(final_scores, axis=1)
        row_objective = expected_distance_by_style[np.arange(len(selected_idx)), selected_idx]
        objective = float(np.mean(row_objective))
        objective_rows.append(
            {
                "lambda": float(lam),
                "mean_expected_distance": objective,
            },
        )
        if (objective < best_objective - 1.0e-12) or (
            abs(objective - best_objective) <= 1.0e-12 and float(lam) < best_lambda
        ):
            best_lambda = float(lam)
            best_objective = objective
            best_selected_style_ids = np.array(
                [candidate_style_ids[idx] for idx in selected_idx],
                dtype=np.int64,
            )

    if best_selected_style_ids is None:
        msg = "Failed to optimize lambda for style compatibility"
        raise RuntimeError(msg)

    selection_counts: dict[str, int] = {str(style_id): 0 for style_id in candidate_style_ids}
    unique_selected, unique_counts = np.unique(best_selected_style_ids, return_counts=True)
    for style_id, count in zip(unique_selected.tolist(), unique_counts.tolist(), strict=True):
        selection_counts[str(int(style_id))] = int(count)

    return {
        "version": "1.0",
        "created_at": datetime.now(tz=UTC).isoformat(),
        "feature_names": feature_cols,
        "control_names": CONTROL_PARAM_NAMES,
        "style_ids": candidate_style_ids,
        "holdout_ratio": 0.2,
        "random_seed": int(random_seed),
        "selection_metric": {
            "objective": "mean_expected_distance",
            "lambda_grid": lambda_grid,
            "best_lambda": best_lambda,
            "best_objective": best_objective,
            "selection_counts": selection_counts,
        },
        "objective_by_lambda": objective_rows,
        "style_models": style_models_payload,
        "style_split_summary": style_split_rows,
        "jvnv_emotion_centroids": {
            emotion: {
                feature: float(value)
                for feature, value in zip(feature_cols, jvnv_centroids[emotion], strict=True)
            }
            for emotion in JVNV_EMOTION_LABELS
        },
    }


def _build_raw_global_profiles(
    *,
    raw_path: str | Path,
    target_style_ids: list[int],
    jvnv_feature_cols: list[str],
) -> tuple[dict[int, dict[str, Any]], dict[int, np.ndarray], list[str], float] | None:
    path = resolve_path(raw_path)
    if not path.exists():
        return None

    raw_df = read_parquet(path).copy()
    ensure_columns(raw_df, ["style_id"], where="voicevox raw")
    raw_df = raw_df[raw_df["style_id"].isin(target_style_ids)].reset_index(drop=True)
    if raw_df.empty:
        return None

    raw_feature_cols = sorted(set(jvnv_feature_cols).intersection(_feature_cols(raw_df)))
    if not raw_feature_cols:
        return None

    means = raw_df[raw_feature_cols].mean(axis=0)
    stds = raw_df[raw_feature_cols].std(axis=0, ddof=0)
    valid_cols = sorted(stds[stds > 0.0].index.tolist())
    if not valid_cols:
        return None

    normalized_raw_df = raw_df[["style_id"]].copy()
    normalized_raw_df[valid_cols] = (raw_df[valid_cols] - means[valid_cols]) / stds[valid_cols]
    style_profiles, style_centroids = _build_style_profiles(
        voice_df=normalized_raw_df,
        feature_cols=valid_cols,
        style_ids=target_style_ids,
    )
    if not style_centroids:
        return None

    max_pairwise_distance = _max_pairwise_centroid_distance(style_centroids)
    return style_profiles, style_centroids, valid_cols, max_pairwise_distance


def _load_style_signal_status(
    *,
    metrics_path: str | Path,
    target_style_ids: list[int],
) -> tuple[str, str | None, str]:
    path = resolve_path(metrics_path)
    if not path.exists():
        return "unavailable", None, f"metrics file not found: {path}"

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        return "unavailable", str(path), f"failed to read metrics file: {exc}"

    status = str(payload.get("style_signal_status", "unavailable"))
    if status not in {"retain_style", "deprioritize_style"}:
        return "unavailable", str(path), "style_signal_status is missing or invalid"

    styles_raw = payload.get("styles_used")
    if isinstance(styles_raw, list):
        try:
            metric_styles = sorted(int(item) for item in styles_raw)
        except Exception:
            return "unavailable", str(path), "styles_used in metrics is invalid"
        if metric_styles != sorted(target_style_ids):
            return "unavailable", str(path), "styles_used mismatch between metrics and target styles"

    return status, str(path), "loaded from style influence metrics"


def run_build_style_mapping(
    *,
    config_path: str,
    jvnv_normalized: str | None,
    voicevox_normalized: str | None,
    target_style_ids_raw: str | None,
    output_dir: str | None,
) -> dict[str, str]:
    config = load_experiment_config(config_path)

    v01_dir = resolve_path(config.v01.output_dir)
    v02_dir = resolve_path(config.v02.output_dir)
    v03_dir = resolve_path(output_dir) if output_dir else resolve_path(config.v03.output_dir)
    output_root = resolve_path(config.paths.output_root)

    v03_dir.mkdir(parents=True, exist_ok=True)
    output_root.mkdir(parents=True, exist_ok=True)

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

    if not jvnv_path.exists() or not voicevox_path.exists():
        msg = f"Input file missing: jvnv={jvnv_path.exists()}, voicevox={voicevox_path.exists()}"
        raise FileNotFoundError(msg)

    jvnv_df = read_parquet(jvnv_path).copy()
    voicevox_df = read_parquet(voicevox_path).copy()

    ensure_columns(jvnv_df, ["emotion"], where="jvnv normalized")
    ensure_columns(voicevox_df, ["style_id"], where="voicevox normalized")

    jvnv_df["emotion_common6"] = jvnv_df["emotion"].map(_canon_jvnv)
    jvnv_df = jvnv_df[jvnv_df["emotion_common6"].notna()].reset_index(drop=True)

    target_style_ids = _parse_target_style_ids(target_style_ids_raw)
    if target_style_ids is None:
        target_style_ids = sorted(
            int(style_id) for style_id in voicevox_df["style_id"].unique().tolist()
        )

    voicevox_df = voicevox_df[voicevox_df["style_id"].isin(target_style_ids)].reset_index(drop=True)
    if voicevox_df.empty:
        msg = f"No rows remain after style_id filter: {target_style_ids}"
        raise RuntimeError(msg)

    jvnv_feature_cols = _feature_cols(jvnv_df)
    feature_cols = sorted(
        set(jvnv_feature_cols).intersection(_feature_cols(voicevox_df)),
    )
    if not feature_cols:
        msg = "No common eGeMAPS feature columns found"
        raise ValueError(msg)

    profile_source = "voicevox_egemaps_normalized(style_id_zscore)"
    profile_source_reason = "default_style_id_zscore_space"

    style_profiles, style_centroids = _build_style_profiles(
        voice_df=voicevox_df,
        feature_cols=feature_cols,
        style_ids=target_style_ids,
    )

    max_pairwise_distance = _max_pairwise_centroid_distance(style_centroids)
    style_signal_status, style_signal_metrics_path, style_signal_info = _load_style_signal_status(
        metrics_path=v03_dir / "style_influence_metrics.json",
        target_style_ids=target_style_ids,
    )

    raw_path = voicevox_path.parent / "voicevox_egemaps_raw.parquet"
    should_force_raw = style_signal_status == "retain_style"
    if should_force_raw:
        raw_profiles = _build_raw_global_profiles(
            raw_path=raw_path,
            target_style_ids=target_style_ids,
            jvnv_feature_cols=jvnv_feature_cols,
        )
        if raw_profiles is not None:
            style_profiles, style_centroids, feature_cols, max_pairwise_distance = raw_profiles
            profile_source = "voicevox_egemaps_raw(global_zscore_style_signal)"
            profile_source_reason = "style_signal_status=retain_style"
        else:
            profile_source_reason = (
                "style_signal_status=retain_style but raw global z-score profile could not be built"
            )

    if (
        not should_force_raw
        and max_pairwise_distance < float(config.v03.style_centroid_degeneracy_threshold)
    ):
        raw_profiles = _build_raw_global_profiles(
            raw_path=raw_path,
            target_style_ids=target_style_ids,
            jvnv_feature_cols=jvnv_feature_cols,
        )
        if raw_profiles is not None:
            style_profiles, style_centroids, feature_cols, max_pairwise_distance = raw_profiles
            profile_source = "voicevox_egemaps_raw(global_zscore_fallback)"
            profile_source_reason = "centroid_degeneracy_fallback"
            logger.warning(
                "Style centroid degeneration detected; fallback to raw global z-score space. max_pairwise_distance=%.6e threshold=%.6e",
                max_pairwise_distance,
                float(config.v03.style_centroid_degeneracy_threshold),
            )

    if not style_centroids:
        msg = "No style centroid could be computed"
        raise RuntimeError(msg)

    # JVNV emotion centroid
    jvnv_centroids: dict[str, np.ndarray] = {}
    for emotion in JVNV_EMOTION_LABELS:
        subset = jvnv_df[jvnv_df["emotion_common6"] == emotion]
        if subset.empty:
            continue
        jvnv_centroids[emotion] = subset[feature_cols].to_numpy(dtype=np.float64).mean(axis=0)

    # emotion x style distance matrix
    distance_records: list[dict[str, Any]] = []
    distance_nested: dict[str, dict[str, float]] = {}
    best_style_by_emotion: dict[str, dict[str, Any]] = {}

    for emotion in JVNV_EMOTION_LABELS:
        centroid = jvnv_centroids.get(emotion)
        if centroid is None:
            continue

        row_distance: dict[str, float] = {}
        for style_id in target_style_ids:
            if style_id not in style_centroids:
                continue
            distance = float(np.linalg.norm(centroid - style_centroids[style_id]))
            row_distance[str(style_id)] = distance
            distance_records.append(
                {
                    "emotion": emotion,
                    "style_id": int(style_id),
                    "character": _style_meta(style_id)["character"],
                    "style_name": _style_meta(style_id)["style_name"],
                    "distance": distance,
                },
            )

        distance_nested[emotion] = row_distance
        if row_distance:
            best_style_id = int(min(row_distance, key=lambda key: row_distance[key]))
            best_style_by_emotion[emotion] = {
                "style_id": best_style_id,
                "style_name": _style_meta(best_style_id)["style_name"],
                "character": _style_meta(best_style_id)["character"],
                "distance": float(row_distance[str(best_style_id)]),
            }

    distance_df = pd.DataFrame(distance_records)
    distance_parquet = v03_dir / "emotion_style_distance_matrix.parquet"
    write_parquet(distance_df, distance_parquet)

    styles_payload = {
        "version": "1.0",
        "created_at": datetime.now(tz=UTC).isoformat(),
        "source_voicevox_normalized": str(voicevox_path),
        "profile_source": profile_source,
        "profile_source_reason": profile_source_reason,
        "style_signal_status": style_signal_status,
        "style_signal_metrics_path": style_signal_metrics_path,
        "style_signal_info": style_signal_info,
        "max_pairwise_centroid_distance": max_pairwise_distance,
        "feature_count": len(feature_cols),
        "styles": {str(style_id): profile for style_id, profile in style_profiles.items()},
    }
    styles_json = v03_dir / "style_profiles.json"
    save_json(styles_payload, styles_json)

    distance_json_payload = {
        "version": "1.0",
        "created_at": datetime.now(tz=UTC).isoformat(),
        "source_jvnv_normalized": str(jvnv_path),
        "source_voicevox_normalized": str(voicevox_path),
        "profile_source": profile_source,
        "profile_source_reason": profile_source_reason,
        "style_signal_status": style_signal_status,
        "style_signal_metrics_path": style_signal_metrics_path,
        "style_signal_info": style_signal_info,
        "max_pairwise_centroid_distance": max_pairwise_distance,
        "feature_count": len(feature_cols),
        "distance_matrix": distance_nested,
        "best_style_by_emotion": best_style_by_emotion,
    }
    distance_json = v03_dir / "emotion_style_distance_matrix.json"
    save_json(distance_json_payload, distance_json)

    selection_policy = "prob_distance_only"
    scoring_payload: dict[str, Any] = {
        "base_term": "sum_p_emotion_negative_distance",
        "compat_term": "disabled",
        "normalization": "zscore_across_styles",
        "lambda": 0.0,
        "lambda_source": "disabled",
    }
    style_control_compatibility_json: Path | None = None
    if style_signal_status == "retain_style":
        if profile_source != "voicevox_egemaps_raw(global_zscore_style_signal)":
            msg = (
                "style_signal_status=retain_style requires raw global-z style profiles, "
                f"but profile_source is '{profile_source}'"
            )
            raise RuntimeError(msg)
        compatibility_payload = _build_style_control_compatibility(
            raw_path=raw_path,
            target_style_ids=target_style_ids,
            feature_cols=feature_cols,
            jvnv_centroids=jvnv_centroids,
            distance_nested=distance_nested,
            random_seed=int(config.v03.random_seed),
        )
        style_control_compatibility_json = v03_dir / "style_control_compatibility.json"
        save_json(compatibility_payload, style_control_compatibility_json)
        best_lambda = float(compatibility_payload["selection_metric"]["best_lambda"])
        selection_policy = "prob_distance_with_control_compat"
        scoring_payload = {
            "base_term": "sum_p_emotion_negative_distance",
            "compat_term": "negative_l2_to_style_control_prediction",
            "normalization": "zscore_across_styles",
            "lambda": best_lambda,
            "lambda_source": "style_control_compatibility",
        }

    selected_character = _character_from_styles(target_style_ids)
    default_style_id = _default_style_id(selected_character, target_style_ids)

    mapping_payload = {
        "version": "2.0",
        "created_at": datetime.now(tz=UTC).isoformat(),
        "selected_character": selected_character,
        "selection_policy": selection_policy,
        "scoring": scoring_payload,
        "styles_used": target_style_ids,
        "style_signal_status": style_signal_status,
        "style_signal_metrics_path": style_signal_metrics_path,
        "profile_source": profile_source,
        "profile_source_reason": profile_source_reason,
        "characters": {
            selected_character: {
                "mapping": best_style_by_emotion,
                "default_style": {
                    "style_id": int(default_style_id),
                    "style_name": _style_meta(default_style_id)["style_name"],
                },
            },
        },
        "artifacts": {
            "style_profiles": str(styles_json),
            "distance_matrix": str(distance_json),
            "distance_matrix_table": str(distance_parquet),
            "style_signal_metrics": style_signal_metrics_path,
            "style_control_compatibility": (
                str(style_control_compatibility_json) if style_control_compatibility_json else None
            ),
        },
    }

    mapping_in_v03 = v03_dir / "emotion_style_mapping.json"
    save_json(mapping_payload, mapping_in_v03)

    style_mapping = output_root / "style_mapping.json"
    save_json(mapping_payload, style_mapping)

    report_lines = [
        "# Emotion-Style Distance Report",
        "",
        f"- JVNV normalized: {jvnv_path}",
        f"- VOICEVOX normalized: {voicevox_path}",
        f"- Feature count: {len(feature_cols)}",
        f"- Target styles: {target_style_ids}",
        f"- Character: {selected_character}",
        f"- Profile source: {profile_source}",
        f"- Profile source reason: {profile_source_reason}",
        f"- Style signal status: {style_signal_status}",
        f"- Style signal metrics: {style_signal_metrics_path}",
        f"- Selection policy: {selection_policy}",
        f"- Scoring lambda: {float(scoring_payload['lambda']):.2f}",
        f"- Style-control compatibility: {style_control_compatibility_json}",
        "",
        "## Best Style by Emotion",
    ]

    for emotion in JVNV_EMOTION_LABELS:
        best = best_style_by_emotion.get(emotion)
        if best is None:
            continue
        report_lines.extend(
            [
                f"### {emotion}",
                f"- style_id: {best['style_id']}",
                f"- style_name: {best['style_name']}",
                f"- distance: {best['distance']:.6f}",
                "",
            ],
        )

    report_path = v03_dir / "emotion_style_distance_report.md"
    save_markdown("\n".join(report_lines), report_path)

    return {
        "style_profiles": str(styles_json),
        "distance_matrix": str(distance_json),
        "distance_table": str(distance_parquet),
        "emotion_style_mapping": str(mapping_in_v03),
        "style_mapping": str(style_mapping),
        "style_control_compatibility": (
            str(style_control_compatibility_json) if style_control_compatibility_json else ""
        ),
        "report": str(report_path),
    }


def main() -> None:
    _configure_logging()
    args = _build_parser().parse_args()

    result = run_build_style_mapping(
        config_path=args.config,
        jvnv_normalized=args.jvnv_normalized,
        voicevox_normalized=args.voicevox_normalized,
        target_style_ids_raw=args.target_style_ids,
        output_dir=args.output_dir,
    )
    logger.info("スタイルマッピング構築完了: %s", result["style_mapping"])


if __name__ == "__main__":
    main()
