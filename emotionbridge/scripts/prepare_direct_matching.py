import argparse
import logging
from datetime import UTC, datetime
from typing import Any

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
from emotionbridge.scripts.match_emotion_params import run_matching

logger = logging.getLogger(__name__)


def _configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="直接マッチング用プロファイル生成（韻律特徴空間ベース）",
    )
    parser.add_argument(
        "--config",
        default="configs/experiment_config.yaml",
        help="実験設定ファイル",
    )
    parser.add_argument(
        "--nearest-k",
        type=int,
        default=None,
        help="近傍件数（未指定時は config.v03.nearest_k）",
    )
    parser.add_argument(
        "--jvnv-normalized",
        default=None,
        help="JVNV正規化特徴量の明示パス",
    )
    parser.add_argument(
        "--voicevox-normalized",
        default=None,
        help="VOICEVOX正規化特徴量の明示パス",
    )
    return parser


def _control_columns(df: pd.DataFrame) -> list[str]:
    return sorted([name for name in df.columns if name.startswith("ctrl_")])


def _stats(values: np.ndarray) -> dict[str, float]:
    return {
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "std": float(np.std(values, ddof=0)),
        "min": float(np.min(values)),
        "max": float(np.max(values)),
    }


def _build_profile(
    rows: pd.DataFrame,
    control_cols: list[str],
) -> dict[str, Any]:
    if rows.empty:
        msg = "No rows to profile"
        raise ValueError(msg)

    control_stats: dict[str, dict[str, float]] = {}
    recommended: dict[str, float] = {}
    for control in control_cols:
        values = rows[control].to_numpy(dtype=np.float64)
        stats = _stats(values)
        control_stats[control] = stats
        recommended[control] = float(np.clip(stats["median"], -1.0, 1.0))

    distances = rows["distance_to_centroid"].to_numpy(dtype=np.float64)
    return {
        "num_matches": len(rows),
        "distance": {
            "mean": float(np.mean(distances)),
            "median": float(np.median(distances)),
            "p90": float(np.quantile(distances, 0.9)),
        },
        "control_stats": control_stats,
        "recommended_control": recommended,
    }


def run_prepare(
    *,
    config_path: str,
    nearest_k: int | None,
    jvnv_normalized: str | None,
    voicevox_normalized: str | None,
) -> dict[str, Any]:
    config = load_experiment_config(config_path)
    v03_dir = resolve_path(config.v03.output_dir)
    v03_dir.mkdir(parents=True, exist_ok=True)

    matching_payload = run_matching(
        config_path=config_path,
        jvnv_normalized=jvnv_normalized,
        voicevox_normalized=voicevox_normalized,
        nearest_k=nearest_k,
    )

    matches_path = resolve_path(matching_payload["output_matches"])
    matches_df = read_parquet(matches_path)
    control_cols = _control_columns(matches_df)
    if not control_cols:
        msg = "No control columns found in matched table"
        raise ValueError(msg)

    target_emotions = config.v03.emotion_labels_common6
    global_profile = _build_profile(matches_df, control_cols)

    per_emotion: dict[str, Any] = {}
    profile_rows: list[dict[str, Any]] = []
    for emotion in target_emotions:
        subset = matches_df[matches_df["target_emotion"] == emotion]
        if subset.empty:
            per_emotion[emotion] = {
                "fallback": "global",
                "recommended_control": global_profile["recommended_control"],
                "num_matches": 0,
            }
            profile_rows.append(
                {
                    "emotion": emotion,
                    **global_profile["recommended_control"],
                    "num_matches": 0,
                    "distance_mean": float("nan"),
                },
            )
            continue

        profile = _build_profile(subset, control_cols)
        per_emotion[emotion] = profile
        profile_rows.append(
            {
                "emotion": emotion,
                **profile["recommended_control"],
                "num_matches": profile["num_matches"],
                "distance_mean": profile["distance"]["mean"],
            },
        )

    payload = {
        "created_at": datetime.now(tz=UTC).isoformat(),
        "config_path": str(resolve_path(config_path)),
        "matching_output": str(matches_path),
        "nearest_k": int(nearest_k if nearest_k is not None else config.v03.nearest_k),
        "target_emotions": target_emotions,
        "global_profile": global_profile,
        "emotion_profiles": per_emotion,
    }

    output_json = v03_dir / "direct_matching_profiles.json"
    output_report = v03_dir / "direct_matching_profiles_report.md"
    output_table = v03_dir / "direct_matching_profiles.parquet"

    save_json(payload, output_json)
    write_parquet(pd.DataFrame(profile_rows), output_table)

    report_lines = [
        "# Direct Matching Profile Report",
        "",
        "## Summary",
        f"- Matching output: {matches_path}",
        f"- Nearest k: {payload['nearest_k']}",
        f"- Target emotions: {target_emotions}",
        "",
        "## Recommended Controls",
        "| Emotion | pitch_shift | pitch_range | speed | energy | pause_weight | matches | dist_mean |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]

    report_lines.extend(
        f"| {row['emotion']} | {row.get('ctrl_pitch_shift', 0.0):.4f} | {row.get('ctrl_pitch_range', 0.0):.4f} | {row.get('ctrl_speed', 0.0):.4f} | {row.get('ctrl_energy', 0.0):.4f} | {row.get('ctrl_pause_weight', 0.0):.4f} | {int(row['num_matches'])} | {row['distance_mean']:.6f} |"
        for row in profile_rows
    )

    save_markdown("\n".join(report_lines), output_report)

    return {
        "profiles_json": str(output_json),
        "profiles_table": str(output_table),
        "profiles_report": str(output_report),
        "matching_output": str(matches_path),
    }


def main() -> None:
    _configure_logging()
    args = _build_parser().parse_args()
    summary = run_prepare(
        config_path=args.config,
        nearest_k=args.nearest_k,
        jvnv_normalized=args.jvnv_normalized,
        voicevox_normalized=args.voicevox_normalized,
    )
    logger.info("直接マッチング準備完了: %s", summary["profiles_json"])


if __name__ == "__main__":
    main()
