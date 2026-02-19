import argparse
from datetime import UTC, datetime
from typing import Any

import pandas as pd

from emotionbridge.constants import CONTROL_PARAM_NAMES, JVNV_EMOTION_LABELS
from emotionbridge.scripts.common import (
    read_parquet,
    resolve_path,
    save_json,
    write_parquet,
)
from pathlib import Path


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="パラメータ生成器の教師表（感情6D→制御5D）を固定出力する",
    )
    parser.add_argument(
        "--recommended-params",
        default="artifacts/prosody/v03/recommended_params.json",
        help="韻律特徴ワークフローで生成したrecommended_params.json",
    )
    parser.add_argument(
        "--matches",
        default="artifacts/prosody/v03/emotion_param_matches.parquet",
        help="近傍サンプル（k件）を保持するParquet。存在すれば統計を付加",
    )
    parser.add_argument(
        "--output-dir",
        default="artifacts/generator/teacher_table",
        help="教師表の出力先ディレクトリ",
    )
    return parser


def _validate_recommended(payload: dict[str, Any]) -> dict[str, Any]:
    emotions = payload.get("emotions")
    if not isinstance(emotions, dict):
        msg = "Invalid recommended_params: 'emotions' must be a dictionary"
        raise ValueError(msg)

    missing_emotions = [emotion for emotion in JVNV_EMOTION_LABELS if emotion not in emotions]
    if missing_emotions:
        msg = f"Missing emotions in recommended_params: {missing_emotions}"
        raise ValueError(msg)

    for emotion in JVNV_EMOTION_LABELS:
        controls = emotions[emotion]
        if not isinstance(controls, dict):
            msg = f"Invalid controls for emotion={emotion}: expected dict"
            raise ValueError(msg)

        missing_controls = [
            f"ctrl_{name}" for name in CONTROL_PARAM_NAMES if f"ctrl_{name}" not in controls
        ]
        if missing_controls:
            msg = f"Missing controls for emotion={emotion}: {missing_controls}"
            raise ValueError(msg)

    return emotions


def _collect_match_stats(matches_path: Path) -> dict[str, Any] | None:
    if not matches_path.exists():
        return None

    matches_df = read_parquet(matches_path)
    if matches_df.empty:
        return None

    summary: dict[str, Any] = {}
    for emotion in JVNV_EMOTION_LABELS:
        subset = matches_df[matches_df["target_emotion"] == emotion]
        if subset.empty:
            continue

        summary[emotion] = {
            "num_rows": len(subset),
            "distance_mean": float(subset["distance_to_centroid"].mean()),
            "distance_std": float(subset["distance_to_centroid"].std(ddof=0)),
        }

    return summary


def run_prepare(
    *,
    recommended_params_path: str,
    matches_path: str,
    output_dir: str,
) -> dict[str, str]:
    recommended_path = resolve_path(recommended_params_path)
    if not recommended_path.exists():
        msg = f"recommended_params.json not found: {recommended_path}"
        raise FileNotFoundError(msg)

    output_root = resolve_path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)

    payload: dict[str, Any]
    with recommended_path.open("r", encoding="utf-8") as file:
        import json

        payload = json.load(file)

    emotions = _validate_recommended(payload)

    table_rows: list[dict[str, Any]] = []
    matrix_rows: list[list[float]] = []
    for emotion in JVNV_EMOTION_LABELS:
        controls = emotions[emotion]
        row = {
            "emotion": emotion,
            **{f"ctrl_{name}": float(controls[f"ctrl_{name}"]) for name in CONTROL_PARAM_NAMES},
        }
        table_rows.append(row)
        matrix_rows.append(
            [float(controls[f"ctrl_{name}"]) for name in CONTROL_PARAM_NAMES],
        )

    table_df = pd.DataFrame(table_rows)

    match_stats = _collect_match_stats(resolve_path(matches_path))

    teacher_payload: dict[str, Any] = {
        "version": "1.0",
        "created_at": datetime.now(tz=UTC).isoformat(),
        "source_recommended_params": str(recommended_path),
        "nearest_k": int(payload.get("nearest_k", 25)),
        "aggregation": str(payload.get("aggregation", "median")),
        "emotion_labels": JVNV_EMOTION_LABELS,
        "control_param_names": CONTROL_PARAM_NAMES,
        "table": table_rows,
        "matrix": matrix_rows,
    }
    if match_stats is not None:
        teacher_payload["match_stats"] = match_stats

    output_json = output_root / "recommended_params.json"
    output_parquet = output_root / "recommended_params.parquet"

    save_json(teacher_payload, output_json)
    write_parquet(table_df, output_parquet)

    return {
        "teacher_json": str(output_json),
        "teacher_table": str(output_parquet),
    }


def main() -> None:
    args = _build_parser().parse_args()
    result = run_prepare(
        recommended_params_path=args.recommended_params,
        matches_path=args.matches,
        output_dir=args.output_dir,
    )
    import json

    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
