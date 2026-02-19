from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd

from emotionbridge.constants import JVNV_EMOTION_LABELS


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="主観評価回答CSVを集計してA/B・MOS・感情識別の結果を出力する",
    )
    parser.add_argument(
        "--eval-dir",
        default="artifacts/prosody/subjective_eval/pilot",
        help="prepare_subjective_eval.py の出力ディレクトリ",
    )
    parser.add_argument(
        "--ab-path",
        default=None,
        help="A/B回答CSV（未指定時: responses/ab_responses.csv）",
    )
    parser.add_argument(
        "--mos-path",
        default=None,
        help="MOS回答CSV（未指定時: responses/mos_responses.csv）",
    )
    parser.add_argument(
        "--emotion-path",
        default=None,
        help="感情識別回答CSV（未指定時: responses/emotion_identification_responses.csv）",
    )
    return parser


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


def _safe_read_csv(path: Path) -> pd.DataFrame | None:
    if not path.exists() or path.stat().st_size == 0:
        return None
    df = pd.read_csv(path)
    if df.empty:
        return None
    return df


def _summarize_ab(
    stimuli_key: pd.DataFrame,
    responses: pd.DataFrame | None,
) -> dict[str, Any]:
    if responses is None:
        return {
            "available": False,
            "message": "A/B responses not found",
        }

    required = ["sample_id", "preferred"]
    missing = [name for name in required if name not in responses.columns]
    if missing:
        return {
            "available": False,
            "message": f"A/B response missing columns: {missing}",
        }

    merged = responses.merge(stimuli_key, on="sample_id", how="inner")
    if merged.empty:
        return {
            "available": False,
            "message": "A/B response rows do not match stimuli keys",
        }

    def _is_bridge(row: pd.Series) -> bool | None:
        preferred = str(row["preferred"]).strip().upper()
        if preferred == "A":
            return str(row["condition_a"]).strip().lower() == "emotionbridge"
        if preferred == "B":
            return str(row["condition_b"]).strip().lower() == "emotionbridge"
        return None

    bridge_flags = pd.Series(
        [_is_bridge(row) for _, row in merged.iterrows()],
        index=merged.index,
        dtype="object",
    )
    valid = bridge_flags.dropna().astype(bool)

    bridge_rate = float(valid.mean()) if len(valid) > 0 else None
    tie_count = int((bridge_flags.isna()).sum())

    return {
        "available": True,
        "num_rows": len(merged),
        "num_valid_ab": len(valid),
        "tie_or_invalid": tie_count,
        "bridge_preference_rate": bridge_rate,
    }


def _summarize_mos(responses: pd.DataFrame | None) -> dict[str, Any]:
    if responses is None:
        return {
            "available": False,
            "message": "MOS responses not found",
        }

    required = ["sample_id", "naturalness_mos_1to5", "emotion_fit_mos_1to5"]
    missing = [name for name in required if name not in responses.columns]
    if missing:
        return {
            "available": False,
            "message": f"MOS response missing columns: {missing}",
        }

    numeric = responses.copy()
    numeric["naturalness_mos_1to5"] = pd.to_numeric(
        numeric["naturalness_mos_1to5"],
        errors="coerce",
    )
    numeric["emotion_fit_mos_1to5"] = pd.to_numeric(
        numeric["emotion_fit_mos_1to5"],
        errors="coerce",
    )
    numeric = numeric.dropna(subset=["naturalness_mos_1to5", "emotion_fit_mos_1to5"])

    if numeric.empty:
        return {
            "available": False,
            "message": "MOS responses have no valid numeric rows",
        }

    return {
        "available": True,
        "num_rows": len(numeric),
        "naturalness_mean": float(numeric["naturalness_mos_1to5"].mean()),
        "emotion_fit_mean": float(numeric["emotion_fit_mos_1to5"].mean()),
    }


def _summarize_emotion_identification(
    stimuli_key: pd.DataFrame,
    responses: pd.DataFrame | None,
) -> dict[str, Any]:
    if responses is None:
        return {
            "available": False,
            "message": "Emotion identification responses not found",
        }

    required = ["sample_id", "predicted_emotion"]
    missing = [name for name in required if name not in responses.columns]
    if missing:
        return {
            "available": False,
            "message": f"Emotion response missing columns: {missing}",
        }

    merged = responses.merge(
        stimuli_key[["sample_id", "target_emotion"]],
        on="sample_id",
        how="inner",
    )
    if merged.empty:
        return {
            "available": False,
            "message": "Emotion response rows do not match stimuli keys",
        }

    merged["predicted_common6"] = merged["predicted_emotion"].map(_canon_common6)
    merged = merged[merged["predicted_common6"].notna()].reset_index(drop=True)
    if merged.empty:
        return {
            "available": False,
            "message": "No valid common6 emotion labels in responses",
        }

    merged["correct"] = merged["predicted_common6"] == merged["target_emotion"]

    by_emotion: dict[str, float] = {}
    for emotion in JVNV_EMOTION_LABELS:
        subset = merged[merged["target_emotion"] == emotion]
        if subset.empty:
            continue
        by_emotion[emotion] = float(subset["correct"].mean())

    return {
        "available": True,
        "num_rows": len(merged),
        "overall_accuracy": float(merged["correct"].mean()),
        "by_emotion_accuracy": by_emotion,
    }


def run_analysis(
    *,
    eval_dir: str,
    ab_path: str | None,
    mos_path: str | None,
    emotion_path: str | None,
) -> dict[str, Any]:
    root = Path(eval_dir)
    key_path = root / "manifests" / "stimuli_key.csv"
    if not key_path.exists():
        msg = f"stimuli key not found: {key_path}"
        raise FileNotFoundError(msg)

    stimuli_key = pd.read_csv(key_path)

    ab_file = Path(ab_path) if ab_path else root / "responses" / "ab_responses.csv"
    mos_file = Path(mos_path) if mos_path else root / "responses" / "mos_responses.csv"
    emo_file = (
        Path(emotion_path)
        if emotion_path
        else root / "responses" / "emotion_identification_responses.csv"
    )

    ab_df = _safe_read_csv(ab_file)
    mos_df = _safe_read_csv(mos_file)
    emo_df = _safe_read_csv(emo_file)

    result = {
        "eval_dir": str(root),
        "stimuli_key": str(key_path),
        "ab": _summarize_ab(stimuli_key, ab_df),
        "mos": _summarize_mos(mos_df),
        "emotion_identification": _summarize_emotion_identification(
            stimuli_key,
            emo_df,
        ),
    }

    summary_path = root / "subjective_eval_summary.json"
    with summary_path.open("w", encoding="utf-8") as file:
        json.dump(result, file, ensure_ascii=False, indent=2)

    lines = [
        "# Subjective Evaluation Summary",
        "",
        f"- Eval dir: {root}",
        "",
        "## A/B",
        f"- available: {result['ab'].get('available')}",
        f"- bridge_preference_rate: {result['ab'].get('bridge_preference_rate')}",
        "",
        "## MOS",
        f"- available: {result['mos'].get('available')}",
        f"- naturalness_mean: {result['mos'].get('naturalness_mean')}",
        f"- emotion_fit_mean: {result['mos'].get('emotion_fit_mean')}",
        "",
        "## Emotion Identification",
        f"- available: {result['emotion_identification'].get('available')}",
        f"- overall_accuracy: {result['emotion_identification'].get('overall_accuracy')}",
    ]

    report_path = root / "subjective_eval_summary.md"
    report_path.write_text("\n".join(lines), encoding="utf-8")

    result["summary_json"] = str(summary_path)
    result["summary_md"] = str(report_path)
    return result


def main() -> None:
    args = _build_parser().parse_args()
    result = run_analysis(
        eval_dir=args.eval_dir,
        ab_path=args.ab_path,
        mos_path=args.mos_path,
        emotion_path=args.emotion_path,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
