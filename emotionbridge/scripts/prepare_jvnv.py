from __future__ import annotations

import argparse
import logging
import operator
import re
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import soundfile as sf

from emotionbridge.scripts.common import (
    load_experiment_config,
    resolve_path,
    save_json,
    write_parquet,
)

if TYPE_CHECKING:
    from pathlib import Path

logger = logging.getLogger(__name__)


def _configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="JVNV NV区間除外（無音化）前処理")
    parser.add_argument(
        "--config",
        default="configs/experiment_config.yaml",
        help="実験設定ファイル",
    )
    return parser


def _collect_nv_label_files(nv_label_dir: Path) -> dict[str, Path]:
    if not nv_label_dir.exists():
        return {}
    return {path.stem: path for path in sorted(nv_label_dir.rglob("*")) if path.is_file()}


def _parse_nv_intervals(
    label_path: Path,
    *,
    duration_sec: float,
) -> list[tuple[float, float]]:
    text = label_path.read_text(encoding="utf-8", errors="ignore")
    intervals: list[tuple[float, float]] = []

    for line in text.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        has_nv_tag = "nv" in stripped.lower()
        numeric_tokens = re.findall(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", stripped)
        if len(numeric_tokens) < 2:
            continue
        if not has_nv_tag and len(numeric_tokens) > 2:
            continue

        start = float(numeric_tokens[0])
        end = float(numeric_tokens[1])
        if end <= start:
            continue
        intervals.append((start, end))

    if not intervals:
        return []

    max_endpoint = max(end for _, end in intervals)
    if duration_sec > 0 and max_endpoint > duration_sec * 2:
        intervals = [(start / 1000.0, end / 1000.0) for start, end in intervals]

    sanitized: list[tuple[float, float]] = []
    for start, end in intervals:
        clamped_start = max(0.0, min(duration_sec, start))
        clamped_end = max(0.0, min(duration_sec, end))
        if clamped_end > clamped_start:
            sanitized.append((clamped_start, clamped_end))

    return sanitized


def _mask_intervals(
    audio: np.ndarray,
    sample_rate: int,
    intervals: list[tuple[float, float]],
) -> np.ndarray:
    masked = audio.copy()
    for start_sec, end_sec in intervals:
        start_idx = round(start_sec * sample_rate)
        end_idx = round(end_sec * sample_rate)
        start_idx = max(0, min(len(masked), start_idx))
        end_idx = max(0, min(len(masked), end_idx))
        if end_idx <= start_idx:
            continue
        masked[start_idx:end_idx] = 0.0
    return masked


def _excise_intervals(
    audio: np.ndarray,
    sample_rate: int,
    intervals: list[tuple[float, float]],
) -> np.ndarray:
    """NV区間を物理的に除去し、非NV区間を連結して返す。

    無音化（mask）と異なり、NV区間のサンプルを完全に除去する。
    eGeMAPS functionals（区間統計量）は無音混入の影響を受けなくなる。
    """
    if not intervals:
        return audio.copy()

    sorted_intervals = sorted(intervals, key=operator.itemgetter(0))
    segments: list[np.ndarray] = []
    current_pos = 0

    for start_sec, end_sec in sorted_intervals:
        start_idx = max(0, min(len(audio), round(start_sec * sample_rate)))
        end_idx = max(0, min(len(audio), round(end_sec * sample_rate)))

        if start_idx > current_pos:
            segments.append(audio[current_pos:start_idx])
        current_pos = max(current_pos, end_idx)

    if current_pos < len(audio):
        segments.append(audio[current_pos:])

    if not segments:
        return np.array([], dtype=audio.dtype)

    return np.concatenate(segments)


def _discover_jvnv_wavs(root_dir: Path) -> list[Path]:
    candidates = []
    for wav_path in sorted(root_dir.glob("*/*/*/*.wav")):
        if "nv_label" in wav_path.parts:
            continue
        candidates.append(wav_path)
    return candidates


def run_prepare(config_path: str) -> dict[str, object]:
    """JVNV 音声の NV 区間処理とマニフェスト生成を行う。

    v01.nv_handling の値に応じて NV 区間の処理方法を切り替える:
    - "excise": NV 区間を物理的に除去（切除後が短すぎる場合は mask にフォールバック）
    - "mask": NV 区間を無音化（ゼロ埋め）
    """
    config = load_experiment_config(config_path)
    jvnv_root = resolve_path(config.paths.jvnv_root)
    nv_label_dir = resolve_path(config.paths.jvnv_nv_label_dir)
    processed_dir = resolve_path(config.paths.jvnv_processed_audio_dir)
    output_dir = resolve_path(config.v01.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    processed_dir.mkdir(parents=True, exist_ok=True)

    if not jvnv_root.exists():
        msg = f"JVNV root not found: {jvnv_root}"
        raise FileNotFoundError(msg)

    wav_paths = _discover_jvnv_wavs(jvnv_root)
    if not wav_paths:
        msg = f"No JVNV wav files found under: {jvnv_root}"
        raise FileNotFoundError(msg)

    label_lookup = _collect_nv_label_files(nv_label_dir)

    rows: list[dict[str, object]] = []
    total_intervals = 0
    total_nv_seconds = 0.0

    for wav_path in wav_paths:
        relative = wav_path.relative_to(jvnv_root)
        parts = relative.parts
        speaker = parts[0] if len(parts) >= 1 else "unknown"
        emotion = parts[1] if len(parts) >= 2 else "unknown"
        session = parts[2] if len(parts) >= 3 else "unknown"
        utterance_id = wav_path.stem

        output_wav_path = processed_dir / relative
        output_wav_path.parent.mkdir(parents=True, exist_ok=True)

        audio_data, sample_rate = sf.read(wav_path, always_2d=False)
        duration_sec = float(len(audio_data) / sample_rate)

        label_path = label_lookup.get(utterance_id)
        intervals: list[tuple[float, float]] = []
        if label_path is not None:
            intervals = _parse_nv_intervals(label_path, duration_sec=duration_sec)

        if output_wav_path.exists() and not config.v01.overwrite_existing_nv_masked:
            pass
        else:
            nv_handling = config.v01.nv_handling
            if nv_handling == "excise" and intervals:
                processed_audio = _excise_intervals(audio_data, sample_rate, intervals)
                min_duration = 0.1
                if len(processed_audio) / sample_rate < min_duration:
                    logger.warning(
                        "切除後の音声が%.1f秒未満: %s (maskにフォールバック)",
                        min_duration,
                        utterance_id,
                    )
                    processed_audio = _mask_intervals(
                        audio_data,
                        sample_rate,
                        intervals,
                    )
                    nv_handling = "mask_fallback"
            else:
                processed_audio = _mask_intervals(audio_data, sample_rate, intervals)
            sf.write(output_wav_path, processed_audio, sample_rate)

        interval_count = len(intervals)
        nv_duration = float(sum(end - start for start, end in intervals))
        total_intervals += interval_count
        total_nv_seconds += nv_duration

        rows.append(
            {
                "utterance_id": utterance_id,
                "speaker": speaker,
                "emotion": emotion,
                "session": session,
                "audio_path_raw": str(wav_path),
                "audio_path_processed": str(output_wav_path),
                "nv_label_path": str(label_path) if label_path is not None else None,
                "nv_interval_count": interval_count,
                "nv_duration_seconds": nv_duration,
                "audio_duration_seconds": duration_sec,
                "nv_handling": config.v01.nv_handling,
                "sample_rate": sample_rate,
            },
        )

    manifest_df = pd.DataFrame(rows)
    manifest_path = output_dir / "jvnv_manifest.parquet"
    write_parquet(manifest_df, manifest_path)

    summary = {
        "jvnv_root": str(jvnv_root),
        "nv_label_dir": str(nv_label_dir),
        "processed_audio_dir": str(processed_dir),
        "manifest_path": str(manifest_path),
        "num_utterances": len(manifest_df),
        "num_with_nv_labels": int(manifest_df["nv_label_path"].notna().sum()),
        "total_nv_intervals": int(total_intervals),
        "total_nv_duration_seconds": float(total_nv_seconds),
        "mean_nv_ratio": float(
            (manifest_df["nv_duration_seconds"] / manifest_df["audio_duration_seconds"])
            .replace([np.inf, -np.inf], np.nan)
            .fillna(0.0)
            .mean(),
        ),
    }

    save_json(summary, output_dir / "prepare_jvnv_summary.json")
    return summary


def main() -> None:
    _configure_logging()
    parser = _build_parser()
    args = parser.parse_args()
    summary = run_prepare(args.config)
    logger.info("JVNV前処理完了: %s", summary["manifest_path"])


if __name__ == "__main__":
    main()
