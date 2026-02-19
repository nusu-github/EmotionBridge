from __future__ import annotations

import argparse
import logging
import os
from dataclasses import dataclass
from typing import Any, TYPE_CHECKING

import pandas as pd

from emotionbridge.scripts.common import (
    ensure_columns,
    load_experiment_config,
    read_parquet,
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
    parser = argparse.ArgumentParser(description="eGeMAPS 一括抽出スクリプト")
    parser.add_argument(
        "--config",
        default="configs/experiment_config.yaml",
        help="実験設定ファイル",
    )
    parser.add_argument(
        "--source",
        required=True,
        choices=["jvnv", "voicevox"],
        help="抽出対象",
    )
    parser.add_argument(
        "--jvnv-audio-key",
        choices=["processed", "raw"],
        default="processed",
        help="JVNV抽出時の入力音声種別",
    )

    # performance knobs
    parser.add_argument(
        "--num-workers",
        type=int,
        default=None,
        help=(
            "並列ワーカー数。未指定（None）の場合はライブラリが自動決定します。"
            "（マルチスレッド時: CPU*5 / マルチプロセス時: CPU）"
        ),
    )
    parser.add_argument(
        "--multiprocessing",
        action="store_true",
        help="マルチプロセスを使用（Windows で安定しやすいがオーバーヘッドは増える）",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=128,
        help="バッチ抽出のチャンクサイズ（大きいほど速いが失敗時の切り分けが重くなる）",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="出力 parquet が存在する場合、既存行（audio_path）をスキップして追記抽出する",
    )
    parser.add_argument(
        "--fail-fast",
        action="store_true",
        help="1件でも抽出失敗が起きたら即エラー終了（デバッグ用）",
    )

    return parser


def _resolve_opensmile_enums(feature_set: str, feature_level: str):
    try:
        import opensmile
    except ImportError as exc:
        msg = "opensmile がインストールされていません。`uv add opensmile` を実行してください。"
        raise ImportError(msg) from exc

    feature_set_map = {
        "eGeMAPSv02": opensmile.FeatureSet.eGeMAPSv02,
    }
    feature_level_map = {
        "Functionals": opensmile.FeatureLevel.Functionals,
    }
    if feature_set not in feature_set_map:
        msg = f"Unsupported feature_set: {feature_set}"
        raise ValueError(msg)
    if feature_level not in feature_level_map:
        msg = f"Unsupported feature_level: {feature_level}"
        raise ValueError(msg)
    return feature_set_map[feature_set], feature_level_map[feature_level]


def _build_extractor(
    config_path: str,
    *,
    num_workers: int | None,
    multiprocessing: bool,
):
    cfg = load_experiment_config(config_path)
    feature_set, feature_level = _resolve_opensmile_enums(
        cfg.extraction.feature_set,
        cfg.extraction.feature_level,
    )

    import opensmile

    # NOTE: num_workers=None triggers auto selection (see opensmile Smile docstring)
    return opensmile.Smile(
        feature_set=feature_set,
        feature_level=feature_level,
        num_workers=num_workers,
        multiprocessing=multiprocessing,
    )


def _path_key(p: str) -> str:
    # Normalized, stable join key across relative/absolute and case differences.
    # (Windows is case-insensitive, so normcase matters.)
    return os.path.normcase(os.path.realpath(os.path.abspath(os.path.expanduser(p))))


def _resolve_audio_path_fast(
    audio_path: str,
    *,
    audio_gen_output_dir: str,
    dataset_dir: str,
    cwd: str,
) -> str | None:
    # Fast path resolution without Path() overhead.
    p = audio_path
    if os.path.isabs(p):
        return p if os.path.exists(p) else None

    for base in (audio_gen_output_dir, dataset_dir, cwd):
        cand = os.path.join(base, p)
        if os.path.exists(cand):
            return cand

    return None


@dataclass(frozen=True)
class _BatchResult:
    features: pd.DataFrame
    failures: int


def _format_feature_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Convert opensmile output to a flat DataFrame.

    Expected input index for process_files() is MultiIndex with 'file', 'start', 'end'.
    This function:
      - extracts 'file' as 'audio_path'
      - drops start/end
      - prefixes feature columns with 'egemaps__'
    """
    if df.empty:
        return pd.DataFrame(columns=["audio_path"])  # consistent schema

    out = df.reset_index()
    if "file" not in out.columns:
        raise RuntimeError(
            "opensmile output missing 'file' index level; got columns: " + str(out.columns)
        )

    # Keep only one row per file for Functionals (defensive).
    # If duplicates exist, take the first occurrence.
    if out.duplicated(subset=["file"]).any():
        out = out.drop_duplicates(subset=["file"], keep="first")

    out = out.rename(columns={"file": "audio_path"})
    for col in ("start", "end"):
        if col in out.columns:
            out = out.drop(columns=[col])

    # Prefix features in bulk
    feature_cols = [c for c in out.columns if c != "audio_path"]
    out = out.rename(columns={c: f"egemaps__{c}" for c in feature_cols})

    # Downcast to float32 for size/speed unless mixed dtype
    for c in out.columns:
        if c.startswith("egemaps__"):
            out[c] = pd.to_numeric(out[c], errors="coerce")
            if pd.api.types.is_numeric_dtype(out[c]):
                out[c] = out[c].astype("float32")

    return out


def _process_files_or_split(
    smile,
    files: list[str],
    *,
    fail_fast: bool,
) -> _BatchResult:
    """Try to extract a batch. If it fails, split to isolate bad files."""

    def rec(batch: list[str]) -> tuple[list[pd.DataFrame], list[str]]:
        try:
            df = smile.process_files(batch)
            return [_format_feature_frame(df)], []
        except Exception as exc:
            if fail_fast:
                raise
            if len(batch) == 1:
                logger.warning("Extraction failed: %s | %s", batch[0], exc)
                return [], batch
            mid = len(batch) // 2
            left = batch[:mid]
            right = batch[mid:]
            g1, b1 = rec(left)
            g2, b2 = rec(right)
            return g1 + g2, b1 + b2

    good_frames, bad_files = rec(files)
    good_df = (
        pd.concat(good_frames, ignore_index=True)
        if good_frames
        else pd.DataFrame(columns=["audio_path"])
    )
    return _BatchResult(features=good_df, failures=len(bad_files))


def _extract_features_in_batches(
    smile,
    files: list[str],
    *,
    chunk_size: int,
    fail_fast: bool,
) -> _BatchResult:
    if not files:
        return _BatchResult(features=pd.DataFrame(columns=["audio_path"]), failures=0)

    frames: list[pd.DataFrame] = []
    failures = 0

    total = len(files)
    for i in range(0, total, chunk_size):
        chunk = files[i : i + chunk_size]
        logger.info("Extracting features: %d-%d / %d", i + 1, min(i + len(chunk), total), total)
        r = _process_files_or_split(smile, chunk, fail_fast=fail_fast)
        if not r.features.empty:
            frames.append(r.features)
        failures += r.failures

    out = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=["audio_path"])
    return _BatchResult(features=out, failures=failures)


def _maybe_resume_filter(
    *,
    output_path: Path,
    key_col: str,
    candidates: pd.DataFrame,
) -> pd.DataFrame:
    """If resume is enabled and output exists, drop already-extracted rows."""
    if not output_path.exists():
        return candidates

    try:
        existing = read_parquet(output_path)
    except Exception as exc:
        logger.warning("Failed to read existing output for resume (%s): %s", output_path, exc)
        return candidates

    if key_col not in existing.columns:
        logger.warning(
            "Resume requested but existing output missing '%s': %s", key_col, output_path
        )
        return candidates

    done = set(existing[key_col].astype(str).map(_path_key))
    before = len(candidates)
    keep = ~candidates[key_col].astype(str).map(_path_key).isin(done)
    filtered = candidates.loc[keep].copy()
    logger.info(
        "Resume: %d -> %d rows (skipped %d already done)",
        before,
        len(filtered),
        before - len(filtered),
    )
    return filtered


def _extract_jvnv(
    config_path: str,
    jvnv_audio_key: str,
    *,
    num_workers: int | None,
    multiprocessing: bool,
    chunk_size: int,
    resume: bool,
    fail_fast: bool,
) -> dict[str, Any]:
    cfg = load_experiment_config(config_path)
    v01_dir = resolve_path(cfg.v01.output_dir)

    manifest_path = v01_dir / "jvnv_manifest.parquet"
    if not manifest_path.exists():
        msg = (
            f"JVNV manifest not found: {manifest_path}. "
            "先に `python -m emotionbridge.scripts.prepare_jvnv` を実行してください。"
        )
        raise FileNotFoundError(msg)

    manifest_df = read_parquet(manifest_path)
    required = [
        "utterance_id",
        "speaker",
        "emotion",
        "session",
        "audio_path_raw",
        "audio_path_processed",
    ]
    ensure_columns(manifest_df, required, where="JVNV manifest")

    audio_col = "audio_path_processed" if jvnv_audio_key == "processed" else "audio_path_raw"

    meta_cols = ["utterance_id", "speaker", "emotion", "session", audio_col]
    meta = manifest_df[meta_cols].copy()
    meta = meta.rename(columns={audio_col: "audio_path"})
    meta["audio_path"] = meta["audio_path"].astype(str)

    # Existence check
    exists = meta["audio_path"].map(os.path.exists)
    missing_files = int((~exists).sum())
    meta = meta.loc[exists].copy()

    # Output path (kept identical to the original script)
    output_name = (
        "jvnv_egemaps_raw.parquet"
        if jvnv_audio_key == "processed"
        else "jvnv_egemaps_with_nv_raw.parquet"
    )
    output_path = v01_dir / output_name

    if resume:
        meta = _maybe_resume_filter(output_path=output_path, key_col="audio_path", candidates=meta)

    if meta.empty:
        raise RuntimeError("No JVNV files to process (all missing or already extracted).")

    files = meta["audio_path"].tolist()

    smile = _build_extractor(
        config_path,
        num_workers=num_workers,
        multiprocessing=multiprocessing,
    )

    batch = _extract_features_in_batches(
        smile,
        files,
        chunk_size=chunk_size,
        fail_fast=fail_fast,
    )

    if batch.features.empty:
        raise RuntimeError("No JVNV features were extracted.")

    # Join features with metadata using a normalized key (do not change output 'audio_path')
    features = batch.features.copy()
    features["__key"] = features["audio_path"].astype(str).map(_path_key)
    meta["__key"] = meta["audio_path"].astype(str).map(_path_key)

    merged = meta.merge(features.drop(columns=["audio_path"]), on="__key", how="inner")
    merged = merged.drop(columns=["__key"])

    # Add domain fields
    merged.insert(0, "source_domain", "jvnv")
    merged.insert(1, "variant", jvnv_audio_key)

    # Normalize dtypes
    for c in ("utterance_id", "speaker", "emotion", "session", "audio_path"):
        if c in merged.columns:
            merged[c] = merged[c].astype(str)

    write_parquet(merged, output_path)

    summary = {
        "source": "jvnv",
        "variant": jvnv_audio_key,
        "input_manifest": str(manifest_path),
        "output_path": str(output_path),
        "num_rows": len(merged),
        "num_features": len([c for c in merged.columns if c.startswith("egemaps__")]),
        "missing_audio_files": int(missing_files),
        "failed_extractions": int(batch.failures),
        "num_workers": num_workers,
        "multiprocessing": bool(multiprocessing),
        "chunk_size": int(chunk_size),
        "resume": bool(resume),
    }
    save_json(summary, v01_dir / f"extract_egemaps_jvnv_{jvnv_audio_key}_summary.json")
    return summary


def _extract_voicevox(
    config_path: str,
    *,
    num_workers: int | None,
    multiprocessing: bool,
    chunk_size: int,
    resume: bool,
    fail_fast: bool,
) -> dict[str, Any]:
    cfg = load_experiment_config(config_path)
    triplet_dataset_path = resolve_path(cfg.paths.triplet_dataset_path)
    audio_gen_output_dir = resolve_path(cfg.paths.audio_gen_output_dir)
    v02_dir = resolve_path(cfg.v02.output_dir)
    v02_dir.mkdir(parents=True, exist_ok=True)

    if not triplet_dataset_path.exists():
        msg = f"Triplet dataset not found: {triplet_dataset_path}"
        raise FileNotFoundError(msg)

    dataset_df = read_parquet(triplet_dataset_path)
    ensure_columns(dataset_df, ["audio_path"], where="triplet dataset")

    # Resolve relative paths
    dataset_dir = str(triplet_dataset_path.parent.parent)
    agod = str(audio_gen_output_dir)
    cwd = os.getcwd()

    audio_paths = dataset_df["audio_path"].astype(str).tolist()
    resolved: list[str | None] = [
        _resolve_audio_path_fast(p, audio_gen_output_dir=agod, dataset_dir=dataset_dir, cwd=cwd)
        for p in audio_paths
    ]

    keep_mask = [p is not None for p in resolved]
    missing_files = int(len(resolved) - sum(keep_mask))

    if sum(keep_mask) == 0:
        raise RuntimeError("No VOICEVOX files to process (all audio missing).")

    # Prepare metadata rows
    kept_df = dataset_df.loc[keep_mask].copy()
    kept_df["audio_path"] = [p for p in resolved if p is not None]

    output_path = v02_dir / "voicevox_egemaps_raw.parquet"

    if resume:
        kept_df = _maybe_resume_filter(
            output_path=output_path, key_col="audio_path", candidates=kept_df
        )

    if kept_df.empty:
        raise RuntimeError("No VOICEVOX files to process (already extracted).")

    files = kept_df["audio_path"].astype(str).tolist()

    smile = _build_extractor(
        config_path,
        num_workers=num_workers,
        multiprocessing=multiprocessing,
    )

    batch = _extract_features_in_batches(
        smile,
        files,
        chunk_size=chunk_size,
        fail_fast=fail_fast,
    )

    if batch.features.empty:
        raise RuntimeError("No VOICEVOX features were extracted.")

    # Keep the original script's column selection behavior
    metadata_columns = [
        "text_id",
        "text",
        "dominant_emotion",
        "style_id",
        "audio_path",
        "generation_timestamp",
    ]
    metadata_columns = [c for c in metadata_columns if c in kept_df.columns]
    passthrough_columns = [
        name for name in kept_df.columns if name.startswith(("ctrl_", "emotion_", "vv_"))
    ]

    meta_cols = [*metadata_columns, *passthrough_columns]
    meta = kept_df[meta_cols].copy()
    meta["audio_path"] = meta["audio_path"].astype(str)

    features = batch.features.copy()
    features["__key"] = features["audio_path"].astype(str).map(_path_key)
    meta["__key"] = meta["audio_path"].astype(str).map(_path_key)

    merged = meta.merge(features.drop(columns=["audio_path"]), on="__key", how="inner")
    merged = merged.drop(columns=["__key"])

    merged.insert(0, "source_domain", "voicevox")

    write_parquet(merged, output_path)

    summary = {
        "source": "voicevox",
        "input_dataset": str(triplet_dataset_path),
        "output_path": str(output_path),
        "num_rows": len(merged),
        "num_features": len([c for c in merged.columns if c.startswith("egemaps__")]),
        "missing_audio_files": int(missing_files),
        "failed_extractions": int(batch.failures),
        "num_workers": num_workers,
        "multiprocessing": bool(multiprocessing),
        "chunk_size": int(chunk_size),
        "resume": bool(resume),
    }
    save_json(summary, v02_dir / "extract_egemaps_voicevox_summary.json")
    return summary


def main() -> None:
    _configure_logging()
    parser = _build_parser()
    args = parser.parse_args()

    summary = (
        _extract_jvnv(
            args.config,
            args.jvnv_audio_key,
            num_workers=args.num_workers,
            multiprocessing=args.multiprocessing,
            chunk_size=args.chunk_size,
            resume=args.resume,
            fail_fast=args.fail_fast,
        )
        if args.source == "jvnv"
        else _extract_voicevox(
            args.config,
            num_workers=args.num_workers,
            multiprocessing=args.multiprocessing,
            chunk_size=args.chunk_size,
            resume=args.resume,
            fail_fast=args.fail_fast,
        )
    )

    logger.info("eGeMAPS抽出完了: %s", summary["output_path"])


if __name__ == "__main__":
    main()
