from __future__ import annotations

import argparse
import csv
import json
import logging
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import numpy as np
import pyworld as pw
import soundfile as sf
from scipy.signal import resample_poly

from emotionbridge.constants import JVNV_EMOTION_LABELS
from emotionbridge.scripts.common import save_json, save_markdown

logger = logging.getLogger(__name__)

MCD_ORDER = 24
MCD_CONST = float((10.0 / np.log(10.0)) * np.sqrt(2.0))
SUBJECTIVE_GATE_STATUS = "pending_not_in_scope"
GO_THRESHOLDS = {
    "pesq_min": 3.5,
    "mcd_max_db": 6.0,
    "f0_rmse_max_hz": 5.0,
}
NO_GO_RECOMMENDED_ACTIONS = [
    "WORLDのパラメータ調整（DIOの代わりにHarvestを使用）",
    "ParselmouthによるPraat操作に切り替え",
    "rubberband + Sox の組み合わせで近似操作",
    "DSP後処理アプローチを断念しVC学習ルートへ移行",
]


@dataclass(frozen=True, slots=True)
class ManifestEntry:
    target_emotion: str
    text: str
    audio_path: Path

    @property
    def key(self) -> tuple[str, str]:
        return self.target_emotion, self.text


def _configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Roundtrip定量評価（PESQ/MCD/F0 RMSE）を実行し、"
            "demo baseline/candidate manifestを比較する"
        ),
    )
    parser.add_argument(
        "--baseline-manifest",
        default="demo/v2/manifest.json",
        help="baseline側 manifest JSON",
    )
    parser.add_argument(
        "--candidate-manifest",
        default="demo/v2-dsp/manifest.json",
        help="candidate側 manifest JSON",
    )
    parser.add_argument(
        "--output-dir",
        default="artifacts/prosody/roundtrip_eval/v2_dsp",
        help="評価レポート出力先ディレクトリ",
    )
    return parser


def _resolve_audio_path(audio_path: str, *, manifest_dir: Path) -> Path:
    raw = Path(audio_path)
    if raw.is_absolute():
        return raw

    cwd_resolved = (Path.cwd() / raw).resolve()
    if cwd_resolved.exists():
        return cwd_resolved

    return (manifest_dir / raw).resolve()


def _load_manifest_entries(manifest_path: str | Path) -> dict[tuple[str, str], ManifestEntry]:
    path = Path(manifest_path)
    if not path.exists():
        msg = f"manifest not found: {path}"
        raise FileNotFoundError(msg)

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        msg = f"invalid JSON manifest: {path}"
        raise ValueError(msg) from exc

    if not isinstance(payload, list):
        msg = f"manifest must be JSON array: {path}"
        raise ValueError(msg)

    required = {"target_emotion", "text", "audio_path"}
    entries: dict[tuple[str, str], ManifestEntry] = {}

    for idx, row in enumerate(payload, start=1):
        if not isinstance(row, dict):
            msg = f"manifest row #{idx} must be object"
            raise ValueError(msg)

        missing = sorted(required.difference(row))
        if missing:
            msg = f"manifest row #{idx} missing required keys: {missing}"
            raise ValueError(msg)

        target_emotion = str(row["target_emotion"]).strip()
        text = str(row["text"])
        audio_path = _resolve_audio_path(str(row["audio_path"]), manifest_dir=path.parent)
        key = (target_emotion, text)

        if key in entries:
            msg = f"duplicate manifest key (target_emotion, text): {key}"
            raise ValueError(msg)

        if not audio_path.exists():
            msg = f"audio file not found for key={key}: {audio_path}"
            raise FileNotFoundError(msg)

        entries[key] = ManifestEntry(
            target_emotion=target_emotion,
            text=text,
            audio_path=audio_path,
        )

    return entries


def _pair_manifest_entries(
    baseline_entries: dict[tuple[str, str], ManifestEntry],
    candidate_entries: dict[tuple[str, str], ManifestEntry],
) -> list[tuple[ManifestEntry, ManifestEntry]]:
    baseline_keys = set(baseline_entries)
    candidate_keys = set(candidate_entries)

    missing_in_candidate = sorted(baseline_keys - candidate_keys)
    extra_in_candidate = sorted(candidate_keys - baseline_keys)
    if missing_in_candidate or extra_in_candidate:
        msg = (
            "manifest key mismatch detected. "
            f"missing_in_candidate={missing_in_candidate}, "
            f"extra_in_candidate={extra_in_candidate}"
        )
        raise ValueError(msg)

    return [(baseline_entries[key], candidate_entries[key]) for key in sorted(baseline_keys)]


def _read_audio_mono(path: Path) -> tuple[np.ndarray, int]:
    wav, sr = sf.read(path, dtype="float64", always_2d=True)
    if wav.shape[0] == 0:
        msg = f"empty audio: {path}"
        raise ValueError(msg)
    mono = np.mean(wav, axis=1, dtype=np.float64)
    return mono, int(sr)


def _resample_to_rate(wav: np.ndarray, src_sr: int, dst_sr: int) -> np.ndarray:
    if src_sr == dst_sr:
        return np.asarray(wav, dtype=np.float64)

    gcd = math.gcd(src_sr, dst_sr)
    up = dst_sr // gcd
    down = src_sr // gcd
    return np.asarray(resample_poly(wav, up=up, down=down), dtype=np.float64)


def _align_pair(
    baseline_wav: np.ndarray,
    baseline_sr: int,
    candidate_wav: np.ndarray,
    candidate_sr: int,
) -> tuple[np.ndarray, np.ndarray, int]:
    ref = np.asarray(baseline_wav, dtype=np.float64)
    deg = np.asarray(candidate_wav, dtype=np.float64)

    if candidate_sr != baseline_sr:
        deg = _resample_to_rate(deg, candidate_sr, baseline_sr)

    min_len = min(ref.size, deg.size)
    if min_len < 1:
        msg = "audio alignment failed: no overlapping samples"
        raise ValueError(msg)

    return ref[:min_len], deg[:min_len], baseline_sr


def _get_pesq_function():
    try:
        from pesq import pesq
    except ModuleNotFoundError as exc:
        msg = (
            "pesq library is required for PESQ calculation. Install dependencies and retry: uv sync"
        )
        raise RuntimeError(msg) from exc
    return pesq


def _compute_pesq_wb(ref: np.ndarray, deg: np.ndarray, sr: int) -> float:
    pesq_fn = _get_pesq_function()
    ref_16k = _resample_to_rate(ref, sr, 16000)
    deg_16k = _resample_to_rate(deg, sr, 16000)
    min_len = min(ref_16k.size, deg_16k.size)
    if min_len < 16:
        msg = "audio too short for PESQ"
        raise ValueError(msg)
    score = pesq_fn(
        16000,
        ref_16k[:min_len].astype(np.float32),
        deg_16k[:min_len].astype(np.float32),
        "wb",
    )
    return float(score)


def _world_analyze(wav: np.ndarray, sr: int) -> tuple[np.ndarray, np.ndarray]:
    world = cast("Any", pw)
    wav64 = np.asarray(wav, dtype=np.float64)
    f0, time_axis = world.dio(wav64, sr, frame_period=5.0)
    f0 = world.stonemask(wav64, f0, time_axis, sr)
    sp = world.cheaptrick(wav64, f0, time_axis, sr)
    coded_sp = world.code_spectral_envelope(sp, sr, MCD_ORDER)
    return f0, coded_sp


def _compute_world_metrics(ref: np.ndarray, deg: np.ndarray, sr: int) -> dict[str, float | int]:
    f0_ref, coded_ref = _world_analyze(ref, sr)
    f0_deg, coded_deg = _world_analyze(deg, sr)

    frame_count = min(coded_ref.shape[0], coded_deg.shape[0], f0_ref.size, f0_deg.size)
    if frame_count < 1:
        msg = "WORLD analysis produced zero frames"
        raise ValueError(msg)

    ref_cep = coded_ref[:frame_count, 1:]
    deg_cep = coded_deg[:frame_count, 1:]
    if ref_cep.shape[1] < 1 or deg_cep.shape[1] < 1:
        msg = "invalid coded spectral envelope for MCD"
        raise ValueError(msg)

    diff = ref_cep - deg_cep
    mcd_db = float(MCD_CONST * np.mean(np.linalg.norm(diff, axis=1)))

    f0_ref_aligned = f0_ref[:frame_count]
    f0_deg_aligned = f0_deg[:frame_count]
    voiced_overlap = (f0_ref_aligned > 0.0) & (f0_deg_aligned > 0.0)
    voiced_overlap_frames = int(np.sum(voiced_overlap))

    if voiced_overlap_frames == 0:
        f0_rmse_hz = float("nan")
    else:
        error = f0_ref_aligned[voiced_overlap] - f0_deg_aligned[voiced_overlap]
        f0_rmse_hz = float(np.sqrt(np.mean(error**2)))

    return {
        "mcd_db": mcd_db,
        "f0_rmse_hz": f0_rmse_hz,
        "f0_voiced_overlap_frames": voiced_overlap_frames,
        "frame_count": int(frame_count),
    }


def _safe_nanmean(values: list[float]) -> float:
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0:
        return float("nan")
    return float(np.nanmean(arr))


def _summarize_by_emotion(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(str(row["target_emotion"]), []).append(row)

    ordered_emotions = [emotion for emotion in JVNV_EMOTION_LABELS if emotion in grouped]
    ordered_emotions.extend(sorted(set(grouped).difference(ordered_emotions)))

    summaries: dict[str, dict[str, Any]] = {}
    for emotion in ordered_emotions:
        subset = grouped[emotion]
        pesq_values = [float(item["pesq"]) for item in subset]
        mcd_values = [float(item["mcd_db"]) for item in subset]
        f0_values = [float(item["f0_rmse_hz"]) for item in subset]
        summaries[emotion] = {
            "count": len(subset),
            "pesq_mean": _safe_nanmean(pesq_values),
            "mcd_mean_db": _safe_nanmean(mcd_values),
            "f0_rmse_mean_hz": _safe_nanmean(f0_values),
            "f0_valid_count": int(np.isfinite(np.asarray(f0_values, dtype=np.float64)).sum()),
        }
    return summaries


def _judge_go_no_go(
    *,
    pesq_mean: float,
    mcd_mean_db: float,
    f0_rmse_mean_hz: float,
) -> dict[str, Any]:
    checks: dict[str, dict[str, Any]] = {}
    failure_reasons: list[str] = []

    def _record_check(name: str, value: float, threshold: float, operator: str) -> bool:
        if not np.isfinite(value):
            checks[name] = {
                "value": value,
                "threshold": threshold,
                "operator": operator,
                "pass": False,
            }
            failure_reasons.append(f"{name} mean is not finite")
            return False

        if operator == ">=":
            ok = value >= threshold
            checks[name] = {
                "value": value,
                "threshold": threshold,
                "operator": operator,
                "pass": ok,
            }
            if not ok:
                failure_reasons.append(f"{name} mean {value:.4f} < {threshold:.4f}")
            return ok

        ok = value <= threshold
        checks[name] = {
            "value": value,
            "threshold": threshold,
            "operator": operator,
            "pass": ok,
        }
        if not ok:
            failure_reasons.append(f"{name} mean {value:.4f} > {threshold:.4f}")
        return ok

    gate_pesq = _record_check("PESQ", pesq_mean, GO_THRESHOLDS["pesq_min"], ">=")
    gate_mcd = _record_check("MCD(dB)", mcd_mean_db, GO_THRESHOLDS["mcd_max_db"], "<=")
    gate_f0 = _record_check("F0 RMSE(Hz)", f0_rmse_mean_hz, GO_THRESHOLDS["f0_rmse_max_hz"], "<=")

    gate_pass = bool(gate_pesq and gate_mcd and gate_f0)
    return {
        "label": "Go" if gate_pass else "No-Go",
        "pass": gate_pass,
        "checks": checks,
        "failure_reasons": failure_reasons,
    }


def _write_per_file_csv(rows: list[dict[str, Any]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "target_emotion",
        "text",
        "baseline_audio_path",
        "candidate_audio_path",
        "sample_rate_hz",
        "num_samples_aligned",
        "pesq",
        "mcd_db",
        "f0_rmse_hz",
        "f0_voiced_overlap_frames",
        "frame_count",
    ]
    with output_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({name: row.get(name) for name in fieldnames})


def _build_markdown_report(result: dict[str, Any]) -> str:
    overall = result["overall"]
    decision_detail = result["decision_detail"]
    by_emotion = result["by_emotion"]

    lines = [
        "# Roundtrip Quantitative Evaluation",
        "",
        "## Summary",
        f"- Baseline manifest: {result['inputs']['baseline_manifest']}",
        f"- Candidate manifest: {result['inputs']['candidate_manifest']}",
        f"- Num files: {overall['num_files']}",
        f"- PESQ mean: {overall['pesq_mean']:.4f}",
        f"- MCD mean: {overall['mcd_mean_db']:.4f} dB",
        f"- F0 RMSE mean: {overall['f0_rmse_mean_hz']:.4f} Hz",
        f"- Subjective gate status: {result['subjective_gate_status']}",
        f"- Decision: **{result['decision']}**",
        "",
        "## Go/No-Go Checks",
        "| Metric | Value | Threshold | Operator | Pass |",
        "|---|---:|---:|:---:|:---:|",
    ]

    checks = decision_detail["checks"]
    lines.extend(
        (
            f"| PESQ | {checks['PESQ']['value']:.4f} | {checks['PESQ']['threshold']:.4f} | >= | {'Yes' if checks['PESQ']['pass'] else 'No'} |",
            f"| MCD(dB) | {checks['MCD(dB)']['value']:.4f} | {checks['MCD(dB)']['threshold']:.4f} | <= | {'Yes' if checks['MCD(dB)']['pass'] else 'No'} |",
            f"| F0 RMSE(Hz) | {checks['F0 RMSE(Hz)']['value']:.4f} | {checks['F0 RMSE(Hz)']['threshold']:.4f} | <= | {'Yes' if checks['F0 RMSE(Hz)']['pass'] else 'No'} |",
        )
    )

    lines.extend(
        [
            "",
            "## By Emotion",
            "| Emotion | Count | PESQ mean | MCD mean (dB) | F0 RMSE mean (Hz) | F0 valid count |",
            "|---|---:|---:|---:|---:|---:|",
        ],
    )
    for emotion, summary in by_emotion.items():
        lines.append(
            f"| {emotion} | {summary['count']} | "
            f"{summary['pesq_mean']:.4f} | {summary['mcd_mean_db']:.4f} | "
            f"{summary['f0_rmse_mean_hz']:.4f} | {summary['f0_valid_count']} |",
        )

    failure_reasons = decision_detail["failure_reasons"]
    if failure_reasons:
        lines.extend(["", "## Failure Reasons"])
        lines.extend(f"- {reason}" for reason in failure_reasons)

    if result["decision"] == "No-Go":
        lines.extend(["", "## Recommended Next Steps (No-Go)"])
        for idx, action in enumerate(result["no_go_recommended_actions"], start=1):
            lines.append(f"{idx}. {action}")

    return "\n".join(lines)


def run_evaluation(
    *,
    baseline_manifest: str | Path,
    candidate_manifest: str | Path,
    output_dir: str | Path,
) -> dict[str, Any]:
    baseline_entries = _load_manifest_entries(baseline_manifest)
    candidate_entries = _load_manifest_entries(candidate_manifest)
    pairs = _pair_manifest_entries(baseline_entries, candidate_entries)

    rows: list[dict[str, Any]] = []
    total = len(pairs)

    for idx, (baseline_item, candidate_item) in enumerate(pairs, start=1):
        try:
            baseline_wav, baseline_sr = _read_audio_mono(baseline_item.audio_path)
            candidate_wav, candidate_sr = _read_audio_mono(candidate_item.audio_path)
            ref, deg, sr = _align_pair(
                baseline_wav,
                baseline_sr,
                candidate_wav,
                candidate_sr,
            )
            pesq_score = _compute_pesq_wb(ref, deg, sr)
            world_metrics = _compute_world_metrics(ref, deg, sr)
        except Exception as exc:
            msg = (
                "failed to evaluate pair "
                f"(target_emotion={baseline_item.target_emotion}, text={baseline_item.text}): {exc}"
            )
            raise RuntimeError(msg) from exc

        row = {
            "target_emotion": baseline_item.target_emotion,
            "text": baseline_item.text,
            "baseline_audio_path": str(baseline_item.audio_path),
            "candidate_audio_path": str(candidate_item.audio_path),
            "sample_rate_hz": sr,
            "num_samples_aligned": int(ref.size),
            "pesq": pesq_score,
            "mcd_db": float(world_metrics["mcd_db"]),
            "f0_rmse_hz": float(world_metrics["f0_rmse_hz"]),
            "f0_voiced_overlap_frames": int(world_metrics["f0_voiced_overlap_frames"]),
            "frame_count": int(world_metrics["frame_count"]),
        }
        rows.append(row)
        logger.info(
            "[%d/%d] %s | PESQ=%.4f, MCD=%.4f dB, F0 RMSE=%.4f Hz",
            idx,
            total,
            baseline_item.target_emotion,
            row["pesq"],
            row["mcd_db"],
            row["f0_rmse_hz"],
        )

    pesq_values = [float(row["pesq"]) for row in rows]
    mcd_values = [float(row["mcd_db"]) for row in rows]
    f0_values = [float(row["f0_rmse_hz"]) for row in rows]
    overall = {
        "num_files": len(rows),
        "pesq_mean": _safe_nanmean(pesq_values),
        "mcd_mean_db": _safe_nanmean(mcd_values),
        "f0_rmse_mean_hz": _safe_nanmean(f0_values),
        "f0_valid_count": int(np.isfinite(np.asarray(f0_values, dtype=np.float64)).sum()),
    }
    decision = _judge_go_no_go(
        pesq_mean=float(overall["pesq_mean"]),
        mcd_mean_db=float(overall["mcd_mean_db"]),
        f0_rmse_mean_hz=float(overall["f0_rmse_mean_hz"]),
    )

    result = {
        "inputs": {
            "baseline_manifest": str(Path(baseline_manifest)),
            "candidate_manifest": str(Path(candidate_manifest)),
        },
        "thresholds": GO_THRESHOLDS,
        "subjective_gate_status": SUBJECTIVE_GATE_STATUS,
        "overall": overall,
        "by_emotion": _summarize_by_emotion(rows),
        "per_file": rows,
        "pesq_mean": float(overall["pesq_mean"]),
        "mcd_mean_db": float(overall["mcd_mean_db"]),
        "f0_rmse_mean_hz": float(overall["f0_rmse_mean_hz"]),
        "decision": str(decision["label"]),
        "decision_detail": decision,
        "no_go_recommended_actions": NO_GO_RECOMMENDED_ACTIONS
        if decision["label"] == "No-Go"
        else [],
    }

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    save_json(result, out_dir / "roundtrip_metrics.json")
    save_markdown(_build_markdown_report(result), out_dir / "roundtrip_report.md")
    _write_per_file_csv(rows, out_dir / "per_file_metrics.csv")

    return result


def main() -> None:
    _configure_logging()
    args = _build_parser().parse_args()

    result = run_evaluation(
        baseline_manifest=args.baseline_manifest,
        candidate_manifest=args.candidate_manifest,
        output_dir=args.output_dir,
    )
    logger.info(
        "Roundtrip evaluation completed: decision=%s, PESQ=%.4f, MCD=%.4f, F0 RMSE=%.4f",
        result["decision"],
        result["pesq_mean"],
        result["mcd_mean_db"],
        result["f0_rmse_mean_hz"],
    )


if __name__ == "__main__":
    main()
