from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from emotionbridge.scripts.build_style_mapping import run_build_style_mapping

FEATURES = ["egemaps__feat0", "egemaps__feat1"]


def _write_config(path: Path, v01_dir: Path, v02_dir: Path, v03_dir: Path, out_root: Path) -> None:
    path.write_text(
        "\n".join(
            [
                "paths:",
                f"  output_root: {out_root}",
                "v01:",
                f"  output_dir: {v01_dir}",
                "v02:",
                f"  output_dir: {v02_dir}",
                "v03:",
                f"  output_dir: {v03_dir}",
                "  style_centroid_degeneracy_threshold: 1.0e-6",
            ],
        ),
        encoding="utf-8",
    )


def _write_inputs(base: Path) -> tuple[Path, Path, Path, Path]:
    v01_dir = base / "v01"
    v02_dir = base / "v02"
    v03_dir = base / "v03"
    out_root = base / "out"
    v01_dir.mkdir(parents=True, exist_ok=True)
    v02_dir.mkdir(parents=True, exist_ok=True)
    v03_dir.mkdir(parents=True, exist_ok=True)
    out_root.mkdir(parents=True, exist_ok=True)

    jvnv_rows = []
    for idx, emotion in enumerate(["anger", "disgust", "fear", "happy", "sad", "surprise"]):
        jvnv_rows.append(
            {
                "emotion": emotion,
                "egemaps__feat0": float(idx),
                "egemaps__feat1": float(idx) * 0.5,
            },
        )
    pd.DataFrame(jvnv_rows).to_parquet(v01_dir / "jvnv_egemaps_normalized.parquet", index=False)

    norm_rows = []
    for style in [1, 3]:
        norm_rows.extend(
            [
                {
                    "style_id": style,
                    "egemaps__feat0": 0.0,
                    "egemaps__feat1": 0.0,
                }
                for _ in range(8)
            ],
        )
    pd.DataFrame(norm_rows).to_parquet(v02_dir / "voicevox_egemaps_normalized.parquet", index=False)

    raw_rows = []
    for style in [1, 3]:
        for idx in range(8):
            anger_prob = 0.7 if idx % 2 == 0 else 0.1
            happy_prob = 0.2 if idx % 2 == 0 else 0.8
            remain = 1.0 - anger_prob - happy_prob
            raw_rows.append(
                {
                    "style_id": style,
                    "ctrl_pitch_shift": -0.8 + 0.2 * idx,
                    "ctrl_pitch_range": -0.4 + 0.1 * idx,
                    "ctrl_speed": -0.2 + 0.1 * idx,
                    "ctrl_energy": -0.3 + 0.15 * idx,
                    "ctrl_pause_weight": 0.1 * (idx - 4),
                    "emotion_anger": anger_prob,
                    "emotion_disgust": remain / 4.0,
                    "emotion_fear": remain / 4.0,
                    "emotion_happy": happy_prob,
                    "emotion_sad": remain / 4.0,
                    "emotion_surprise": remain / 4.0,
                    "egemaps__feat0": 1.0 if style == 1 else -1.0,
                    "egemaps__feat1": 0.2 if style == 1 else -0.2,
                },
            )
    pd.DataFrame(raw_rows).to_parquet(v02_dir / "voicevox_egemaps_raw.parquet", index=False)
    return v01_dir, v02_dir, v03_dir, out_root


def test_build_style_mapping_uses_style_signal_retain(tmp_path: Path) -> None:
    v01_dir, v02_dir, v03_dir, out_root = _write_inputs(tmp_path)
    config_path = tmp_path / "config.yaml"
    _write_config(config_path, v01_dir, v02_dir, v03_dir, out_root)

    metrics_payload = {
        "style_signal_status": "retain_style",
        "styles_used": [1, 3],
    }
    (v03_dir / "style_influence_metrics.json").write_text(
        json.dumps(metrics_payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    result = run_build_style_mapping(
        config_path=str(config_path),
        jvnv_normalized=None,
        voicevox_normalized=None,
        target_style_ids_raw="1,3",
        output_dir=None,
    )
    payload = json.loads(Path(result["distance_matrix"]).read_text(encoding="utf-8"))
    mapping = json.loads(Path(result["style_mapping"]).read_text(encoding="utf-8"))

    assert payload["style_signal_status"] == "retain_style"
    assert "global_zscore_style_signal" in payload["profile_source"]
    assert payload["style_signal_metrics_path"] == str(v03_dir / "style_influence_metrics.json")
    assert mapping["style_signal_status"] == "retain_style"
    assert mapping["version"] == "2.0"
    assert mapping["selection_policy"] == "prob_distance_with_control_compat"
    assert isinstance(mapping["scoring"]["lambda"], float)
    compat_path = mapping["artifacts"]["style_control_compatibility"]
    assert isinstance(compat_path, str)
    compat_payload = json.loads(Path(compat_path).read_text(encoding="utf-8"))
    assert compat_payload["selection_metric"]["objective"] == "mean_expected_distance"
    assert isinstance(compat_payload["selection_metric"]["best_lambda"], float)


def test_build_style_mapping_sets_unavailable_when_metrics_missing(tmp_path: Path) -> None:
    v01_dir, v02_dir, v03_dir, out_root = _write_inputs(tmp_path)
    config_path = tmp_path / "config.yaml"
    _write_config(config_path, v01_dir, v02_dir, v03_dir, out_root)

    result = run_build_style_mapping(
        config_path=str(config_path),
        jvnv_normalized=None,
        voicevox_normalized=None,
        target_style_ids_raw="1,3",
        output_dir=None,
    )
    payload = json.loads(Path(result["distance_matrix"]).read_text(encoding="utf-8"))
    mapping = json.loads(Path(result["style_mapping"]).read_text(encoding="utf-8"))

    assert payload["style_signal_status"] == "unavailable"
    assert payload["style_signal_metrics_path"] is None
    assert "metrics file not found" in payload["style_signal_info"]
    assert mapping["style_signal_status"] == "unavailable"
    assert mapping["version"] == "2.0"
    assert mapping["selection_policy"] == "prob_distance_only"
    assert mapping["artifacts"]["style_control_compatibility"] is None
