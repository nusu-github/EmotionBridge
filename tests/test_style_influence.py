from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from emotionbridge.scripts.evaluate_style_influence import run_evaluation

CONTROL_COLUMNS = [
    "ctrl_pitch_shift",
    "ctrl_pitch_range",
    "ctrl_speed",
    "ctrl_energy",
    "ctrl_pause_weight",
]
FEATURE_COLUMNS = [
    "egemaps__feat0",
    "egemaps__feat1",
    "egemaps__feat2",
    "egemaps__feat3",
]


def _write_config(path: Path, v01_dir: Path, v02_dir: Path, v03_dir: Path) -> None:
    path.write_text(
        "\n".join(
            [
                "v01:",
                f"  output_dir: {v01_dir}",
                "v02:",
                f"  output_dir: {v02_dir}",
                "v03:",
                f"  output_dir: {v03_dir}",
                "  style_effect_min_partial_eta2: 0.01",
                "  style_effect_min_feature_ratio: 0.15",
                "  style_interaction_min_partial_eta2: 0.01",
                "  style_interaction_min_feature_ratio: 0.15",
            ],
        ),
        encoding="utf-8",
    )


def _make_jvnv(feature_cols: list[str]) -> pd.DataFrame:
    emotions = ["anger", "disgust", "fear", "happy", "sad", "surprise"]
    rows: list[dict[str, float | str]] = []
    for idx, emotion in enumerate(emotions):
        row: dict[str, float | str] = {"emotion": emotion}
        for f_idx, feature in enumerate(feature_cols):
            row[feature] = float(idx + f_idx)
        rows.append(row)
    return pd.DataFrame(rows)


def _make_raw_dataset(
    *,
    rng: np.random.Generator,
    with_style_main: bool = False,
    with_interaction: bool = False,
    noise_scale: float = 0.02,
    n_per_style: int = 200,
) -> pd.DataFrame:
    styles = [1, 3]
    controls_template = rng.uniform(-1.0, 1.0, size=(n_per_style, len(CONTROL_COLUMNS)))
    rows: list[dict[str, float | int]] = []
    for style in styles:
        controls = controls_template.copy()
        base = (
            0.8 * controls[:, 0]
            - 0.3 * controls[:, 1]
            + 0.4 * controls[:, 2]
            + 0.2 * controls[:, 3]
            - 0.1 * controls[:, 4]
        )
        for i in range(n_per_style):
            row: dict[str, float | int] = {"style_id": style}
            for c_idx, control in enumerate(CONTROL_COLUMNS):
                row[control] = float(controls[i, c_idx])

            y0 = base[i] + rng.normal(0.0, noise_scale)
            y1 = 0.5 * base[i] + rng.normal(0.0, noise_scale)
            y2 = -0.2 * base[i] + rng.normal(0.0, noise_scale)
            y3 = 0.1 * base[i] + rng.normal(0.0, noise_scale)

            if with_style_main:
                y0 += 1.2 if style == 3 else -1.2
            if with_interaction:
                y1 += (1.5 if style == 3 else -1.5) * controls[i, 0]

            row[FEATURE_COLUMNS[0]] = float(y0)
            row[FEATURE_COLUMNS[1]] = float(y1)
            row[FEATURE_COLUMNS[2]] = float(y2)
            row[FEATURE_COLUMNS[3]] = float(y3)
            rows.append(row)
    return pd.DataFrame(rows)


def _run(
    tmp_path: Path,
    raw_df: pd.DataFrame,
) -> dict:
    v01_dir = tmp_path / "v01"
    v02_dir = tmp_path / "v02"
    v03_dir = tmp_path / "v03"
    v01_dir.mkdir(parents=True, exist_ok=True)
    v02_dir.mkdir(parents=True, exist_ok=True)
    v03_dir.mkdir(parents=True, exist_ok=True)

    raw_path = v02_dir / "voicevox_egemaps_raw.parquet"
    jvnv_path = v01_dir / "jvnv_egemaps_normalized.parquet"
    config_path = tmp_path / "config.yaml"

    raw_df.to_parquet(raw_path, index=False)
    _make_jvnv(FEATURE_COLUMNS).to_parquet(jvnv_path, index=False)
    _write_config(config_path, v01_dir, v02_dir, v03_dir)

    return run_evaluation(
        config_path=str(config_path),
        voicevox_raw=str(raw_path),
        jvnv_normalized=str(jvnv_path),
        target_style_ids_raw=None,
        output_dir=str(v03_dir),
    )


def test_style_influence_deprioritize_when_no_style_effect(tmp_path: Path) -> None:
    rng = np.random.default_rng(0)
    raw_df = _make_raw_dataset(
        rng=rng,
        with_style_main=False,
        with_interaction=False,
        noise_scale=0.0,
    )
    result = _run(tmp_path, raw_df)

    assert result["style_signal_status"] == "deprioritize_style"
    assert Path(result["feature_table_path"]).exists()
    assert Path(tmp_path / "v03" / "style_influence_metrics.json").exists()
    assert Path(tmp_path / "v03" / "style_influence_report.md").exists()


def test_style_influence_retain_when_style_main_effect_exists(tmp_path: Path) -> None:
    rng = np.random.default_rng(1)
    raw_df = _make_raw_dataset(rng=rng, with_style_main=True, with_interaction=False)
    result = _run(tmp_path, raw_df)

    assert result["style_signal_status"] == "retain_style"
    assert result["style_main_feature_ratio"] >= 0.15


def test_style_influence_retain_when_interaction_effect_exists(tmp_path: Path) -> None:
    rng = np.random.default_rng(2)
    raw_df = _make_raw_dataset(rng=rng, with_style_main=False, with_interaction=True)
    result = _run(tmp_path, raw_df)

    assert result["style_signal_status"] == "retain_style"
    assert result["interaction_feature_ratio"] >= 0.15
