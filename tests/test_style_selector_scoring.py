from __future__ import annotations

import json
from typing import TYPE_CHECKING

import pytest

from emotionbridge.constants import CONTROL_PARAM_NAMES, JVNV_EMOTION_LABELS
from emotionbridge.inference.bridge_pipeline import RuleBasedStyleSelector

if TYPE_CHECKING:
    from pathlib import Path


def _write_json(path: Path, payload: dict) -> Path:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def _base_mapping_payload(
    *,
    styles_used: list[int],
    selection_policy: str,
    scoring_lambda: float,
    artifacts: dict[str, str | None],
) -> dict:
    return {
        "version": "2.0",
        "created_at": "2026-02-21T00:00:00+00:00",
        "selected_character": "zundamon",
        "selection_policy": selection_policy,
        "scoring": {
            "base_term": "sum_p_emotion_negative_distance",
            "compat_term": (
                "negative_l2_to_style_control_prediction"
                if selection_policy == "prob_distance_with_control_compat"
                else "disabled"
            ),
            "normalization": "zscore_across_styles",
            "lambda": scoring_lambda,
            "lambda_source": (
                "style_control_compatibility"
                if selection_policy == "prob_distance_with_control_compat"
                else "disabled"
            ),
        },
        "styles_used": styles_used,
        "style_signal_status": "retain_style",
        "profile_source": "voicevox_egemaps_raw(global_zscore_style_signal)",
        "profile_source_reason": "style_signal_status=retain_style",
        "characters": {
            "zundamon": {
                "mapping": {
                    "anger": {"style_id": styles_used[0], "style_name": f"style_{styles_used[0]}"},
                },
                "default_style": {
                    "style_id": styles_used[0],
                    "style_name": f"style_{styles_used[0]}",
                },
            },
        },
        "artifacts": artifacts,
    }


def _distance_payload(styles: list[int], rows: dict[str, dict[str, float]]) -> dict:
    distance_matrix = {}
    for emotion in JVNV_EMOTION_LABELS:
        distance_matrix[emotion] = rows.get(
            emotion,
            {str(style_id): 2.0 for style_id in styles},
        )
    return {
        "version": "1.0",
        "distance_matrix": distance_matrix,
    }


def _style_profiles_payload(styles: list[int]) -> dict:
    return {
        "version": "1.0",
        "styles": {
            str(style_id): {
                "style_id": style_id,
                "character": "zundamon",
                "style_name": f"style_{style_id}",
            }
            for style_id in styles
        },
    }


def test_select_uses_full_emotion_distribution(tmp_path: Path) -> None:
    styles = [1, 3]
    distance = _distance_payload(
        styles,
        {
            "anger": {"1": 1.0, "3": 2.0},
            "happy": {"1": 10.0, "3": 0.0},
        },
    )
    distance_path = _write_json(tmp_path / "distance.json", distance)
    profiles_path = _write_json(tmp_path / "profiles.json", _style_profiles_payload(styles))
    mapping = _base_mapping_payload(
        styles_used=styles,
        selection_policy="prob_distance_only",
        scoring_lambda=0.0,
        artifacts={
            "distance_matrix": str(distance_path),
            "style_profiles": str(profiles_path),
            "distance_matrix_table": str(tmp_path / "distance.parquet"),
            "style_signal_metrics": str(tmp_path / "style_influence_metrics.json"),
            "style_control_compatibility": None,
        },
    )
    mapping_path = _write_json(tmp_path / "style_mapping.json", mapping)

    selector = RuleBasedStyleSelector(mapping_path)
    style_id, style_name = selector.select(
        {"anger": 0.51, "happy": 0.49},
        "zundamon",
    )
    assert style_id == 3
    assert style_name == "style_3"


def test_select_can_flip_rank_with_compatibility_term(tmp_path: Path) -> None:
    styles = [1, 3, 5]
    distance = _distance_payload(
        styles,
        {
            "anger": {"1": 1.0, "3": 1.1, "5": 3.0},
        },
    )
    distance_path = _write_json(tmp_path / "distance.json", distance)
    profiles_path = _write_json(tmp_path / "profiles.json", _style_profiles_payload(styles))
    compatibility = {
        "version": "1.0",
        "style_ids": styles,
        "feature_names": ["egemaps__feat0"],
        "control_names": CONTROL_PARAM_NAMES,
        "selection_metric": {
            "best_lambda": 0.5,
        },
        "jvnv_emotion_centroids": {
            emotion: {"egemaps__feat0": 0.0} for emotion in JVNV_EMOTION_LABELS
        },
        "style_models": {
            "1": {"coef": [[0.0, 0.0, 0.0, 0.0, 0.0]], "intercept": [5.0]},
            "3": {"coef": [[0.0, 0.0, 0.0, 0.0, 0.0]], "intercept": [1.0]},
            "5": {"coef": [[0.0, 0.0, 0.0, 0.0, 0.0]], "intercept": [3.0]},
        },
    }
    compatibility_path = _write_json(tmp_path / "compatibility.json", compatibility)
    mapping = _base_mapping_payload(
        styles_used=styles,
        selection_policy="prob_distance_with_control_compat",
        scoring_lambda=0.5,
        artifacts={
            "distance_matrix": str(distance_path),
            "style_profiles": str(profiles_path),
            "distance_matrix_table": str(tmp_path / "distance.parquet"),
            "style_signal_metrics": str(tmp_path / "style_influence_metrics.json"),
            "style_control_compatibility": str(compatibility_path),
        },
    )
    mapping_path = _write_json(tmp_path / "style_mapping.json", mapping)

    selector = RuleBasedStyleSelector(mapping_path)
    style_id, _ = selector.select(
        {"anger": 1.0},
        "zundamon",
        control_params={
            "pitch_shift": 0.0,
            "pitch_range": 0.0,
            "speed": 0.0,
            "energy": 0.0,
            "pause_weight": 0.0,
        },
    )
    assert style_id == 3


def test_select_tie_breaks_by_smallest_style_id(tmp_path: Path) -> None:
    styles = [1, 3]
    distance = _distance_payload(
        styles,
        {
            "anger": {"1": 1.0, "3": 1.0},
            "happy": {"1": 1.0, "3": 1.0},
        },
    )
    distance_path = _write_json(tmp_path / "distance.json", distance)
    profiles_path = _write_json(tmp_path / "profiles.json", _style_profiles_payload(styles))
    mapping = _base_mapping_payload(
        styles_used=styles,
        selection_policy="prob_distance_only",
        scoring_lambda=0.0,
        artifacts={
            "distance_matrix": str(distance_path),
            "style_profiles": str(profiles_path),
            "distance_matrix_table": str(tmp_path / "distance.parquet"),
            "style_signal_metrics": str(tmp_path / "style_influence_metrics.json"),
            "style_control_compatibility": None,
        },
    )
    mapping_path = _write_json(tmp_path / "style_mapping.json", mapping)

    selector = RuleBasedStyleSelector(mapping_path)
    style_id, _ = selector.select({"anger": 1.0}, "zundamon")
    assert style_id == 1


def test_selector_rejects_legacy_mapping_schema(tmp_path: Path) -> None:
    legacy_path = _write_json(
        tmp_path / "legacy_style_mapping.json",
        {
            "version": "1.0",
            "characters": {
                "zundamon": {
                    "mapping": {
                        "anger": {"style_id": 3, "style_name": "ノーマル"},
                    },
                    "default_style": {"style_id": 3, "style_name": "ノーマル"},
                },
            },
        },
    )
    with pytest.raises(ValueError, match="schema version"):
        RuleBasedStyleSelector(legacy_path)
