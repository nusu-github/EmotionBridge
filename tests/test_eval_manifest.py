from __future__ import annotations

import json
from datetime import datetime

from emotionbridge.eval.manifest import (
    EVALUATION_SCHEMA_VERSION,
    build_evaluation_manifest,
    write_evaluation_manifest,
)


def test_build_evaluation_manifest_fields() -> None:
    manifest = build_evaluation_manifest(
        task="v02_responsiveness",
        gate={"label": "Go", "pass": True, "checks": {}, "failure_reasons": []},
        summary={"num_rows": 10, "num_features": 88},
        inputs={"input_path": "in.parquet"},
        artifacts={"metrics_json": "metrics.json"},
        metadata={"gate_policy": "feature_only"},
    )

    assert manifest["schema_version"] == EVALUATION_SCHEMA_VERSION
    assert manifest["task"] == "v02_responsiveness"
    assert manifest["gate"]["label"] == "Go"
    assert manifest["summary"]["num_rows"] == 10
    assert manifest["inputs"]["input_path"] == "in.parquet"
    assert manifest["artifacts"]["metrics_json"] == "metrics.json"
    assert manifest["metadata"]["gate_policy"] == "feature_only"
    assert datetime.fromisoformat(manifest["generated_at"])


def test_write_evaluation_manifest_creates_file(tmp_path) -> None:
    manifest = build_evaluation_manifest(
        task="v01_separation",
        gate={"label": "No-Go", "pass": False, "checks": {}, "failure_reasons": ["x"]},
        summary={"num_rows": 12},
    )

    output_path = tmp_path / "nested" / "manifest.json"
    written_path = write_evaluation_manifest(manifest, output_path)

    assert written_path == output_path
    assert output_path.exists()

    reloaded = json.loads(output_path.read_text(encoding="utf-8"))
    assert reloaded == manifest
