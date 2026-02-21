from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

EVALUATION_SCHEMA_VERSION = "2.0"


def build_evaluation_manifest(
    *,
    task: str,
    gate: dict[str, Any],
    summary: dict[str, Any],
    inputs: dict[str, Any] | None = None,
    artifacts: dict[str, Any] | None = None,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "schema_version": EVALUATION_SCHEMA_VERSION,
        "task": task,
        "generated_at": datetime.now(tz=timezone.utc).isoformat(),
        "gate": gate,
        "summary": summary,
        "inputs": inputs or {},
        "artifacts": artifacts or {},
        "metadata": metadata or {},
    }


def write_evaluation_manifest(
    manifest: dict[str, Any],
    output_path: str | Path,
) -> Path:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(manifest, file, ensure_ascii=False, indent=2)
    return path
