import argparse
import json
import logging
import shutil
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from emotionbridge.constants import JVNV_EMOTION_LABELS
from emotionbridge.scripts.common import save_json

logger = logging.getLogger(__name__)


VOICEVOX_STYLE_METADATA: dict[int, dict[str, str]] = {
    3: {"character": "zundamon", "style_name": "ノーマル"},
    1: {"character": "zundamon", "style_name": "あまあま"},
    7: {"character": "zundamon", "style_name": "ツンツン"},
    5: {"character": "zundamon", "style_name": "セクシー"},
    22: {"character": "zundamon", "style_name": "ささやき"},
    38: {"character": "zundamon", "style_name": "ヒソヒソ"},
    75: {"character": "zundamon", "style_name": "ヘロヘロ"},
    76: {"character": "zundamon", "style_name": "なみだめ"},
    2: {"character": "shikoku_metan", "style_name": "ノーマル"},
    0: {"character": "shikoku_metan", "style_name": "あまあま"},
    6: {"character": "shikoku_metan", "style_name": "ツンツン"},
    4: {"character": "shikoku_metan", "style_name": "セクシー"},
    36: {"character": "shikoku_metan", "style_name": "ささやき"},
    37: {"character": "shikoku_metan", "style_name": "ヒソヒソ"},
}


def _configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="style_mapping.json を手動で調整する",
    )
    parser.add_argument(
        "--mapping-path",
        default="artifacts/prosody/style_mapping.json",
        help="入力 style_mapping.json",
    )
    parser.add_argument(
        "--output-path",
        default=None,
        help="出力先（未指定時は mapping-path を上書き）",
    )
    parser.add_argument(
        "--character",
        default=None,
        help="調整対象キャラクター（未指定時は selected_character を使用）",
    )
    parser.add_argument(
        "--set",
        action="append",
        default=[],
        help="emotion=style_id（カンマ区切り可、複数回指定可）",
    )
    parser.add_argument(
        "--default-style-id",
        type=int,
        default=None,
        help="default_style の style_id を上書き",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="書き込みせず変更内容だけ表示",
    )
    parser.add_argument(
        "--no-backup",
        action="store_true",
        help="上書き時のバックアップ作成を無効化",
    )
    return parser


def _parse_set_args(raw_values: list[str]) -> dict[str, int]:
    assignments: dict[str, int] = {}

    for raw in raw_values:
        chunks = [chunk.strip() for chunk in raw.split(",") if chunk.strip()]
        for chunk in chunks:
            if "=" not in chunk:
                msg = f"Invalid --set format: '{chunk}'. expected emotion=style_id"
                raise ValueError(msg)

            emotion_raw, style_raw = chunk.split("=", 1)
            emotion = emotion_raw.strip().lower()
            if emotion not in JVNV_EMOTION_LABELS:
                msg = f"Unknown emotion label: '{emotion}'. valid={JVNV_EMOTION_LABELS}"
                raise ValueError(msg)

            try:
                style_id = int(style_raw.strip())
            except ValueError as error:
                msg = f"style_id must be int: '{style_raw}'"
                raise ValueError(msg) from error

            assignments[emotion] = style_id

    return assignments


def _resolve_character(payload: dict[str, Any], requested: str | None) -> str:
    characters = payload.get("characters")
    if not isinstance(characters, dict) or not characters:
        msg = "Invalid style mapping: 'characters' must be a non-empty dict"
        raise ValueError(msg)

    if requested is not None:
        if requested not in characters:
            msg = f"character '{requested}' is not found in mapping"
            raise KeyError(msg)
        return requested

    selected = payload.get("selected_character")
    if isinstance(selected, str) and selected in characters:
        return selected

    if len(characters) == 1:
        return next(iter(characters.keys()))

    msg = "Multiple characters exist. specify --character"
    raise ValueError(msg)


def _resolve_style_name(
    *,
    style_id: int,
    character: str,
    mapping: dict[str, Any],
) -> str:
    metadata = VOICEVOX_STYLE_METADATA.get(style_id)
    if metadata is not None:
        if metadata["character"] != character:
            msg = f"style_id={style_id} belongs to '{metadata['character']}', not '{character}'"
            raise ValueError(msg)
        return metadata["style_name"]

    for payload in mapping.values():
        if not isinstance(payload, dict):
            continue
        existing_style_id = payload.get("style_id")
        if not isinstance(existing_style_id, int):
            continue
        if existing_style_id != style_id:
            continue
        style_name = payload.get("style_name")
        if isinstance(style_name, str) and style_name:
            return style_name

    return f"style_{style_id}"


def _load_distance_matrix(
    payload: dict[str, Any],
    *,
    mapping_path: Path,
) -> dict[str, dict[str, float]]:
    artifacts = payload.get("artifacts")
    if not isinstance(artifacts, dict):
        return {}

    distance_path_raw = artifacts.get("distance_matrix")
    if not isinstance(distance_path_raw, str) or not distance_path_raw:
        return {}

    distance_path = Path(distance_path_raw)
    if not distance_path.is_absolute():
        distance_path = (mapping_path.parent / distance_path).resolve()

    if not distance_path.exists():
        logger.warning("distance_matrix not found: %s", distance_path)
        return {}

    try:
        payload = json.loads(distance_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        logger.warning("distance_matrix is not valid JSON: %s", distance_path)
        return {}

    matrix = payload.get("distance_matrix")
    if not isinstance(matrix, dict):
        return {}

    normalized: dict[str, dict[str, float]] = {}
    for emotion, row in matrix.items():
        if not isinstance(row, dict):
            continue
        casted: dict[str, float] = {}
        for style_id, value in row.items():
            if isinstance(value, (int, float)):
                casted[str(style_id)] = float(value)
        normalized[str(emotion)] = casted

    return normalized


def run_adjust_style_mapping(
    *,
    mapping_path: str,
    output_path: str | None,
    character: str | None,
    set_args: list[str],
    default_style_id: int | None,
    dry_run: bool,
    no_backup: bool,
) -> dict[str, Any]:
    in_path = Path(mapping_path)
    if not in_path.exists():
        msg = f"style mapping not found: {in_path}"
        raise FileNotFoundError(msg)

    payload = json.loads(in_path.read_text(encoding="utf-8"))
    selected_character = _resolve_character(payload, requested=character)

    characters = payload.get("characters")
    if not isinstance(characters, dict):
        msg = "Invalid style mapping: 'characters' must be dict"
        raise ValueError(msg)

    character_payload = characters.get(selected_character)
    if not isinstance(character_payload, dict):
        msg = f"character payload is invalid: {selected_character}"
        raise ValueError(msg)

    mapping = character_payload.get("mapping")
    if not isinstance(mapping, dict):
        mapping = {}
        character_payload["mapping"] = mapping

    assignments = _parse_set_args(set_args)
    if not assignments and default_style_id is None:
        msg = "No changes requested. specify --set and/or --default-style-id"
        raise ValueError(msg)

    distance_matrix = _load_distance_matrix(payload, mapping_path=in_path)

    styles_used = payload.get("styles_used")
    if not isinstance(styles_used, list):
        styles_used = []
    used_ids = {value for value in styles_used if isinstance(value, int)}

    changes: list[dict[str, Any]] = []
    for emotion, style_id in assignments.items():
        old = mapping.get(emotion)
        old_style_id = old.get("style_id") if isinstance(old, dict) else None
        old_style_name = old.get("style_name") if isinstance(old, dict) else None

        style_name = _resolve_style_name(
            style_id=style_id,
            character=selected_character,
            mapping=mapping,
        )

        updated = dict(old) if isinstance(old, dict) else {}
        updated["style_id"] = style_id
        updated["style_name"] = style_name
        updated["character"] = selected_character

        row = distance_matrix.get(emotion, {})
        distance = row.get(str(style_id)) if isinstance(row, dict) else None
        if isinstance(distance, (int, float)):
            updated["distance"] = float(distance)
        else:
            updated.pop("distance", None)

        mapping[emotion] = updated
        used_ids.add(style_id)

        changes.append(
            {
                "type": "mapping",
                "emotion": emotion,
                "old_style_id": old_style_id,
                "old_style_name": old_style_name,
                "new_style_id": style_id,
                "new_style_name": style_name,
            },
        )

    if default_style_id is not None:
        old_default = character_payload.get("default_style")
        old_default_id = old_default.get("style_id") if isinstance(old_default, dict) else None
        old_default_name = old_default.get("style_name") if isinstance(old_default, dict) else None

        default_style_name = _resolve_style_name(
            style_id=default_style_id,
            character=selected_character,
            mapping=mapping,
        )
        character_payload["default_style"] = {
            "style_id": default_style_id,
            "style_name": default_style_name,
        }
        used_ids.add(default_style_id)

        changes.append(
            {
                "type": "default_style",
                "old_style_id": old_default_id,
                "old_style_name": old_default_name,
                "new_style_id": default_style_id,
                "new_style_name": default_style_name,
            },
        )

    payload["selected_character"] = selected_character
    payload["styles_used"] = sorted(used_ids)
    payload["created_at"] = datetime.now(tz=UTC).isoformat()
    payload["manual_adjustment"] = {
        "updated_at": datetime.now(tz=UTC).isoformat(),
        "changes": changes,
    }

    out_path = Path(output_path) if output_path else in_path
    out_path.parent.mkdir(parents=True, exist_ok=True)

    backup_path: Path | None = None
    same_file = out_path.resolve() == in_path.resolve()
    if not dry_run and same_file and not no_backup:
        suffix = datetime.now(tz=UTC).strftime("%Y%m%d_%H%M%S")
        backup_path = in_path.with_suffix(f"{in_path.suffix}.bak_{suffix}")
        shutil.copy2(in_path, backup_path)

    if not dry_run:
        save_json(payload, out_path)

    return {
        "mapping_path": str(in_path),
        "output_path": str(out_path),
        "backup_path": str(backup_path) if backup_path is not None else None,
        "character": selected_character,
        "dry_run": dry_run,
        "changes": changes,
        "styles_used": payload["styles_used"],
    }


def main() -> None:
    _configure_logging()
    args = _build_parser().parse_args()

    result = run_adjust_style_mapping(
        mapping_path=args.mapping_path,
        output_path=args.output_path,
        character=args.character,
        set_args=args.set,
        default_style_id=args.default_style_id,
        dry_run=args.dry_run,
        no_backup=args.no_backup,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
