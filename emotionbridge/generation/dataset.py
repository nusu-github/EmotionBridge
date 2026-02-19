"""三つ組データセットの永続化モジュール。

TripletRecord のリストを Apache Parquet 形式で書き出し・読み込みする。
列名規則:
- emotion_*: 感情エンコーダ出力 (8列)
- ctrl_*: 制御空間パラメータ (5列, [-1, +1])
- vv_*: VOICEVOX AudioQueryに適用された実値
"""

import json
import logging
from dataclasses import dataclass
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from emotionbridge.constants import CONTROL_PARAM_NAMES, EMOTION_LABELS

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class TripletRecord:
    """三つ組データセットの1レコード。"""

    text_id: int
    text: str
    emotion_vec: list[float]  # len=8
    dominant_emotion: str
    control_params: list[float]  # len=5
    audio_path: str
    audio_duration_seconds: float
    audio_file_size_bytes: int
    sample_rate: int
    style_id: int
    voicevox_params: dict[str, float]
    is_valid: bool
    generation_timestamp: str


# Parquetスキーマ定義
_PARQUET_SCHEMA = pa.schema(
    [
        pa.field("text_id", pa.int32()),
        pa.field("text", pa.string()),
    ]
    + [pa.field(f"emotion_{label}", pa.float32()) for label in EMOTION_LABELS]
    + [
        pa.field("dominant_emotion", pa.string()),
    ]
    + [pa.field(f"ctrl_{name}", pa.float32()) for name in CONTROL_PARAM_NAMES]
    + [
        pa.field("audio_path", pa.string()),
        pa.field("audio_duration_sec", pa.float32()),
        pa.field("audio_file_size_bytes", pa.int32()),
        pa.field("sample_rate", pa.int32()),
        pa.field("style_id", pa.int32()),
        pa.field("vv_speedScale", pa.float32()),
        pa.field("vv_pitchScale", pa.float32()),
        pa.field("vv_intonationScale", pa.float32()),
        pa.field("vv_volumeScale", pa.float32()),
        pa.field("vv_prePhonemeLength", pa.float32()),
        pa.field("vv_postPhonemeLength", pa.float32()),
        pa.field("vv_pauseLengthScale", pa.float32()),
        pa.field("is_valid", pa.bool_()),
        pa.field("generation_timestamp", pa.string()),
    ],
)

# voicevox_params のキー順序（Parquet列名との対応）
_VV_PARAM_KEYS = [
    "speedScale",
    "pitchScale",
    "intonationScale",
    "volumeScale",
    "prePhonemeLength",
    "postPhonemeLength",
    "pauseLengthScale",
]


def save_dataset(records: list[TripletRecord], output_path: Path) -> None:
    """TripletRecordリストをParquetファイルに書き出す。

    Args:
        records: 書き出すレコードのリスト。
        output_path: 出力ファイルパス。

    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    columns: dict[str, list[object]] = {
        "text_id": [],
        "text": [],
    }
    for label in EMOTION_LABELS:
        columns[f"emotion_{label}"] = []
    columns["dominant_emotion"] = []
    for name in CONTROL_PARAM_NAMES:
        columns[f"ctrl_{name}"] = []
    columns["audio_path"] = []
    columns["audio_duration_sec"] = []
    columns["audio_file_size_bytes"] = []
    columns["sample_rate"] = []
    columns["style_id"] = []
    for key in _VV_PARAM_KEYS:
        columns[f"vv_{key}"] = []
    columns["is_valid"] = []
    columns["generation_timestamp"] = []

    for rec in records:
        columns["text_id"].append(rec.text_id)
        columns["text"].append(rec.text)

        for i, label in enumerate(EMOTION_LABELS):
            columns[f"emotion_{label}"].append(rec.emotion_vec[i])

        columns["dominant_emotion"].append(rec.dominant_emotion)

        for i, name in enumerate(CONTROL_PARAM_NAMES):
            columns[f"ctrl_{name}"].append(rec.control_params[i])

        columns["audio_path"].append(rec.audio_path)
        columns["audio_duration_sec"].append(rec.audio_duration_seconds)
        columns["audio_file_size_bytes"].append(rec.audio_file_size_bytes)
        columns["sample_rate"].append(rec.sample_rate)
        columns["style_id"].append(rec.style_id)

        for key in _VV_PARAM_KEYS:
            columns[f"vv_{key}"].append(rec.voicevox_params.get(key, 0.0))

        columns["is_valid"].append(rec.is_valid)
        columns["generation_timestamp"].append(rec.generation_timestamp)

    table = pa.table(columns, schema=_PARQUET_SCHEMA)
    pq.write_table(table, str(output_path))

    logger.info(
        "データセット書き出し完了: %s (%d レコード)",
        output_path,
        len(records),
    )


def load_dataset(path: Path) -> list[TripletRecord]:
    """ParquetファイルからTripletRecordリストを読み込む。

    Args:
        path: Parquetファイルのパス。

    Returns:
        TripletRecordのリスト。

    """
    table = pq.read_table(str(path))
    columns = {name: table.column(name).to_pylist() for name in table.schema.names}

    n_rows = table.num_rows
    records: list[TripletRecord] = []

    for i in range(n_rows):
        emotion_vec = [
            float(columns[f"emotion_{label}"][i]) for label in EMOTION_LABELS
        ]
        control_params = [
            float(columns[f"ctrl_{name}"][i]) for name in CONTROL_PARAM_NAMES
        ]
        voicevox_params: dict[str, float] = {}
        for key in _VV_PARAM_KEYS:
            col_name = f"vv_{key}"
            if col_name in columns:
                voicevox_params[key] = float(columns[col_name][i])

        records.append(
            TripletRecord(
                text_id=int(columns["text_id"][i]),
                text=str(columns["text"][i]),
                emotion_vec=emotion_vec,
                dominant_emotion=str(columns["dominant_emotion"][i]),
                control_params=control_params,
                audio_path=str(columns["audio_path"][i]),
                audio_duration_seconds=float(columns["audio_duration_sec"][i]),
                audio_file_size_bytes=int(columns["audio_file_size_bytes"][i]),
                sample_rate=int(columns["sample_rate"][i]),
                style_id=int(columns["style_id"][i]),
                voicevox_params=voicevox_params,
                is_valid=bool(columns["is_valid"][i]),
                generation_timestamp=str(columns["generation_timestamp"][i]),
            ),
        )

    logger.info(
        "データセット読み込み完了: %s (%d レコード)",
        path,
        len(records),
    )
    return records


def save_metadata(metadata: dict[str, object], output_path: Path) -> None:
    """メタデータをJSONファイルに書き出す。

    Args:
        metadata: メタデータ辞書。
        output_path: 出力ファイルパス。

    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

    logger.info("メタデータ書き出し完了: %s", output_path)
