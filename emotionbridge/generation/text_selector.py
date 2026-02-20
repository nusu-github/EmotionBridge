"""テキスト選定モジュール。

JVNVのtranscription.csvから感情分布に基づいて学習用テキストを層化サンプリングする。
"""

import csv
import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from emotionbridge.config import TextSelectionConfig
from emotionbridge.constants import JVNV_EMOTION_LABELS
from emotionbridge.inference.encoder import EmotionEncoder

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class SelectedText:
    """選定されたテキストの情報。"""

    text_id: int
    text: str
    emotion_vec: np.ndarray  # shape (6,)
    dominant_emotion: str
    dominant_intensity: float
    source_split: str


def _load_jvnv_csv(path: Path) -> list[dict[str, str]]:
    """JVNV transcription.csv をパイプ区切りで読み込む。

    形式: utterance_id | nv_phrase | full_text

    """
    entries: list[dict[str, str]] = []
    with path.open("r", encoding="utf-8") as f:
        reader = csv.reader(f, delimiter="|")
        for row in reader:
            if len(row) < 3:  # noqa: PLR2004
                continue
            utterance_id = row[0].strip()
            full_text = row[2].strip()
            if utterance_id and full_text:
                entries.append({"utterance_id": utterance_id, "full_text": full_text})
    return entries


def _filter_jvnv_candidates(
    entries: list[dict[str, str]],
    config: TextSelectionConfig,
) -> list[dict[str, str]]:
    """感情ラベルとテキスト長でフィルタリングする。"""
    valid_emotions = set(JVNV_EMOTION_LABELS)
    candidates: list[dict[str, str]] = []
    for entry in entries:
        emotion_label = entry["utterance_id"].split("_")[0]
        if emotion_label not in valid_emotions:
            continue
        text = entry["full_text"]
        if config.min_text_length <= len(text) <= config.max_text_length:
            candidates.append(
                {
                    "text": text,
                    "utterance_id": entry["utterance_id"],
                    "emotion_label": emotion_label,
                }
            )
    return candidates


def load_jvnv_texts(
    transcription_path: str,
    encoder: EmotionEncoder,
    config: TextSelectionConfig,
) -> list[SelectedText]:
    """JVNV transcription.csv からテキストを読み込み、感情エンコードして返す。

    utterance_id のプレフィックスから感情ラベルを取得（例: anger_regular_01 → anger）。

    Args:
        transcription_path: transcription.csv のパス。
        encoder: 感情エンコーダ（6D softmax 出力）。
        config: テキスト選定設定。

    Returns:
        選定されたテキストのリスト。

    """
    path = Path(transcription_path)
    if not path.exists():
        msg = f"Transcription file not found: {path}"
        raise FileNotFoundError(msg)

    raw_entries = _load_jvnv_csv(path)
    logger.info("JVNV transcription 読み込み: %d 件", len(raw_entries))

    candidates = _filter_jvnv_candidates(raw_entries, config)
    logger.info("テキスト候補数: %d (フィルタ後)", len(candidates))

    # 感情別にグループ化してサンプリング
    emotion_groups: dict[str, list[dict[str, str]]] = {label: [] for label in JVNV_EMOTION_LABELS}
    for cand in candidates:
        emotion_groups[cand["emotion_label"]].append(cand)

    for label, group in emotion_groups.items():
        logger.info("  %s: %d 件", label, len(group))

    rng = np.random.default_rng(config.random_seed)
    sampled: list[dict[str, str]] = []
    for label in JVNV_EMOTION_LABELS:
        group = emotion_groups[label]
        if not group:
            logger.warning("感情 '%s' のテキスト候補が0件。スキップ。", label)
            continue
        n_select = min(config.texts_per_emotion, len(group))
        if n_select < len(group):
            indices = rng.choice(len(group), size=n_select, replace=False)
            sampled.extend(group[i] for i in indices)
        else:
            sampled.extend(group)

    # EmotionEncoder でバッチエンコード → SelectedText 構築
    texts = [s["text"] for s in sampled]
    emotion_vecs = encoder.encode_batch(texts)

    selected: list[SelectedText] = []
    for i, cand in enumerate(sampled):
        vec = emotion_vecs[i]
        dominant_idx = int(np.argmax(vec))
        selected.append(
            SelectedText(
                text_id=i,
                text=cand["text"],
                emotion_vec=vec,
                dominant_emotion=JVNV_EMOTION_LABELS[dominant_idx],
                dominant_intensity=float(vec[dominant_idx]),
                source_split="jvnv",
            ),
        )

    logger.info("選定テキスト数: %d", len(selected))
    return selected
