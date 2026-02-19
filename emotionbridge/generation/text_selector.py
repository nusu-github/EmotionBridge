"""テキスト選定モジュール。

JVNVのtranscription.csvまたはWRIMEデータセットから
感情分布に基づいて学習用テキストを層化サンプリングする。
"""

import csv
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from emotionbridge.config import TextSelectionConfig
from emotionbridge.constants import EMOTION_LABELS, JVNV_EMOTION_LABELS
from emotionbridge.data.wrime import PreparedSplit
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


class TextSelector:
    """WRIMEデータセットから学習用テキストを選定する。

    build_wrime_splits() の全splitからテキストをプールし、
    dominant emotion (argmax) でグループ化して各グループから
    texts_per_emotion 件を均等サンプリングする。
    """

    def __init__(
        self,
        config: TextSelectionConfig,
        encoder: EmotionEncoder,
    ) -> None:
        self._config = config
        self._encoder = encoder

    def select(
        self,
        splits: dict[str, PreparedSplit],
    ) -> list[SelectedText]:
        """WRIMEの全splitからテキストを選定する。

        1. 全splitのテキストをプール
        2. テキスト長フィルタ
        3. dominant emotion (argmax) でグループ化
        4. 各グループから texts_per_emotion 件を均等サンプリング
        5. text_id を0始まりの連番で付与

        Args:
            splits: build_wrime_splits() の出力。

        Returns:
            選定されたテキストのリスト。

        """
        rng = np.random.default_rng(self._config.random_seed)

        # 全splitからテキストをプール
        candidates: list[dict[str, Any]] = []
        for split_name, split in splits.items():
            for i in range(split.size):
                text = split.texts[i]
                text_len = len(text)
                if text_len < self._config.min_text_length:
                    continue
                if text_len > self._config.max_text_length:
                    continue

                targets = split.targets[i]  # 正規化済み [0, 1]
                dominant_idx = int(np.argmax(targets))
                dominant_emotion = EMOTION_LABELS[dominant_idx]
                dominant_intensity = float(targets[dominant_idx])

                candidates.append(
                    {
                        "text": text,
                        "targets": targets,
                        "dominant_emotion": dominant_emotion,
                        "dominant_idx": dominant_idx,
                        "dominant_intensity": dominant_intensity,
                        "source_split": split_name,
                    },
                )

        logger.info(
            "テキスト候補数: %d (長さフィルタ後)",
            len(candidates),
        )

        # dominant emotion でグループ化
        emotion_groups: dict[str, list[dict[str, Any]]] = {label: [] for label in EMOTION_LABELS}
        for cand in candidates:
            emotion = str(cand["dominant_emotion"])
            emotion_groups[emotion].append(cand)

        for label, group in emotion_groups.items():
            logger.info("  %s: %d 件", label, len(group))

        # 各グループから texts_per_emotion 件を層化サンプリング
        selected: list[SelectedText] = []
        text_id_counter = 0

        for label in EMOTION_LABELS:
            group = emotion_groups[label]
            if not group:
                logger.warning(
                    "感情 '%s' のテキスト候補が0件。スキップ。",
                    label,
                )
                continue

            n_select = min(self._config.texts_per_emotion, len(group))
            sampled = self._stratified_sample_from_group(group, n_select, rng)

            for cand in sampled:
                text = str(cand["text"])
                targets = cand["targets"]
                assert isinstance(targets, np.ndarray)
                emotion_vec = targets.copy()

                selected.append(
                    SelectedText(
                        text_id=text_id_counter,
                        text=text,
                        emotion_vec=emotion_vec,
                        dominant_emotion=str(cand["dominant_emotion"]),
                        dominant_intensity=float(cand["dominant_intensity"]),
                        source_split=str(cand["source_split"]),
                    ),
                )
                text_id_counter += 1

        logger.info("選定テキスト数: %d", len(selected))
        return selected

    def _stratified_sample_from_group(
        self,
        group: list[dict[str, Any]],
        n_select: int,
        rng: np.random.Generator,
    ) -> list[dict[str, Any]]:
        """Dominant intensity に基づく層化サンプリング。

        intensity_bins 個のビンに分割し、各ビンから均等にサンプリングする。
        ビン内の候補が不足する場合は他のビンから補充する。

        Args:
            group: 同一 dominant emotion のテキスト候補リスト。
            n_select: 選定件数。
            rng: 乱数ジェネレータ。

        Returns:
            選定されたテキスト候補のリスト。

        """
        n_bins = self._config.intensity_bins

        # intensity でビン分割
        intensities = np.array(
            [float(c["dominant_intensity"]) for c in group],
            dtype=np.float32,
        )
        bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
        # 最大値も含むように右端を少し広げる
        bin_edges[-1] += 1e-6

        bins: list[list[int]] = [[] for _ in range(n_bins)]
        for idx, intensity in enumerate(intensities):
            for b in range(n_bins):
                if bin_edges[b] <= intensity < bin_edges[b + 1]:
                    bins[b].append(idx)
                    break

        # 各ビンから均等にサンプリング
        per_bin = n_select // n_bins
        remainder = n_select % n_bins
        sampled_indices: list[int] = []

        for b in range(n_bins):
            target = per_bin + (1 if b < remainder else 0)
            available = bins[b]
            if len(available) <= target:
                sampled_indices.extend(available)
            else:
                chosen = rng.choice(available, size=target, replace=False)
                sampled_indices.extend(chosen.tolist())

        # 不足分を全体から補充
        if len(sampled_indices) < n_select:
            remaining_indices = [i for i in range(len(group)) if i not in set(sampled_indices)]
            shortfall = n_select - len(sampled_indices)
            if remaining_indices:
                extra = rng.choice(
                    remaining_indices,
                    size=min(shortfall, len(remaining_indices)),
                    replace=False,
                )
                sampled_indices.extend(extra.tolist())

        return [group[i] for i in sampled_indices]


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
