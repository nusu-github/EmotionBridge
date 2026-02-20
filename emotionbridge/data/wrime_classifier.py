from dataclasses import dataclass
from typing import Any

import numpy as np
from datasets import DatasetDict, concatenate_datasets, load_dataset
from sklearn.model_selection import train_test_split

from emotionbridge.config import DataConfig
from emotionbridge.constants import JVNV_EMOTION_LABELS

# WRIME 8Dラベル（データセット読み込み専用）
_WRIME_LABELS = ["joy", "sadness", "anticipation", "surprise", "anger", "fear", "disgust", "trust"]

# WRIME 8Dのうち、JVNV 6感情をJVNV順で取り出すためのインデックス
# _WRIME_LABELS = [joy, sadness, anticipation, surprise, anger, fear, disgust, trust]
# JVNV: [anger, disgust, fear, happy, sad, surprise]
_WRIME_TO_JVNV_INDICES = [4, 6, 5, 0, 1, 3]


@dataclass(slots=True)
class PreparedClassifierSplit:
    texts: list[str]
    labels: np.ndarray
    raw_targets: np.ndarray
    soft_targets: np.ndarray | None = None

    @property
    def size(self) -> int:
        return len(self.texts)


def _extract_label_from_nested(
    example: dict[str, Any],
    source: str,
) -> np.ndarray | None:
    labels = example.get(source)
    if not isinstance(labels, dict):
        return None
    return np.asarray(
        [float(labels[label]) for label in _WRIME_LABELS],
        dtype=np.float32,
    )


def _extract_label_from_flat(example: dict[str, Any], source: str) -> np.ndarray | None:
    candidates = []
    for sep in ["_", ".", "/"]:
        keys = [f"{source}{sep}{label}" for label in _WRIME_LABELS]
        if all(key in example for key in keys):
            candidates = keys
            break

    if not candidates:
        return None

    return np.asarray([float(example[key]) for key in candidates], dtype=np.float32)


def _extract_label_vector(example: dict[str, Any], source: str) -> np.ndarray:
    nested = _extract_label_from_nested(example, source)
    if nested is not None:
        return nested

    flat = _extract_label_from_flat(example, source)
    if flat is not None:
        return flat

    available = ", ".join(sorted(example.keys()))
    msg = f"Could not find label source '{source}'. Available keys: {available}"
    raise KeyError(msg)


def _load_all_records(
    data_config: DataConfig,
) -> tuple[list[str], list[np.ndarray], dict[str, int]]:
    dataset: DatasetDict = load_dataset(
        data_config.dataset_name,
        name=data_config.dataset_config_name,
        trust_remote_code=True,
    )

    split_sizes = {
        split_name: len(split_dataset) for split_name, split_dataset in dataset.items()
    }

    if data_config.use_official_split:
        ordered = [
            dataset[split_name]
            for split_name in ["train", "validation", "test"]
            if split_name in dataset
        ]
        merged = concatenate_datasets(ordered)
    else:
        merged = concatenate_datasets([dataset[name] for name in dataset])

    texts: list[str] = []
    raw_targets: list[np.ndarray] = []

    for example in merged:
        text = str(example[data_config.text_field]).strip()
        if not text:
            continue
        raw = _extract_label_vector(example, data_config.label_source)
        texts.append(text)
        raw_targets.append(raw)

    return texts, raw_targets, split_sizes


def _safe_split(
    indices: np.ndarray,
    test_size: float,
    random_state: int,
    stratify: np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray, bool]:
    try:
        train_idx, test_idx = train_test_split(
            indices,
            test_size=test_size,
            random_state=random_state,
            stratify=stratify,
        )
        return np.asarray(train_idx), np.asarray(test_idx), stratify is not None
    except ValueError:
        train_idx, test_idx = train_test_split(
            indices,
            test_size=test_size,
            random_state=random_state,
            stratify=None,
        )
        return np.asarray(train_idx), np.asarray(test_idx), False


def _build_soft_targets(raw_targets: np.ndarray, temperature: float) -> np.ndarray:
    safe_temperature = max(float(temperature), 1e-6)
    sums = np.maximum(raw_targets.sum(axis=1, keepdims=True), 1e-8)
    probs = raw_targets / sums

    if not np.isclose(safe_temperature, 1.0):
        scaled = np.power(np.clip(probs, 1e-8, None), 1.0 / safe_temperature)
        probs = scaled / np.maximum(scaled.sum(axis=1, keepdims=True), 1e-8)

    return probs.astype(np.float32)


def build_classifier_splits(
    data_config: DataConfig,
) -> tuple[dict[str, PreparedClassifierSplit], dict[str, Any]]:
    if not np.isclose(
        data_config.train_ratio + data_config.val_ratio + data_config.test_ratio,
        1.0,
    ):
        msg = "train_ratio + val_ratio + test_ratio must be 1.0"
        raise ValueError(msg)

    texts, raw_targets_list, original_sizes = _load_all_records(data_config)
    raw_targets8 = np.vstack(raw_targets_list).astype(np.float32)

    raw_max = raw_targets8.max(axis=1)
    filter_mask = raw_max > data_config.filter_max_intensity_lte

    filtered_texts = [
        text for text, keep in zip(texts, filter_mask, strict=True) if keep
    ]
    filtered_raw_targets8 = raw_targets8[filter_mask]
    filtered_raw_targets6 = filtered_raw_targets8[:, _WRIME_TO_JVNV_INDICES]

    label_conversion = str(getattr(data_config, "label_conversion", "argmax"))
    if label_conversion not in {"argmax", "soft_label"}:
        msg = "data.label_conversion must be one of: argmax, soft_label"
        raise ValueError(msg)

    hard_labels = np.argmax(filtered_raw_targets6, axis=1).astype(np.int64)
    soft_targets: np.ndarray | None = None

    if label_conversion == "soft_label":
        temperature = float(getattr(data_config, "soft_label_temperature", 1.0))
        soft_targets = _build_soft_targets(filtered_raw_targets6, temperature)

    stratify = hard_labels if data_config.stratify_after_filter else None

    indices = np.arange(len(filtered_texts))
    train_idx, holdout_idx, stratified_first = _safe_split(
        indices=indices,
        test_size=(data_config.val_ratio + data_config.test_ratio),
        random_state=data_config.random_seed,
        stratify=stratify,
    )

    holdout_ratio = data_config.val_ratio + data_config.test_ratio
    if holdout_ratio == 0:
        msg = "val_ratio + test_ratio must be greater than 0"
        raise ValueError(msg)
    test_ratio_within_holdout = data_config.test_ratio / holdout_ratio

    holdout_stratify = (
        hard_labels[holdout_idx]
        if stratified_first and data_config.stratify_after_filter
        else None
    )
    val_idx, test_idx, stratified_second = _safe_split(
        indices=holdout_idx,
        test_size=test_ratio_within_holdout,
        random_state=data_config.random_seed,
        stratify=holdout_stratify,
    )

    def _make_split(selected: np.ndarray) -> PreparedClassifierSplit:
        split_soft_targets = soft_targets[selected] if soft_targets is not None else None
        return PreparedClassifierSplit(
            texts=[filtered_texts[i] for i in selected],
            labels=hard_labels[selected],
            raw_targets=filtered_raw_targets6[selected],
            soft_targets=split_soft_targets,
        )

    splits = {
        "train": _make_split(train_idx),
        "val": _make_split(val_idx),
        "test": _make_split(test_idx),
    }

    class_counts = np.bincount(hard_labels, minlength=len(JVNV_EMOTION_LABELS))
    metadata = {
        "original_split_sizes": original_sizes,
        "num_records_total": len(texts),
        "num_records_filtered": len(filtered_texts),
        "num_records_removed": int((~filter_mask).sum()),
        "filtered_ratio": float(filter_mask.mean()) if len(filter_mask) else 0.0,
        "stratified_first_split": stratified_first,
        "stratified_second_split": stratified_second,
        "label_conversion": label_conversion,
        "class_distribution_filtered": {
            label: int(count)
            for label, count in zip(JVNV_EMOTION_LABELS, class_counts, strict=True)
        },
    }

    return splits, metadata


def _build_class_distribution(labels: np.ndarray) -> dict[str, dict[str, float]]:
    counts = np.bincount(labels.astype(np.int64), minlength=len(JVNV_EMOTION_LABELS))
    total = int(np.sum(counts))
    denominator = max(total, 1)
    return {
        label: {
            "count": int(count),
            "ratio": float(count / denominator),
        }
        for label, count in zip(JVNV_EMOTION_LABELS, counts, strict=True)
    }


def build_classifier_data_report(
    splits: dict[str, PreparedClassifierSplit],
    metadata: dict[str, Any],
) -> dict[str, Any]:
    return {
        "metadata": metadata,
        "split_sizes": {name: split.size for name, split in splits.items()},
        "class_distribution": {
            split_name: _build_class_distribution(split.labels)
            for split_name, split in splits.items()
        },
    }
