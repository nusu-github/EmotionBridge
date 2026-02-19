from dataclasses import dataclass
from typing import Any

import numpy as np
from datasets import DatasetDict, concatenate_datasets, load_dataset
from sklearn.model_selection import train_test_split

from emotionbridge.config import DataConfig
from emotionbridge.constants import EMOTION_LABELS, LABEL_SCALE_MAX


@dataclass(slots=True)
class PreparedSplit:
    texts: list[str]
    targets: np.ndarray
    raw_targets: np.ndarray

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
        [float(labels[label]) for label in EMOTION_LABELS],
        dtype=np.float32,
    )


def _extract_label_from_flat(example: dict[str, Any], source: str) -> np.ndarray | None:
    candidates = []
    for sep in ["_", ".", "/"]:
        keys = [f"{source}{sep}{label}" for label in EMOTION_LABELS]
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
    raise KeyError(
        msg,
    )


def _load_all_records(
    data_config: DataConfig,
) -> tuple[list[str], list[np.ndarray], dict[str, int]]:
    dataset: DatasetDict = load_dataset(
        data_config.dataset_name,
        name=data_config.dataset_config_name,
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


def build_wrime_splits(
    data_config: DataConfig,
) -> tuple[dict[str, PreparedSplit], dict[str, Any]]:
    if not np.isclose(
        data_config.train_ratio + data_config.val_ratio + data_config.test_ratio,
        1.0,
    ):
        msg = "train_ratio + val_ratio + test_ratio must be 1.0"
        raise ValueError(msg)

    texts, raw_targets_list, original_sizes = _load_all_records(data_config)
    raw_targets = np.vstack(raw_targets_list).astype(np.float32)

    raw_max = raw_targets.max(axis=1)
    filter_mask = raw_max > data_config.filter_max_intensity_lte

    filtered_texts = [
        text for text, keep in zip(texts, filter_mask, strict=True) if keep
    ]
    filtered_raw_targets = raw_targets[filter_mask]
    normalized_targets = filtered_raw_targets / LABEL_SCALE_MAX

    stratify_key = np.argmax(filtered_raw_targets, axis=1)
    stratify = stratify_key if data_config.stratify_after_filter else None

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
        stratify_key[holdout_idx]
        if stratified_first and data_config.stratify_after_filter
        else None
    )
    val_idx, test_idx, stratified_second = _safe_split(
        indices=holdout_idx,
        test_size=test_ratio_within_holdout,
        random_state=data_config.random_seed,
        stratify=holdout_stratify,
    )

    def _make_split(selected: np.ndarray) -> PreparedSplit:
        return PreparedSplit(
            texts=[filtered_texts[i] for i in selected],
            targets=normalized_targets[selected],
            raw_targets=filtered_raw_targets[selected],
        )

    splits = {
        "train": _make_split(train_idx),
        "val": _make_split(val_idx),
        "test": _make_split(test_idx),
    }

    metadata = {
        "original_split_sizes": original_sizes,
        "num_records_total": len(texts),
        "num_records_filtered": len(filtered_texts),
        "num_records_removed": int((~filter_mask).sum()),
        "filtered_ratio": float(filter_mask.mean()) if len(filter_mask) else 0.0,
        "stratified_first_split": stratified_first,
        "stratified_second_split": stratified_second,
    }
    return splits, metadata


def estimate_unk_ratio(
    tokenizer: Any,
    texts: list[str],
    max_samples: int = 2000,
) -> float:
    sampled = texts[:max_samples]
    if not sampled:
        return 0.0

    unk_id = tokenizer.unk_token_id
    if unk_id is None:
        return 0.0

    encoded = tokenizer(
        sampled,
        add_special_tokens=False,
        truncation=True,
        max_length=128,
    )

    total_tokens = 0
    unk_tokens = 0
    for input_ids in encoded["input_ids"]:
        total_tokens += len(input_ids)
        unk_tokens += sum(token == unk_id for token in input_ids)

    if total_tokens == 0:
        return 0.0
    return float(unk_tokens / total_tokens)


def build_data_report(
    splits: dict[str, PreparedSplit],
    metadata: dict[str, Any],
) -> dict[str, Any]:
    report = {
        "metadata": metadata,
        "split_sizes": {name: split.size for name, split in splits.items()},
        "emotion_means": {},
    }

    for split_name, split in splits.items():
        split_mean = (
            split.targets.mean(axis=0)
            if split.size
            else np.zeros(len(EMOTION_LABELS), dtype=np.float32)
        )
        report["emotion_means"][split_name] = {
            emotion: float(value)
            for emotion, value in zip(EMOTION_LABELS, split_mean, strict=True)
        }

    return report
