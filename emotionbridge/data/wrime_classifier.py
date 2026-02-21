from typing import Any, Mapping

import numpy as np
from datasets import ClassLabel, Dataset, DatasetDict, concatenate_datasets, load_dataset

from emotionbridge.config import DataConfig
from emotionbridge.constants import JVNV_EMOTION_LABELS

# WRIME 8Dラベル（データセット読み込み専用）
_WRIME_LABELS = ["joy", "sadness", "anticipation", "surprise", "anger", "fear", "disgust", "trust"]
_WRIME_LABEL_SOURCE = "avg_readers"

# WRIME 8Dのうち、JVNV 6感情をJVNV順で取り出すためのインデックス
# _WRIME_LABELS = [joy, sadness, anticipation, surprise, anger, fear, disgust, trust]
# JVNV: [anger, disgust, fear, happy, sad, surprise]
_WRIME_TO_JVNV_INDICES = [4, 6, 5, 0, 1, 3]


def _extract_label_from_nested(labels: Any) -> np.ndarray | None:
    if not isinstance(labels, Mapping):
        return None
    try:
        return np.asarray([float(labels[label]) for label in _WRIME_LABELS], dtype=np.float32)
    except (KeyError, TypeError, ValueError):
        return None


def _extract_label_vector(labels: Any) -> np.ndarray:
    nested = _extract_label_from_nested(labels)
    if nested is not None:
        return nested

    msg = (
        f"WRIME labels must be a mapping at '{_WRIME_LABEL_SOURCE}' and include keys:"
        f" {', '.join(_WRIME_LABELS)}"
    )
    raise KeyError(msg)


def _load_all_records(data_config: DataConfig) -> tuple[Dataset, dict[str, int]]:
    dataset: DatasetDict = load_dataset(
        data_config.dataset_name,
        name=data_config.dataset_config_name,
    )

    split_sizes = {
        str(split_name): len(split_dataset) for split_name, split_dataset in dataset.items()
    }

    split_names = list(dataset.keys())
    merged = concatenate_datasets([dataset[name] for name in split_names])
    return merged, split_sizes


def _build_soft_targets(raw_targets: np.ndarray, temperature: float) -> np.ndarray:
    safe_temperature = max(float(temperature), 1e-6)
    sums = np.maximum(raw_targets.sum(axis=1, keepdims=True), 1e-8)
    probs = raw_targets / sums

    if not np.isclose(safe_temperature, 1.0):
        scaled = np.power(np.clip(probs, 1e-8, None), 1.0 / safe_temperature)
        probs = scaled / np.maximum(scaled.sum(axis=1, keepdims=True), 1e-8)

    return probs.astype(np.float32)


def _prepare_dataset(
    data_config: DataConfig, merged_dataset: Dataset
) -> tuple[Dataset, dict[str, Any]]:
    label_conversion = str(getattr(data_config, "label_conversion", "argmax"))
    if label_conversion not in {"argmax", "soft_label"}:
        msg = "data.label_conversion must be one of: argmax, soft_label"
        raise ValueError(msg)

    soft_temperature = float(getattr(data_config, "soft_label_temperature", 1.0))
    max_intensity_threshold = float(data_config.filter_max_intensity_lte)

    if merged_dataset.num_rows == 0:
        msg = "WRIME dataset is empty after loading splits"
        raise ValueError(msg)

    if data_config.text_field not in merged_dataset.column_names:
        columns = ", ".join(sorted(merged_dataset.column_names))
        msg = f"text field '{data_config.text_field}' not found in dataset columns: {columns}"
        raise KeyError(msg)

    if _WRIME_LABEL_SOURCE not in merged_dataset.column_names:
        columns = ", ".join(sorted(merged_dataset.column_names))
        msg = f"label source '{_WRIME_LABEL_SOURCE}' not found in dataset columns: {columns}"
        raise KeyError(msg)

    def _process_batch(examples: Mapping[str, list[Any]]) -> dict[str, list[Any]]:
        texts_in = examples[data_config.text_field]
        labels_in = examples[_WRIME_LABEL_SOURCE]

        texts_out: list[str] = []
        raw_targets_out: list[list[float]] = []
        labels_out: list[int] = []
        keep_flags: list[bool] = []
        removed_empty_flags: list[bool] = []
        removed_low_intensity_flags: list[bool] = []

        for text_value, label_value in zip(texts_in, labels_in, strict=True):
            text = str(text_value).strip()

            if not text:
                texts_out.append("")
                raw_targets_out.append([0.0] * len(JVNV_EMOTION_LABELS))
                labels_out.append(0)
                keep_flags.append(False)
                removed_empty_flags.append(True)
                removed_low_intensity_flags.append(False)
                continue

            raw_target_8d = _extract_label_vector(label_value)
            raw_target = raw_target_8d[_WRIME_TO_JVNV_INDICES]
            label = int(np.argmax(raw_target))

            texts_out.append(text)
            raw_targets_out.append(raw_target.tolist())
            labels_out.append(label)

            if float(np.max(raw_target_8d)) <= max_intensity_threshold:
                keep_flags.append(False)
                removed_empty_flags.append(False)
                removed_low_intensity_flags.append(True)
            else:
                keep_flags.append(True)
                removed_empty_flags.append(False)
                removed_low_intensity_flags.append(False)

        output: dict[str, list[Any]] = {
            "text": texts_out,
            "raw_target": raw_targets_out,
            "label": labels_out,
            "_keep": keep_flags,
            "_removed_empty_text": removed_empty_flags,
            "_removed_low_intensity": removed_low_intensity_flags,
        }

        if label_conversion == "soft_label":
            soft_labels = np.zeros(
                (len(raw_targets_out), len(JVNV_EMOTION_LABELS)),
                dtype=np.float32,
            )
            valid_mask = np.asarray(keep_flags, dtype=bool)
            if np.any(valid_mask):
                valid_targets = np.asarray(raw_targets_out, dtype=np.float32)[valid_mask]
                soft_labels[valid_mask] = _build_soft_targets(valid_targets, soft_temperature)
            output["soft_labels"] = soft_labels.tolist()

        return output

    transformed = merged_dataset.map(
        _process_batch,
        batched=True,
        remove_columns=merged_dataset.column_names,
    )

    keep_mask = np.asarray(transformed["_keep"], dtype=bool)
    removed_empty_count = int(
        np.sum(np.asarray(transformed["_removed_empty_text"], dtype=np.int64))
    )
    removed_low_intensity_count = int(
        np.sum(np.asarray(transformed["_removed_low_intensity"], dtype=np.int64)),
    )

    filtered = transformed.filter(lambda example: bool(example["_keep"]))
    filtered = filtered.remove_columns(["_keep", "_removed_empty_text", "_removed_low_intensity"])

    num_records_loaded = int(merged_dataset.num_rows)
    num_records_non_empty_text = max(num_records_loaded - removed_empty_count, 0)
    num_records_filtered = int(np.sum(keep_mask))
    metadata = {
        "num_records_loaded": num_records_loaded,
        "num_records_non_empty_text": num_records_non_empty_text,
        "num_records_filtered": num_records_filtered,
        "num_records_removed": max(num_records_loaded - num_records_filtered, 0),
        "num_records_removed_empty_text": removed_empty_count,
        "num_records_removed_low_intensity": removed_low_intensity_count,
        "filtered_ratio": (
            float(num_records_filtered / num_records_loaded) if num_records_loaded > 0 else 0.0
        ),
        "label_conversion": label_conversion,
    }
    return filtered, metadata


def _split_stratified(
    dataset: Dataset,
    *,
    test_size: float,
    random_seed: int,
    split_name: str,
) -> DatasetDict:
    try:
        return dataset.train_test_split(
            test_size=test_size,
            seed=random_seed,
            stratify_by_column="label",
        )
    except ValueError as exc:
        labels = np.asarray(dataset["label"], dtype=np.int64)
        counts = np.bincount(labels, minlength=len(JVNV_EMOTION_LABELS))
        distribution = {
            label: int(count) for label, count in zip(JVNV_EMOTION_LABELS, counts, strict=True)
        }
        msg = (
            f"Stratified split failed at '{split_name}'. "
            f"test_size={test_size}, dataset_size={int(dataset.num_rows)}, "
            f"class_distribution={distribution}"
        )
        raise ValueError(msg) from exc


def build_classifier_splits(
    data_config: DataConfig,
) -> tuple[DatasetDict, dict[str, Any]]:
    if not np.isclose(
        data_config.train_ratio + data_config.val_ratio + data_config.test_ratio,
        1.0,
    ):
        msg = "train_ratio + val_ratio + test_ratio must be 1.0"
        raise ValueError(msg)

    merged_dataset, original_sizes = _load_all_records(data_config)
    prepared_dataset, prepared_meta = _prepare_dataset(data_config, merged_dataset)

    labels = np.asarray(prepared_dataset["label"], dtype=np.int64)
    class_counts = np.bincount(labels, minlength=len(JVNV_EMOTION_LABELS))

    holdout_ratio = data_config.val_ratio + data_config.test_ratio
    if holdout_ratio == 0:
        msg = "val_ratio + test_ratio must be greater than 0"
        raise ValueError(msg)

    split_source = prepared_dataset.cast_column(
        "label",
        ClassLabel(names=list(JVNV_EMOTION_LABELS)),
    )

    first_split = _split_stratified(
        split_source,
        test_size=holdout_ratio,
        random_seed=data_config.random_seed,
        split_name="train_holdout",
    )

    test_ratio_within_holdout = data_config.test_ratio / holdout_ratio
    second_split = _split_stratified(
        first_split["test"],
        test_size=test_ratio_within_holdout,
        random_seed=data_config.random_seed,
        split_name="val_test",
    )

    splits = DatasetDict(
        {
            "train": first_split["train"],
            "val": second_split["train"],
            "test": second_split["test"],
        },
    )

    metadata = {
        "original_split_sizes": original_sizes,
        **prepared_meta,
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


def build_classifier_data_report(splits: DatasetDict, metadata: dict[str, Any]) -> dict[str, Any]:
    return {
        "metadata": metadata,
        "split_sizes": {name: int(split.num_rows) for name, split in splits.items()},
        "class_distribution": {
            split_name: _build_class_distribution(np.asarray(split["label"], dtype=np.int64))
            for split_name, split in splits.items()
        },
    }
