from typing import Any, Mapping

import numpy as np
from datasets import ClassLabel, Dataset, DatasetDict, concatenate_datasets, load_dataset

from emotionbridge.config import DataConfig
from emotionbridge.constants import JVNV_EMOTION_LABELS

# WRIME 8Dラベル（データセット読み込み専用）
_WRIME_LABELS = ["joy", "sadness", "anticipation", "surprise", "anger", "fear", "disgust", "trust"]

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


def _extract_label_from_flat(example: Mapping[str, Any], source: str) -> np.ndarray | None:
    candidates = []
    for sep in ["_", ".", "/"]:
        keys = [f"{source}{sep}{label}" for label in _WRIME_LABELS]
        if all(key in example for key in keys):
            candidates = keys
            break

    if not candidates:
        return None

    return np.asarray([float(example[key]) for key in candidates], dtype=np.float32)


def _extract_label_vector(example: Mapping[str, Any], source: str) -> np.ndarray:
    nested = _extract_label_from_nested(example.get(source))
    if nested is not None:
        return nested

    flat = _extract_label_from_flat(example, source)
    if flat is not None:
        return flat

    available = ", ".join(sorted(example.keys()))
    msg = f"Could not find label source '{source}'. Available keys: {available}"
    raise KeyError(msg)


def _load_all_records(data_config: DataConfig) -> tuple[Dataset, dict[str, int]]:
    dataset: DatasetDict = load_dataset(
        data_config.dataset_name,
        name=data_config.dataset_config_name,
        trust_remote_code=True,
    )

    split_sizes = {str(split_name): len(split_dataset) for split_name, split_dataset in dataset.items()}

    if data_config.use_official_split:
        split_names = [name for name in ["train", "validation", "test"] if name in dataset]
        if not split_names:
            split_names = list(dataset.keys())
    else:
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


def _prepare_dataset(data_config: DataConfig, merged_dataset: Dataset) -> tuple[Dataset, dict[str, Any]]:
    def _extract_text_and_label(example: Mapping[str, Any]) -> dict[str, Any]:
        text = str(example.get(data_config.text_field, "")).strip()
        raw_target_8d = _extract_label_vector(example, data_config.label_source)
        return {
            "text": text,
            "raw_target_8d": raw_target_8d.tolist(),
        }

    extracted = merged_dataset.map(_extract_text_and_label)
    extracted = extracted.filter(lambda example: bool(example["text"]))

    num_records_total = int(extracted.num_rows)

    filtered = extracted.filter(
        lambda example: max(float(value) for value in example["raw_target_8d"])
        > data_config.filter_max_intensity_lte,
    )

    label_conversion = str(getattr(data_config, "label_conversion", "argmax"))
    if label_conversion not in {"argmax", "soft_label"}:
        msg = "data.label_conversion must be one of: argmax, soft_label"
        raise ValueError(msg)

    soft_temperature = float(getattr(data_config, "soft_label_temperature", 1.0))

    def _to_common_labels(example: Mapping[str, Any]) -> dict[str, Any]:
        raw_target_8d = np.asarray(example["raw_target_8d"], dtype=np.float32)
        raw_target = raw_target_8d[_WRIME_TO_JVNV_INDICES]

        output: dict[str, Any] = {
            "raw_target": raw_target.tolist(),
            "label": int(np.argmax(raw_target)),
        }

        if label_conversion == "soft_label":
            soft_labels = _build_soft_targets(raw_target[None, :], soft_temperature)[0]
            output["soft_labels"] = soft_labels.tolist()

        return output

    filtered = filtered.map(_to_common_labels)
    filtered = filtered.remove_columns(["raw_target_8d"])

    num_records_filtered = int(filtered.num_rows)
    metadata = {
        "num_records_total": num_records_total,
        "num_records_filtered": num_records_filtered,
        "num_records_removed": max(num_records_total - num_records_filtered, 0),
        "filtered_ratio": (
            float(num_records_filtered / num_records_total) if num_records_total > 0 else 0.0
        ),
        "label_conversion": label_conversion,
    }
    return filtered, metadata


def _split_with_optional_stratify(
    dataset: Dataset,
    *,
    test_size: float,
    random_seed: int,
    stratify: bool,
) -> tuple[DatasetDict, bool]:
    if stratify:
        try:
            return (
                dataset.train_test_split(
                    test_size=test_size,
                    seed=random_seed,
                    stratify_by_column="label",
                ),
                True,
            )
        except ValueError:
            pass

    return (
        dataset.train_test_split(
            test_size=test_size,
            seed=random_seed,
        ),
        False,
    )


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

    first_split, stratified_first = _split_with_optional_stratify(
        split_source,
        test_size=holdout_ratio,
        random_seed=data_config.random_seed,
        stratify=data_config.stratify_after_filter,
    )

    test_ratio_within_holdout = data_config.test_ratio / holdout_ratio
    second_split, stratified_second = _split_with_optional_stratify(
        first_split["test"],
        test_size=test_ratio_within_holdout,
        random_seed=data_config.random_seed,
        stratify=data_config.stratify_after_filter and stratified_first,
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
        "stratified_first_split": stratified_first,
        "stratified_second_split": stratified_second,
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
