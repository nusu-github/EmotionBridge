from collections.abc import Mapping
from typing import Any

import numpy as np
from datasets import ClassLabel, Dataset, DatasetDict, concatenate_datasets, load_dataset
from sklearn.cluster import KMeans

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
    data_config: DataConfig,
    merged_dataset: Dataset,
) -> tuple[Dataset, dict[str, Any]]:
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
        keep_flags: list[bool] = []
        removed_empty_flags: list[bool] = []
        removed_low_intensity_flags: list[bool] = []

        for text_value, label_value in zip(texts_in, labels_in, strict=True):
            text = str(text_value).strip()

            if not text:
                texts_out.append("")
                raw_targets_out.append([0.0] * len(JVNV_EMOTION_LABELS))
                keep_flags.append(False)
                removed_empty_flags.append(True)
                removed_low_intensity_flags.append(False)
                continue

            raw_target_8d = _extract_label_vector(label_value)
            raw_target = raw_target_8d[_WRIME_TO_JVNV_INDICES]

            texts_out.append(text)
            raw_targets_out.append(raw_target.tolist())

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
            "_keep": keep_flags,
            "_removed_empty_text": removed_empty_flags,
            "_removed_low_intensity": removed_low_intensity_flags,
        }

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
        np.sum(np.asarray(transformed["_removed_empty_text"], dtype=np.int64)),
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
        "soft_label_temperature": soft_temperature,
    }
    return filtered, metadata


def _split_stratified(
    dataset: Dataset,
    *,
    test_size: float,
    random_seed: int,
    split_name: str,
    stratify_column: str,
    group_names: list[str],
) -> DatasetDict:
    try:
        return dataset.train_test_split(
            test_size=test_size,
            seed=random_seed,
            stratify_by_column=stratify_column,
        )
    except ValueError as exc:
        labels = np.asarray(dataset[stratify_column], dtype=np.int64)
        counts = np.bincount(labels, minlength=len(group_names))
        distribution = {group: int(count) for group, count in zip(group_names, counts, strict=True)}
        msg = (
            f"Stratified split failed at '{split_name}'. "
            f"test_size={test_size}, dataset_size={int(dataset.num_rows)}, "
            f"group_distribution={distribution}"
        )
        raise ValueError(msg) from exc


def _build_stratification_clusters(
    soft_labels: np.ndarray,
    *,
    random_seed: int,
    max_clusters: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    if soft_labels.ndim != 2:
        msg = f"soft_labels must be 2D for clustering, got shape={soft_labels.shape}"
        raise ValueError(msg)

    num_samples = int(soft_labels.shape[0])
    if num_samples < 2:
        msg = "At least two samples are required to build stratification clusters"
        raise ValueError(msg)

    max_candidate = max(min(int(max_clusters), num_samples), 2)
    attempts: list[dict[str, int]] = []

    for num_clusters in range(max_candidate, 1, -1):
        model = KMeans(n_clusters=num_clusters, random_state=random_seed, n_init=20)
        cluster_ids = model.fit_predict(soft_labels).astype(np.int64)
        counts = np.bincount(cluster_ids, minlength=num_clusters)
        attempts.append(
            {
                "num_clusters": int(num_clusters),
                "min_cluster_size": int(np.min(counts)),
                "max_cluster_size": int(np.max(counts)),
            },
        )

        if int(np.min(counts)) >= 2:
            return cluster_ids, {
                "num_clusters": int(num_clusters),
                "cluster_distribution": {
                    f"cluster_{index}": int(count) for index, count in enumerate(counts.tolist())
                },
                "clustering_attempts": attempts,
            }

    msg = (
        "Failed to create valid stratification clusters from soft labels. "
        f"Need cluster size >= 2, attempts={attempts}"
    )
    raise ValueError(msg)


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

    soft_labels = np.asarray(prepared_dataset["soft_labels"], dtype=np.float32)
    cluster_ids, clustering_meta = _build_stratification_clusters(
        soft_labels,
        random_seed=data_config.random_seed,
        max_clusters=len(JVNV_EMOTION_LABELS),
    )
    cluster_names = [f"cluster_{i}" for i in range(clustering_meta["num_clusters"])]

    split_source = prepared_dataset.add_column("stratify_cluster", cluster_ids.tolist())
    split_source = split_source.cast_column(
        "stratify_cluster",
        ClassLabel(names=cluster_names),
    )

    holdout_ratio = data_config.val_ratio + data_config.test_ratio
    if holdout_ratio == 0:
        msg = "val_ratio + test_ratio must be greater than 0"
        raise ValueError(msg)

    first_split = _split_stratified(
        split_source,
        test_size=holdout_ratio,
        random_seed=data_config.random_seed,
        split_name="train_holdout",
        stratify_column="stratify_cluster",
        group_names=cluster_names,
    )

    test_ratio_within_holdout = data_config.test_ratio / holdout_ratio
    second_split = _split_stratified(
        first_split["test"],
        test_size=test_ratio_within_holdout,
        random_seed=data_config.random_seed,
        split_name="val_test",
        stratify_column="stratify_cluster",
        group_names=cluster_names,
    )

    splits = DatasetDict(
        {
            "train": first_split["train"].remove_columns(["stratify_cluster"]),
            "val": second_split["train"].remove_columns(["stratify_cluster"]),
            "test": second_split["test"].remove_columns(["stratify_cluster"]),
        },
    )

    mean_soft_distribution = np.mean(soft_labels, axis=0)

    metadata = {
        "original_split_sizes": original_sizes,
        **prepared_meta,
        "soft_distribution_filtered": {
            label: float(value)
            for label, value in zip(
                JVNV_EMOTION_LABELS,
                mean_soft_distribution.tolist(),
                strict=True,
            )
        },
        **clustering_meta,
    }

    return splits, metadata


def _build_soft_distribution(soft_labels: np.ndarray) -> dict[str, dict[str, float]]:
    means = np.mean(soft_labels, axis=0)
    stds = np.std(soft_labels, axis=0)
    return {
        label: {
            "mean_prob": float(mean),
            "std_prob": float(std),
        }
        for label, mean, std in zip(JVNV_EMOTION_LABELS, means, stds, strict=True)
    }


def build_classifier_data_report(splits: DatasetDict, metadata: dict[str, Any]) -> dict[str, Any]:
    return {
        "metadata": metadata,
        "split_sizes": {name: int(split.num_rows) for name, split in splits.items()},
        "soft_distribution": {
            split_name: _build_soft_distribution(np.asarray(split["soft_labels"], dtype=np.float32))
            for split_name, split in splits.items()
        },
    }
