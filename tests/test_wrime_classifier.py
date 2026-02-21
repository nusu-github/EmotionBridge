from __future__ import annotations

import unittest
from unittest.mock import patch

import pytest
from datasets import Dataset, DatasetDict

from emotionbridge.config import ClassifierDataConfig
from emotionbridge.data.wrime_classifier import (
    build_classifier_data_report,
    build_classifier_splits,
)

_WRIME_LABELS = [
    "joy",
    "sadness",
    "anticipation",
    "surprise",
    "anger",
    "fear",
    "disgust",
    "trust",
]
_WRIME_TO_JVNV_INDICES = [4, 6, 5, 0, 1, 3]


def _make_label_dict(common6_index: int, intensity: float = 3.0) -> dict[str, float]:
    wrime_index = _WRIME_TO_JVNV_INDICES[common6_index]
    values = [0.2] * len(_WRIME_LABELS)
    values[wrime_index] = float(intensity)
    return {name: float(value) for name, value in zip(_WRIME_LABELS, values, strict=True)}


def _build_mock_dataset() -> DatasetDict:
    rows: list[dict[str, object]] = []

    for label_index in range(6):
        rows.extend(
            {
                "sentence": f"text-{label_index}-{sample_index}",
                "avg_readers": _make_label_dict(label_index, intensity=3.0),
            }
            for sample_index in range(6)
        )

    rows.extend(
        (
            {"sentence": "low-0", "avg_readers": _make_label_dict(0, intensity=1.0)},
            {"sentence": "low-1", "avg_readers": _make_label_dict(1, intensity=1.0)},
            {"sentence": "   ", "avg_readers": _make_label_dict(2, intensity=3.0)},
        )
    )

    middle = len(rows) // 2
    return DatasetDict(
        {
            "train": Dataset.from_list(rows[:middle]),
            "test": Dataset.from_list(rows[middle:]),
        },
    )


def _build_flat_label_dataset() -> DatasetDict:
    rows: list[dict[str, object]] = []
    for sample_index in range(4):
        base = {
            "sentence": f"flat-{sample_index}",
        }
        base.update({f"avg_readers_{label}": 0.2 for label in _WRIME_LABELS})
        base["avg_readers_joy"] = 3.0
        rows.append(base)

    return DatasetDict(
        {
            "train": Dataset.from_list(rows),
        },
    )


def _build_cluster_failure_dataset() -> DatasetDict:
    rows = [
        {
            "sentence": "a",
            "avg_readers": _make_label_dict(0, intensity=3.0),
        },
        {
            "sentence": "b",
            "avg_readers": _make_label_dict(1, intensity=3.0),
        },
        {
            "sentence": "c",
            "avg_readers": _make_label_dict(2, intensity=3.0),
        },
    ]
    return DatasetDict(
        {
            "train": Dataset.from_list(rows),
        },
    )


class TestWrimeClassifier(unittest.TestCase):
    def test_build_classifier_splits_soft_only(self) -> None:
        config = ClassifierDataConfig(
            dataset_name="dummy",
            dataset_config_name="dummy",
            train_ratio=0.5,
            val_ratio=0.25,
            test_ratio=0.25,
            random_seed=42,
            filter_max_intensity_lte=1,
            soft_label_temperature=0.7,
        )

        with patch(
            "emotionbridge.data.wrime_classifier.load_dataset",
            return_value=_build_mock_dataset(),
        ):
            splits, metadata = build_classifier_splits(config)

        assert set(splits.keys()) == {"train", "val", "test"}
        assert sum(split.num_rows for split in splits.values()) == metadata["num_records_filtered"]
        assert metadata["num_records_loaded"] == 39
        assert metadata["num_records_non_empty_text"] == 38
        assert metadata["num_records_filtered"] == 36
        assert metadata["num_records_removed"] == 3
        assert metadata["num_records_removed_empty_text"] == 1
        assert metadata["num_records_removed_low_intensity"] == 2
        assert metadata["soft_label_temperature"] == pytest.approx(0.7)
        assert metadata["num_clusters"] >= 2
        assert sum(metadata["cluster_distribution"].values()) == metadata["num_records_filtered"]

        for split in splits.values():
            assert "soft_labels" in split.column_names
            assert "label" not in split.column_names
            for vector in split["soft_labels"]:
                assert len(vector) == 6
                assert abs(sum(vector) - 1.0) < 1e-5

        report = build_classifier_data_report(splits, metadata)
        assert report["split_sizes"]["train"] == splits["train"].num_rows
        assert report["split_sizes"]["val"] == splits["val"].num_rows
        assert report["split_sizes"]["test"] == splits["test"].num_rows
        assert "soft_distribution" in report

    def test_soft_label_temperature_changes_sharpness(self) -> None:
        low_temp = ClassifierDataConfig(
            dataset_name="dummy",
            dataset_config_name="dummy",
            train_ratio=0.5,
            val_ratio=0.25,
            test_ratio=0.25,
            random_seed=42,
            filter_max_intensity_lte=1,
            soft_label_temperature=0.7,
        )
        high_temp = ClassifierDataConfig(
            dataset_name="dummy",
            dataset_config_name="dummy",
            train_ratio=0.5,
            val_ratio=0.25,
            test_ratio=0.25,
            random_seed=42,
            filter_max_intensity_lte=1,
            soft_label_temperature=1.5,
        )

        with patch(
            "emotionbridge.data.wrime_classifier.load_dataset",
            return_value=_build_mock_dataset(),
        ):
            low_splits, _ = build_classifier_splits(low_temp)

        with patch(
            "emotionbridge.data.wrime_classifier.load_dataset",
            return_value=_build_mock_dataset(),
        ):
            high_splits, _ = build_classifier_splits(high_temp)

        low_max = max(max(row) for row in low_splits["train"]["soft_labels"])
        high_max = max(max(row) for row in high_splits["train"]["soft_labels"])
        assert low_max > high_max

    def test_legacy_flat_label_schema_raises(self) -> None:
        config = ClassifierDataConfig(
            dataset_name="dummy",
            dataset_config_name="dummy",
            train_ratio=0.8,
            val_ratio=0.1,
            test_ratio=0.1,
            filter_max_intensity_lte=1,
        )

        with (
            patch(
                "emotionbridge.data.wrime_classifier.load_dataset",
                return_value=_build_flat_label_dataset(),
            ),
            pytest.raises(KeyError, match="label source 'avg_readers' not found"),
        ):
            build_classifier_splits(config)

    def test_stratified_split_fail_fast_when_clusters_invalid(self) -> None:
        config = ClassifierDataConfig(
            dataset_name="dummy",
            dataset_config_name="dummy",
            train_ratio=0.5,
            val_ratio=0.25,
            test_ratio=0.25,
            random_seed=42,
            filter_max_intensity_lte=1,
        )

        with (
            patch(
                "emotionbridge.data.wrime_classifier.load_dataset",
                return_value=_build_cluster_failure_dataset(),
            ),
            pytest.raises(
                ValueError,
                match="Failed to create valid stratification clusters",
            ),
        ):
            build_classifier_splits(config)


if __name__ == "__main__":
    unittest.main()
