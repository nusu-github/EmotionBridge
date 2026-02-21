from __future__ import annotations

from unittest.mock import patch
import unittest

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

    rows.append(
        {
            "sentence": "low-0",
            "avg_readers": _make_label_dict(0, intensity=1.0),
        },
    )
    rows.append(
        {
            "sentence": "low-1",
            "avg_readers": _make_label_dict(1, intensity=1.0),
        },
    )
    rows.append(
        {
            "sentence": "   ",
            "avg_readers": _make_label_dict(2, intensity=3.0),
        },
    )

    middle = len(rows) // 2
    return DatasetDict(
        {
            "train": Dataset.from_list(rows[:middle]),
            "test": Dataset.from_list(rows[middle:]),
        },
    )


class TestWrimeClassifier(unittest.TestCase):
    def test_build_classifier_splits_argmax(self) -> None:
        config = ClassifierDataConfig(
            dataset_name="dummy",
            dataset_config_name="dummy",
            train_ratio=0.5,
            val_ratio=0.25,
            test_ratio=0.25,
            random_seed=42,
            filter_max_intensity_lte=1,
            label_conversion="argmax",
            stratify_after_filter=True,
        )

        with patch(
            "emotionbridge.data.wrime_classifier.load_dataset",
            return_value=_build_mock_dataset(),
        ):
            splits, metadata = build_classifier_splits(config)

        assert set(splits.keys()) == {"train", "val", "test"}
        assert sum(split.num_rows for split in splits.values()) == metadata["num_records_filtered"]
        assert metadata["num_records_total"] == 38
        assert metadata["num_records_filtered"] == 36
        assert metadata["num_records_removed"] == 2
        assert "soft_labels" not in splits["train"].column_names

        filtered_count = sum(metadata["class_distribution_filtered"].values())
        assert filtered_count == metadata["num_records_filtered"]

        report = build_classifier_data_report(splits, metadata)
        assert report["split_sizes"]["train"] == splits["train"].num_rows
        assert report["split_sizes"]["val"] == splits["val"].num_rows
        assert report["split_sizes"]["test"] == splits["test"].num_rows

    def test_build_classifier_splits_soft_label(self) -> None:
        config = ClassifierDataConfig(
            dataset_name="dummy",
            dataset_config_name="dummy",
            train_ratio=0.5,
            val_ratio=0.25,
            test_ratio=0.25,
            random_seed=7,
            filter_max_intensity_lte=1,
            label_conversion="soft_label",
            soft_label_temperature=0.7,
            stratify_after_filter=True,
        )

        with patch(
            "emotionbridge.data.wrime_classifier.load_dataset",
            return_value=_build_mock_dataset(),
        ):
            splits, metadata = build_classifier_splits(config)

        assert metadata["label_conversion"] == "soft_label"
        for split in splits.values():
            assert "soft_labels" in split.column_names
            for vector in split["soft_labels"]:
                assert len(vector) == 6
                assert abs(sum(vector) - 1.0) < 1e-5

    def test_invalid_label_conversion_raises(self) -> None:
        config = ClassifierDataConfig(
            dataset_name="dummy",
            dataset_config_name="dummy",
            train_ratio=0.5,
            val_ratio=0.25,
            test_ratio=0.25,
            filter_max_intensity_lte=1,
            label_conversion="invalid",
        )

        with patch(
            "emotionbridge.data.wrime_classifier.load_dataset",
            return_value=_build_mock_dataset(),
        ):
            try:
                build_classifier_splits(config)
            except ValueError:
                pass
            else:
                msg = "Expected ValueError for invalid label_conversion"
                raise AssertionError(msg)


if __name__ == "__main__":
    unittest.main()
