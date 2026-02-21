from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import patch

from datasets import Dataset, DatasetDict
import pytest
import torch
from torch import nn
from transformers import TrainingArguments

from emotionbridge.config import (
    ClassifierConfig,
    ClassifierDataConfig,
    ClassifierModelConfig,
    ClassifierTrainConfig,
)
from emotionbridge.training.classifier_trainer import (
    ClassifierBatchCollator,
    EmotionTrainer,
    train_classifier,
)


class _DummyTokenizer:
    def __call__(
        self,
        texts: list[str],
        padding: bool,
        truncation: bool,
        max_length: int,
        return_tensors: str,
    ) -> dict[str, torch.Tensor]:
        del padding, truncation, max_length, return_tensors
        batch_size = len(texts)
        return {
            "input_ids": torch.ones((batch_size, 4), dtype=torch.long),
            "attention_mask": torch.ones((batch_size, 4), dtype=torch.long),
        }


class _DummyModel(nn.Module):
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> SimpleNamespace:
        del input_ids, attention_mask
        logits = torch.tensor(
            [
                [1.2, 0.2, -0.5],
                [0.1, 1.1, 0.2],
            ],
            dtype=torch.float32,
        )
        return SimpleNamespace(logits=logits)


class _TinyClassifier(nn.Module):
    base_model_prefix = "encoder"

    def __init__(self) -> None:
        super().__init__()
        self.encoder = nn.Linear(4, 4)
        self.classifier = nn.Linear(4, 6)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> SimpleNamespace:
        del attention_mask
        hidden = input_ids.float()
        logits = self.classifier(hidden)
        return SimpleNamespace(logits=logits)


def _make_soft_splits() -> tuple[DatasetDict, dict[str, object]]:
    rows = [
        {
            "text": "sample-1",
            "raw_target": [3.0, 0.2, 0.2, 0.2, 0.2, 0.2],
            "soft_labels": [0.75, 0.05, 0.05, 0.05, 0.05, 0.05],
        },
        {
            "text": "sample-2",
            "raw_target": [0.2, 3.0, 0.2, 0.2, 0.2, 0.2],
            "soft_labels": [0.05, 0.75, 0.05, 0.05, 0.05, 0.05],
        },
        {
            "text": "sample-3",
            "raw_target": [0.2, 0.2, 3.0, 0.2, 0.2, 0.2],
            "soft_labels": [0.05, 0.05, 0.75, 0.05, 0.05, 0.05],
        },
    ]
    dataset = Dataset.from_list(rows)
    return (
        DatasetDict({"train": dataset, "val": dataset, "test": dataset}),
        {"num_records_filtered": len(rows)},
    )


def test_batch_collator_outputs_soft_targets_as_labels() -> None:
    collator = ClassifierBatchCollator(_DummyTokenizer(), max_length=16)
    batch = [
        {
            "text": "a",
            "soft_labels": [0.70, 0.10, 0.10, 0.05, 0.03, 0.02],
        },
        {
            "text": "b",
            "soft_labels": [0.05, 0.75, 0.05, 0.05, 0.05, 0.05],
        },
    ]

    encoded = collator(batch)

    assert encoded["labels"].dtype == torch.float32
    assert tuple(encoded["labels"].shape) == (2, 6)
    assert torch.allclose(
        encoded["labels"].sum(dim=1),
        torch.tensor([1.0, 1.0], dtype=torch.float32),
        atol=1e-6,
    )


def test_emotion_trainer_compute_loss_uses_soft_cross_entropy() -> None:
    with TemporaryDirectory() as tmp_dir:
        empty_dataset = Dataset.from_list([])
        trainer = EmotionTrainer(
            model=_DummyModel(),
            args=TrainingArguments(output_dir=tmp_dir, per_device_train_batch_size=2),
            train_dataset=empty_dataset,
            eval_dataset=empty_dataset,
        )

        labels = torch.tensor(
            [
                [0.80, 0.10, 0.10],
                [0.10, 0.70, 0.20],
            ],
            dtype=torch.float32,
        )
        inputs = {
            "input_ids": torch.ones((2, 4), dtype=torch.long),
            "attention_mask": torch.ones((2, 4), dtype=torch.long),
            "labels": labels,
        }

        loss = trainer.compute_loss(_DummyModel(), inputs)

    assert isinstance(loss, torch.Tensor)

    logits = torch.tensor(
        [
            [1.2, 0.2, -0.5],
            [0.1, 1.1, 0.2],
        ],
        dtype=torch.float32,
    )
    expected = -(labels * torch.log_softmax(logits, dim=-1)).sum(dim=-1).mean()
    assert float(loss) == pytest.approx(float(expected), rel=1e-6)


def test_train_classifier_uses_mean_kl_for_best_model_selection() -> None:
    with TemporaryDirectory() as tmp_dir:
        splits, metadata = _make_soft_splits()
        trainer_instances: list[_FakeTrainer] = []

        class _FakeTrainer:
            def __init__(self, *args, **kwargs) -> None:
                del args
                self.init_kwargs = kwargs
                self.saved_path: str | None = None
                trainer_instances.append(self)

            def train(self) -> None:
                return None

            def evaluate(self, dataset, metric_key_prefix: str = "eval") -> dict[str, float]:
                del dataset, metric_key_prefix
                return {
                    "test_mean_kl": 0.111,
                    "test_brier_score": 0.222,
                }

            def save_model(self, path: str) -> None:
                self.saved_path = path

            def is_world_process_zero(self) -> bool:
                return True

            def remove_callback(self, callback_type) -> None:
                del callback_type

        config = ClassifierConfig(
            data=ClassifierDataConfig(
                dataset_name="dummy",
                dataset_config_name="dummy",
                max_length=16,
                random_seed=42,
            ),
            model=ClassifierModelConfig(
                pretrained_model_name="dummy-model",
                dropout=0.1,
                num_classes=6,
            ),
            train=ClassifierTrainConfig(
                output_dir=tmp_dir,
                batch_size=2,
                num_epochs=1,
                bert_lr=2e-5,
                head_lr=1e-3,
                weight_decay=0.01,
                warmup_ratio=0.1,
                early_stopping_patience=1,
                device="cpu",
                num_workers=0,
                pin_memory=False,
                log_every_steps=1,
                gradient_accumulation_steps=1,
                mixed_precision="no",
            ),
        )

        with (
            patch(
                "emotionbridge.training.classifier_trainer.AutoTokenizer.from_pretrained",
                return_value=_DummyTokenizer(),
            ),
            patch(
                "emotionbridge.training.classifier_trainer.build_classifier_splits",
                return_value=(splits, metadata),
            ),
            patch(
                "emotionbridge.training.classifier_trainer.AutoModelForSequenceClassification.from_pretrained",
                return_value=_TinyClassifier(),
            ),
            patch(
                "emotionbridge.training.classifier_trainer.TrainingArguments",
                side_effect=SimpleNamespace,
            ),
            patch(
                "emotionbridge.training.classifier_trainer.EmotionTrainer",
                _FakeTrainer,
            ),
            patch(
                "emotionbridge.training.classifier_trainer.build_classifier_data_report",
                return_value={"ok": True},
            ),
            patch("emotionbridge.training.classifier_trainer.save_effective_config"),
        ):
            result = train_classifier(config)

        assert result is not None
        assert result["selected_model_metric"]["name"] == "mean_kl"
        assert result["selected_model_metric"]["direction"] == "lower_is_better"
        assert result["metrics"]["test_mean_kl"] == pytest.approx(0.111)

        trainer_instance = trainer_instances[0]
        training_args = trainer_instance.init_kwargs["args"]
        assert training_args.metric_for_best_model == "mean_kl"
        assert training_args.greater_is_better is False

        reports_dir = Path(tmp_dir) / "reports"
        assert (reports_dir / "evaluation.json").exists()
        assert (reports_dir / "data_report.json").exists()
