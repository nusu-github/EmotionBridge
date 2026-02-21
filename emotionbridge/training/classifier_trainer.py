import json
import logging
from pathlib import Path
from typing import Any, cast

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.optim import AdamW
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
    set_seed,
)

from emotionbridge.config import ClassifierConfig, save_effective_config
from emotionbridge.constants import JVNV_EMOTION_LABELS, KEY_EMOTION_LABELS, NUM_JVNV_EMOTIONS
from emotionbridge.data import build_classifier_data_report, build_classifier_splits
from emotionbridge.training.classification_metrics import compute_classification_metrics

logger = logging.getLogger(__name__)


class ClassifierBatchCollator:
    def __init__(self, tokenizer: Any, max_length: int) -> None:
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __call__(self, batch: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        texts = [sample["text"] for sample in batch]
        soft_targets = np.asarray([sample["soft_labels"] for sample in batch], dtype=np.float32)

        if soft_targets.ndim != 2:
            msg = f"soft_labels must be 2D (N, C), got shape={soft_targets.shape}"
            raise ValueError(msg)

        encoded = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        encoded["labels"] = torch.tensor(soft_targets, dtype=torch.float32)
        return encoded


class EmotionTrainer(Trainer):
    """EmotionBridge 固有の損失計算（soft label専用）をサポートする Trainer。"""

    @staticmethod
    def _to_scalar(value: Any) -> float | None:
        if isinstance(value, (int, float, np.integer, np.floating)):
            scalar = float(value)
            if np.isfinite(scalar):
                return scalar
        return None

    def _flatten_log_values(self, logs: dict[str, Any]) -> dict[str, float]:
        flattened: dict[str, float] = {}
        for key, value in logs.items():
            scalar = self._to_scalar(value)
            if scalar is not None:
                flattened[key] = scalar
                continue

            if isinstance(value, dict):
                for sub_key, sub_value in value.items():
                    sub_scalar = self._to_scalar(sub_value)
                    if sub_scalar is not None:
                        flattened[f"{key}_{sub_key}"] = sub_scalar

        return flattened

    def log(self, logs: dict[str, Any], *args: Any, **kwargs: Any) -> None:
        super().log(self._flatten_log_values(logs), *args, **kwargs)

    def compute_loss(
        self,
        model: nn.Module,
        inputs: dict[str, torch.Tensor],
        return_outputs: bool = False,
        num_items_in_batch: torch.Tensor | int | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, Any]:
        del num_items_in_batch

        targets = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits

        log_probs = F.log_softmax(logits, dim=-1)
        loss = -(targets * log_probs).sum(dim=-1).mean()

        return (loss, outputs) if return_outputs else loss


def _create_compute_metrics(key_emotions: list[str]):
    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        return compute_classification_metrics(
            logits,
            labels,
            JVNV_EMOTION_LABELS,
            key_emotions,
        )

    return compute_metrics


def _split_model_parameters(model: nn.Module) -> tuple[list[nn.Parameter], list[nn.Parameter]]:
    base_model_prefix = getattr(model, "base_model_prefix", None)
    if (
        isinstance(base_model_prefix, str)
        and base_model_prefix
        and hasattr(model, base_model_prefix)
    ):
        encoder = list(getattr(model, base_model_prefix).parameters())
        encoder_prefix = f"{base_model_prefix}."
        head = [
            parameter
            for name, parameter in model.named_parameters()
            if not name.startswith(encoder_prefix)
        ]
        return encoder, head

    return [], list(model.parameters())


def train_classifier(config: ClassifierConfig) -> dict[str, Any] | None:
    set_seed(config.data.random_seed)

    output_dir = Path(config.train.output_dir)
    reports_dir = output_dir / "reports"
    output_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(config.model.pretrained_model_name)
    splits, metadata = build_classifier_splits(config.data)

    train_dataset = splits["train"]
    val_dataset = splits["val"]
    test_dataset = splits["test"]

    id2label = dict(enumerate(JVNV_EMOTION_LABELS))
    label2id = {label: i for i, label in enumerate(JVNV_EMOTION_LABELS)}

    model = AutoModelForSequenceClassification.from_pretrained(
        config.model.pretrained_model_name,
        num_labels=NUM_JVNV_EMOTIONS,
        id2label=id2label,
        label2id=label2id,
        classifier_dropout=config.model.dropout,
    )

    training_args = TrainingArguments(
        output_dir=str(output_dir / "checkpoints"),
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_steps=config.train.log_every_steps,
        learning_rate=config.train.head_lr,
        per_device_train_batch_size=config.train.batch_size,
        per_device_eval_batch_size=config.train.batch_size,
        num_train_epochs=config.train.num_epochs,
        weight_decay=config.train.weight_decay,
        warmup_ratio=config.train.warmup_ratio,
        load_best_model_at_end=True,
        metric_for_best_model="mean_kl",
        greater_is_better=False,
        fp16=(config.train.mixed_precision == "fp16"),
        bf16=(config.train.mixed_precision == "bf16"),
        dataloader_num_workers=config.train.num_workers,
        dataloader_pin_memory=config.train.pin_memory,
        gradient_accumulation_steps=config.train.gradient_accumulation_steps,
        remove_unused_columns=False,
    )

    encoder_params, head_params = _split_model_parameters(model)
    optimizer_grouped_parameters: list[dict[str, Any]] = []
    if encoder_params:
        optimizer_grouped_parameters.append(
            {
                "params": encoder_params,
                "lr": config.train.bert_lr,
            },
        )
    if head_params:
        optimizer_grouped_parameters.append(
            {
                "params": head_params,
                "lr": config.train.head_lr,
            },
        )

    if not optimizer_grouped_parameters:
        msg = "No trainable parameters were found for optimizer setup"
        raise ValueError(msg)

    optimizer = AdamW(optimizer_grouped_parameters, weight_decay=config.train.weight_decay)

    trainer = EmotionTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        processing_class=tokenizer,
        data_collator=ClassifierBatchCollator(tokenizer, config.data.max_length),
        compute_metrics=_create_compute_metrics(KEY_EMOTION_LABELS),
        callbacks=[
            EarlyStoppingCallback(
                early_stopping_patience=config.train.early_stopping_patience,
            ),
        ],
        optimizers=(optimizer, None),
    )

    trainer.train()

    trainer.remove_callback(EarlyStoppingCallback)
    test_metrics = trainer.evaluate(cast("Any", test_dataset), metric_key_prefix="test")

    trainer.save_model(str(output_dir / "checkpoints" / "best_model"))

    data_report = build_classifier_data_report(splits, metadata)
    selected_metric_value = test_metrics.get("test_mean_kl", test_metrics.get("eval_mean_kl"))

    if trainer.is_world_process_zero():
        save_effective_config(config, output_dir / "effective_config.yaml")
        with (reports_dir / "data_report.json").open("w", encoding="utf-8") as file:
            json.dump(data_report, file, ensure_ascii=False, indent=2)

        with (reports_dir / "evaluation.json").open("w", encoding="utf-8") as file:
            json.dump(
                {
                    "metrics": test_metrics,
                    "selected_model_metric": {
                        "name": "mean_kl",
                        "direction": "lower_is_better",
                        "value": selected_metric_value,
                    },
                    "key_emotions": KEY_EMOTION_LABELS,
                },
                file,
                ensure_ascii=False,
                indent=2,
            )

    return {
        "output_dir": str(output_dir),
        "checkpoint": str(output_dir / "checkpoints" / "best_model"),
        "metrics": test_metrics,
        "selected_model_metric": {
            "name": "mean_kl",
            "direction": "lower_is_better",
            "value": selected_metric_value,
        },
    }
