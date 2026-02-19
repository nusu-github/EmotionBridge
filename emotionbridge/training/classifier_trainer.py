from __future__ import annotations

import logging
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from transformers import (
    AutoTokenizer,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
    set_seed,
)

from emotionbridge.config import ClassifierConfig, save_effective_config
from emotionbridge.constants import (
    JVNV_EMOTION_LABELS,
    KEY_EMOTION_LABELS,
    NUM_JVNV_EMOTIONS,
)
from emotionbridge.data import (
    build_classifier_data_report,
    build_classifier_splits,
)
from emotionbridge.model import TextEmotionClassifier
from emotionbridge.training.classification_metrics import compute_classification_metrics

logger = logging.getLogger(__name__)


class EmotionClassifierTextDataset(Dataset):
    def __init__(
        self,
        texts: list[str],
        labels: np.ndarray,
        soft_targets: np.ndarray | None = None,
    ) -> None:
        self._texts = texts
        self._labels = labels
        self._soft_targets = soft_targets

    def __len__(self) -> int:
        return len(self._texts)

    def __getitem__(self, index: int) -> dict[str, Any]:
        sample: dict[str, Any] = {
            "text": self._texts[index],
            "label": self._labels[index],
        }
        if self._soft_targets is not None:
            sample["soft_target"] = self._soft_targets[index]
        return sample


class ClassifierBatchCollator:
    def __init__(self, tokenizer: Any, max_length: int) -> None:
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __call__(self, batch: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        texts = [sample["text"] for sample in batch]
        labels = np.asarray([sample["label"] for sample in batch], dtype=np.int64)
        encoded = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        # Trainer は 'labels' キーを期待する
        encoded["labels"] = torch.tensor(labels, dtype=torch.long)

        if all("soft_target" in sample for sample in batch):
            soft_targets = np.asarray(
                [sample["soft_target"] for sample in batch],
                dtype=np.float32,
            )
            encoded["soft_labels"] = torch.tensor(soft_targets, dtype=torch.float32)

        return encoded


class EmotionTrainer(Trainer):
    """EmotionBridge 固有の損失計算（soft_label, class_weights）をサポートする Trainer。"""

    def __init__(
        self,
        *args,
        class_weights: torch.Tensor | None = None,
        label_conversion: str = "argmax",
        soft_label_temperature: float = 1.0,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights
        self.label_conversion = label_conversion
        self.soft_label_temperature = soft_label_temperature

    def compute_loss(
        self,
        model: nn.Module,
        inputs: dict[str, torch.Tensor],
        return_outputs: bool = False,
        num_items_in_batch: int | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, Any]:
        labels = inputs.pop("labels")
        soft_labels = inputs.pop("soft_labels", None)

        outputs = model(**inputs)
        logits = outputs.logits

        if self.label_conversion == "soft_label":
            if soft_labels is None:
                msg = "soft_label training requires soft_labels in batch"
                raise ValueError(msg)
            safe_temp = max(float(self.soft_label_temperature), 1e-6)
            log_probs = F.log_softmax(logits / safe_temp, dim=-1)
            loss = -(soft_labels * log_probs).sum(dim=-1).mean()
        else:
            loss_fct = nn.CrossEntropyLoss(weight=self.class_weights)
            loss = loss_fct(logits, labels)

        return (loss, outputs) if return_outputs else loss


def _resolve_class_weights(
    config: ClassifierConfig,
    train_labels: np.ndarray,
    device: torch.device,
) -> tuple[torch.Tensor | None, dict[str, Any]]:
    counts = np.bincount(train_labels.astype(np.int64), minlength=NUM_JVNV_EMOTIONS)

    if config.train.class_weight_mode == "none":
        return None, {"mode": "none", "weights": None, "class_counts": {}}

    if config.train.class_weight_mode == "manual":
        weights = np.asarray(config.train.class_weights, dtype=np.float32)
    elif config.train.class_weight_mode == "inverse_frequency":
        weights = 1.0 / np.maximum(counts.astype(np.float32), 1.0)
        weights /= max(float(weights.mean()), 1e-8)
    else:
        msg = f"Unknown class_weight_mode: {config.train.class_weight_mode}"
        raise ValueError(msg)

    tensor = torch.tensor(weights, dtype=torch.float32, device=device)
    summary = {
        "mode": config.train.class_weight_mode,
        "weights": {
            label: float(value)
            for label, value in zip(JVNV_EMOTION_LABELS, weights, strict=True)
        },
    }
    return tensor, summary


def _create_compute_metrics():
    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        return compute_classification_metrics(logits, labels, JVNV_EMOTION_LABELS)

    return compute_metrics


def _load_transfer_weights(
    model: TextEmotionClassifier,
    transfer_from: str,
) -> dict[str, Any]:
    path = Path(transfer_from)
    if not path.exists():
        msg = f"transfer checkpoint not found: {path}"
        raise FileNotFoundError(msg)

    # Trainer で保存されたディレクトリ形式から重みをロード
    from transformers import AutoModelForSequenceClassification

    temp_model = AutoModelForSequenceClassification.from_pretrained(str(path))
    state_dict = temp_model.state_dict()

    current = model.state_dict()
    compatible = {
        key: value
        for key, value in state_dict.items()
        if key in current and current[key].shape == value.shape
    }

    missing, unexpected = model.load_state_dict(compatible, strict=False)
    return {
        "path": str(path),
        "loaded_keys": len(compatible),
        "missing_keys": len(missing),
        "unexpected_keys": len(unexpected),
    }


def train_classifier(config: ClassifierConfig) -> dict[str, Any] | None:
    set_seed(config.data.random_seed)

    output_dir = Path(config.train.output_dir)
    reports_dir = output_dir / "reports"
    output_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(config.model.pretrained_model_name)
    splits, metadata = build_classifier_splits(config.data)

    train_dataset = EmotionClassifierTextDataset(
        splits["train"].texts,
        splits["train"].labels,
        splits["train"].soft_targets,
    )
    val_dataset = EmotionClassifierTextDataset(
        splits["val"].texts,
        splits["val"].labels,
        splits["val"].soft_targets,
    )
    test_dataset = EmotionClassifierTextDataset(
        splits["test"].texts,
        splits["test"].labels,
        splits["test"].soft_targets,
    )

    id2label = {i: label for i, label in enumerate(JVNV_EMOTION_LABELS)}
    label2id = {label: i for i, label in enumerate(JVNV_EMOTION_LABELS)}

    model = TextEmotionClassifier(
        pretrained_model_name=config.model.pretrained_model_name,
        num_classes=NUM_JVNV_EMOTIONS,
        dropout=config.model.dropout,
        id2label=id2label,
        label2id=label2id,
    )

    transfer_summary: dict[str, Any] | None = None
    if config.model.transfer_from:
        transfer_summary = _load_transfer_weights(model, config.model.transfer_from)

    class_weights, class_weight_summary = _resolve_class_weights(
        config,
        splits["train"].labels,
        torch.device("cpu"),  # Trainer が適切に移動させるため一旦CPUで作成
    )

    training_args = TrainingArguments(
        output_dir=str(output_dir / "checkpoints"),
        evaluation_strategy="epoch",
        save_strategy="epoch",
        logging_steps=config.train.log_every_steps,
        learning_rate=config.train.head_lr,  # デフォルト。後で最適化グループを調整可能
        per_device_train_batch_size=config.train.batch_size,
        per_device_eval_batch_size=config.train.batch_size,
        num_train_epochs=config.train.num_epochs,
        weight_decay=config.train.weight_decay,
        warmup_ratio=config.train.warmup_ratio,
        load_best_model_at_end=True,
        metric_for_best_model="eval_macro_f1",
        greater_is_better=True,
        fp16=(config.train.mixed_precision == "fp16"),
        bf16=(config.train.mixed_precision == "bf16"),
        dataloader_num_workers=config.train.num_workers,
        dataloader_pin_memory=config.train.pin_memory,
        gradient_accumulation_steps=config.train.gradient_accumulation_steps,
        report_to=["tensorboard"],
    )

    # 異なる学習率の設定
    optimizer_grouped_parameters = [
        {
            "params": model.get_encoder_parameters(),
            "lr": config.train.bert_lr,
        },
        {
            "params": model.get_head_parameters(),
            "lr": config.train.head_lr,
        },
    ]
    from torch.optim import AdamW

    optimizer = AdamW(optimizer_grouped_parameters, weight_decay=config.train.weight_decay)

    trainer = EmotionTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=ClassifierBatchCollator(tokenizer, config.data.max_length),
        compute_metrics=_create_compute_metrics(),
        callbacks=[
            EarlyStoppingCallback(
                early_stopping_patience=config.train.early_stopping_patience,
            ),
        ],
        optimizers=(optimizer, None),  # スケジューラは Trainer に任せる
        class_weights=class_weights,
        label_conversion=config.data.label_conversion,
        soft_label_temperature=config.data.soft_label_temperature,
    )

    trainer.train()

    # テストセットでの評価
    test_metrics = trainer.evaluate(test_dataset, metric_key_prefix="test")

    # Go/No-Go 判定
    go_no_go = _go_no_go_classifier(test_metrics, config)

    # モデル保存
    trainer.save_model(str(output_dir / "checkpoints" / "best_model"))

    # メタデータ保存
    if trainer.is_main_process:
        save_effective_config(config, output_dir / "effective_config.yaml")
        data_report = build_classifier_data_report(splits, metadata)
        with (reports_dir / "evaluation.json").open("w", encoding="utf-8") as f:
            import json

            json.dump(
                {
                    "metrics": test_metrics,
                    "go_no_go": go_no_go,
                    "class_weighting": class_weight_summary,
                    "transfer": transfer_summary,
                },
                f,
                ensure_ascii=False,
                indent=2,
            )

    return {
        "output_dir": str(output_dir),
        "checkpoint": str(output_dir / "checkpoints" / "best_model"),
        "metrics": test_metrics,
        "go_no_go": go_no_go,
        "transfer": transfer_summary,
    }


def _go_no_go_classifier(metrics: dict[str, Any], config: ClassifierConfig) -> dict[str, Any]:
    key_emotions = config.eval.key_emotions or KEY_EMOTION_LABELS
    macro_f1 = metrics.get("test_macro_f1", metrics.get("eval_macro_f1", 0.0))

    checks: dict[str, bool] = {
        "macro_f1": macro_f1 >= config.eval.go_macro_f1_min,
    }
    per_class = metrics.get("test_per_class_f1", metrics.get("eval_per_class_f1", {}))
    for emotion in key_emotions:
        checks[f"{emotion}_f1"] = (
            per_class.get(emotion, 0.0) >= config.eval.go_key_emotion_f1_min
        )

    return {
        **checks,
        "go": all(checks.values()),
        "values": {"macro_f1": macro_f1, "key_emotion_f1": per_class},
    }
