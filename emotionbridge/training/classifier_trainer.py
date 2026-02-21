import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.optim import AdamW
from torch.utils.data import Dataset
from transformers import (
    AutoModelForSequenceClassification,
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
            class_weights = self.class_weights
            if class_weights is not None and (
                class_weights.device != logits.device or class_weights.dtype != logits.dtype
            ):
                class_weights = class_weights.to(device=logits.device, dtype=logits.dtype)
                self.class_weights = class_weights
            loss_fct = nn.CrossEntropyLoss(weight=class_weights)
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
            label: float(value) for label, value in zip(JVNV_EMOTION_LABELS, weights, strict=True)
        },
    }
    return tensor, summary


def _create_compute_metrics():
    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        return compute_classification_metrics(logits, labels, JVNV_EMOTION_LABELS)

    return compute_metrics


def _load_transfer_weights(
    model: nn.Module,
    transfer_from: str,
) -> dict[str, Any]:
    path = Path(transfer_from)
    if not path.exists():
        msg = f"transfer checkpoint not found: {path}"
        raise FileNotFoundError(msg)

    # Trainer で保存されたディレクトリ形式から重みをロード
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

    id2label = dict(enumerate(JVNV_EMOTION_LABELS))
    label2id = {label: i for i, label in enumerate(JVNV_EMOTION_LABELS)}

    model = AutoModelForSequenceClassification.from_pretrained(
        config.model.pretrained_model_name,
        num_labels=NUM_JVNV_EMOTIONS,
        id2label=id2label,
        label2id=label2id,
        classifier_dropout=config.model.dropout,
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
        eval_strategy="epoch",
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
        remove_unused_columns=False,
    )

    # 異なる学習率の設定
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
    if trainer.is_world_process_zero():
        save_effective_config(config, output_dir / "effective_config.yaml")
        build_classifier_data_report(splits, metadata)
        with (reports_dir / "evaluation.json").open("w", encoding="utf-8") as f:
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
        checks[f"{emotion}_f1"] = per_class.get(emotion, 0.0) >= config.eval.go_key_emotion_f1_min

    return {
        **checks,
        "go": all(checks.values()),
        "values": {"macro_f1": macro_f1, "key_emotion_f1": per_class},
    }
