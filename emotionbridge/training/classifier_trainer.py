import json
import logging
import math
import random
from dataclasses import asdict
from pathlib import Path
from typing import Any, cast

import numpy as np
import torch
import torch.nn.functional as F
from accelerate import Accelerator
from torch import nn
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torch.utils.tensorboard import SummaryWriter
from tqdm.auto import tqdm
from transformers import AutoTokenizer, get_linear_schedule_with_warmup

from emotionbridge.config import Phase0ClassifierConfig, save_effective_config
from emotionbridge.constants import (
    JVNV_EMOTION_LABELS,
    KEY_EMOTION_LABELS,
    NUM_JVNV_EMOTIONS,
)
from emotionbridge.data import (
    build_classifier_data_report,
    build_phase0_classifier_splits,
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
        encoded["labels"] = torch.tensor(labels, dtype=torch.long)

        if all("soft_target" in sample for sample in batch):
            soft_targets = np.asarray(
                [sample["soft_target"] for sample in batch],
                dtype=np.float32,
            )
            encoded["soft_labels"] = torch.tensor(soft_targets, dtype=torch.float32)

        return encoded


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _to_scalar(value: Any) -> float:
    if isinstance(value, torch.Tensor):
        return float(value.detach().cpu().float().mean())
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, list):
        if not value:
            return 0.0
        return float(np.mean([_to_scalar(element) for element in value]))
    if isinstance(value, dict):
        if not value:
            return 0.0
        return float(np.mean([_to_scalar(element) for element in value.values()]))
    msg = f"Unsupported scalar conversion type: {type(value)!r}"
    raise TypeError(msg)


def _soft_cross_entropy(
    logits: torch.Tensor,
    soft_targets: torch.Tensor,
    temperature: float,
) -> torch.Tensor:
    safe_temperature = max(float(temperature), 1e-6)
    log_probs = F.log_softmax(logits / safe_temperature, dim=-1)
    return -(soft_targets * log_probs).sum(dim=-1).mean()


def _compute_loss(
    *,
    logits: torch.Tensor,
    labels: torch.Tensor,
    soft_labels: torch.Tensor | None,
    criterion: nn.CrossEntropyLoss | None,
    label_conversion: str,
    temperature: float,
) -> torch.Tensor:
    if label_conversion == "soft_label":
        if soft_labels is None:
            msg = "soft_label training requires soft_labels in batch"
            raise ValueError(msg)
        return _soft_cross_entropy(logits, soft_labels, temperature)

    if criterion is None:
        msg = "CrossEntropy criterion is not initialized"
        raise ValueError(msg)
    return criterion(logits, labels)


def _resolve_class_weights(
    config: Phase0ClassifierConfig,
    train_labels: np.ndarray,
    device: torch.device,
) -> tuple[torch.Tensor | None, dict[str, Any]]:
    counts = np.bincount(train_labels.astype(np.int64), minlength=NUM_JVNV_EMOTIONS)

    if config.train.class_weight_mode == "none":
        return (
            None,
            {
                "mode": "none",
                "weights": None,
                "class_counts": {
                    label: int(count)
                    for label, count in zip(JVNV_EMOTION_LABELS, counts, strict=True)
                },
            },
        )

    if config.train.class_weight_mode == "manual":
        if config.train.class_weights is None:
            msg = "train.class_weights is required when class_weight_mode=manual"
            raise ValueError(msg)

        weights = np.asarray(config.train.class_weights, dtype=np.float32)
        if weights.shape != (NUM_JVNV_EMOTIONS,):
            msg = (
                "train.class_weights length must match number of classes: "
                f"{NUM_JVNV_EMOTIONS}"
            )
            raise ValueError(msg)
        if np.any(weights <= 0):
            msg = "train.class_weights must be all positive"
            raise ValueError(msg)

    elif config.train.class_weight_mode == "inverse_frequency":
        weights = 1.0 / np.maximum(counts.astype(np.float32), 1.0)
        weights /= max(float(weights.mean()), 1e-8)
    else:
        msg = "train.class_weight_mode must be one of: none, inverse_frequency, manual"
        raise ValueError(msg)

    tensor = torch.tensor(weights, dtype=torch.float32, device=device)
    summary = {
        "mode": config.train.class_weight_mode,
        "weights": {
            label: float(value)
            for label, value in zip(JVNV_EMOTION_LABELS, weights, strict=True)
        },
        "class_counts": {
            label: int(count)
            for label, count in zip(JVNV_EMOTION_LABELS, counts, strict=True)
        },
    }
    return tensor, summary


def _build_weighted_sampler(labels: np.ndarray) -> WeightedRandomSampler:
    counts = np.bincount(labels.astype(np.int64), minlength=NUM_JVNV_EMOTIONS)
    class_weights = 1.0 / np.maximum(counts.astype(np.float64), 1.0)
    sample_weights = class_weights[labels.astype(np.int64)]
    return WeightedRandomSampler(
        weights=torch.tensor(sample_weights, dtype=torch.double),
        num_samples=len(labels),
        replacement=True,
    )


def _create_dataloaders(config: Phase0ClassifierConfig, tokenizer: Any):
    splits, metadata = build_phase0_classifier_splits(config.data)
    collator = ClassifierBatchCollator(
        tokenizer=tokenizer,
        max_length=config.data.max_length,
    )

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

    sampler: WeightedRandomSampler | None = None
    if config.train.use_weighted_sampler:
        sampler = _build_weighted_sampler(splits["train"].labels)

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.train.batch_size,
        shuffle=sampler is None,
        sampler=sampler,
        num_workers=config.train.num_workers,
        pin_memory=config.train.pin_memory,
        collate_fn=collator,
    )
    loaders = {
        "train": train_loader,
        "val": DataLoader(
            val_dataset,
            batch_size=config.train.batch_size,
            shuffle=False,
            num_workers=config.train.num_workers,
            pin_memory=config.train.pin_memory,
            collate_fn=collator,
        ),
        "test": DataLoader(
            test_dataset,
            batch_size=config.train.batch_size,
            shuffle=False,
            num_workers=config.train.num_workers,
            pin_memory=config.train.pin_memory,
            collate_fn=collator,
        ),
    }

    sampler_info = {
        "enabled": config.train.use_weighted_sampler,
        "type": "WeightedRandomSampler" if sampler is not None else "shuffle",
    }
    return loaders, splits, metadata, sampler_info


def _build_progress_bar(
    iterable: Any,
    *,
    total: int,
    desc: str,
    enabled: bool,
):
    if not enabled:
        return iterable, None

    bar = tqdm(
        iterable,
        total=total,
        desc=desc,
        dynamic_ncols=True,
        leave=False,
    )
    return bar, bar


def _safe_set_postfix(
    progress_bar: Any,
    values: dict[str, str],
) -> None:
    if progress_bar is None:
        return
    progress_bar.set_postfix(values, refresh=False)


def _run_eval(
    model: torch.nn.Module,
    data_loader: DataLoader,
    accelerator: Accelerator,
    *,
    criterion: nn.CrossEntropyLoss | None,
    label_conversion: str,
    soft_label_temperature: float,
    progress_desc: str,
    show_progress: bool,
) -> tuple[float, np.ndarray, np.ndarray]:
    model.eval()
    losses: list[float] = []
    all_logits: list[np.ndarray] = []
    all_labels: list[np.ndarray] = []

    eval_iter, eval_bar = _build_progress_bar(
        data_loader,
        total=len(data_loader),
        desc=progress_desc,
        enabled=show_progress,
    )

    with torch.no_grad():
        for batch in eval_iter:
            labels = cast("torch.Tensor", batch.pop("labels"))
            soft_labels = cast("torch.Tensor | None", batch.pop("soft_labels", None))

            logits = cast("torch.Tensor", model(**batch))
            loss = _compute_loss(
                logits=logits,
                labels=labels,
                soft_labels=soft_labels,
                criterion=criterion,
                label_conversion=label_conversion,
                temperature=soft_label_temperature,
            )
            reduced_loss = accelerator.reduce(loss.detach(), reduction="mean")
            reduced_loss_scalar = _to_scalar(reduced_loss)
            losses.append(reduced_loss_scalar)
            _safe_set_postfix(eval_bar, {"loss": f"{reduced_loss_scalar:.4f}"})

            gathered_logits, gathered_labels = accelerator.gather_for_metrics(
                (logits.detach(), labels.detach()),
            )
            all_logits.append(cast("torch.Tensor", gathered_logits).cpu().numpy())
            all_labels.append(cast("torch.Tensor", gathered_labels).cpu().numpy())

    if eval_bar is not None:
        eval_bar.close()

    avg_loss = float(np.mean(losses)) if losses else 0.0
    logits = np.vstack(all_logits) if all_logits else np.zeros((0, NUM_JVNV_EMOTIONS))
    labels = (
        np.concatenate(all_labels) if all_labels else np.zeros((0,), dtype=np.int64)
    )
    return avg_loss, logits, labels.astype(np.int64)


def _go_no_go_classifier(
    metrics: dict[str, Any],
    config: Phase0ClassifierConfig,
) -> dict[str, Any]:
    key_emotions = config.eval.key_emotions or KEY_EMOTION_LABELS

    checks: dict[str, bool] = {
        "macro_f1": metrics["macro_f1"] >= config.eval.go_macro_f1_min,
    }
    for emotion in key_emotions:
        checks[f"{emotion}_f1"] = (
            metrics["per_class_f1"].get(emotion, 0.0)
            >= config.eval.go_key_emotion_f1_min
        )

    values = {
        "macro_f1": metrics["macro_f1"],
        "accuracy": metrics["accuracy"],
        "key_emotion_f1": {
            emotion: metrics["per_class_f1"].get(emotion, 0.0)
            for emotion in key_emotions
        },
    }

    return {
        **checks,
        "go": all(checks.values()),
        "thresholds": {
            "macro_f1_min": config.eval.go_macro_f1_min,
            "key_emotion_f1_min": config.eval.go_key_emotion_f1_min,
            "key_emotions": key_emotions,
        },
        "values": values,
    }


def _validate_settings(config: Phase0ClassifierConfig) -> None:
    if config.train.gradient_accumulation_steps < 1:
        msg = "train.gradient_accumulation_steps must be >= 1"
        raise ValueError(msg)

    if config.train.mixed_precision not in {"no", "fp16", "bf16"}:
        msg = "train.mixed_precision must be one of: no, fp16, bf16"
        raise ValueError(msg)

    if config.train.device not in {"cpu", "cuda"}:
        msg = "train.device must be either 'cpu' or 'cuda'"
        raise ValueError(msg)

    if config.train.class_weight_mode not in {"none", "inverse_frequency", "manual"}:
        msg = "train.class_weight_mode must be one of: none, inverse_frequency, manual"
        raise ValueError(msg)

    if config.data.label_conversion not in {"argmax", "soft_label"}:
        msg = "data.label_conversion must be one of: argmax, soft_label"
        raise ValueError(msg)

    if config.model.num_classes != NUM_JVNV_EMOTIONS:
        msg = f"model.num_classes must match JVNV class count: {NUM_JVNV_EMOTIONS}"
        raise ValueError(msg)


def _load_transfer_weights(
    model: TextEmotionClassifier,
    transfer_from: str,
) -> dict[str, Any]:
    path = Path(transfer_from)
    if not path.exists():
        msg = f"transfer checkpoint not found: {path}"
        raise FileNotFoundError(msg)

    checkpoint = torch.load(path, map_location="cpu")
    state_dict = checkpoint.get("model_state_dict")
    if not isinstance(state_dict, dict):
        msg = "model_state_dict not found in transfer checkpoint"
        raise ValueError(msg)

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


def train_phase0_classifier(config: Phase0ClassifierConfig) -> dict[str, Any] | None:
    _validate_settings(config)
    _set_seed(config.data.random_seed)

    accelerator = Accelerator(
        cpu=config.train.device == "cpu",
        mixed_precision=config.train.mixed_precision,
        gradient_accumulation_steps=config.train.gradient_accumulation_steps,
    )

    output_dir = Path(config.train.output_dir)
    checkpoints_dir = output_dir / "checkpoints"
    reports_dir = output_dir / "reports"
    tb_dir = output_dir / "tensorboard"

    if accelerator.is_main_process:
        checkpoints_dir.mkdir(parents=True, exist_ok=True)
        reports_dir.mkdir(parents=True, exist_ok=True)

    accelerator.wait_for_everyone()

    writer: SummaryWriter | None = None
    if accelerator.is_main_process:
        writer = SummaryWriter(log_dir=str(tb_dir))

    tokenizer = AutoTokenizer.from_pretrained(config.model.pretrained_model_name)
    loaders, splits, metadata, sampler_info = _create_dataloaders(config, tokenizer)
    data_report = build_classifier_data_report(splits, metadata)
    data_report["sampler"] = sampler_info

    model = TextEmotionClassifier(
        pretrained_model_name=config.model.pretrained_model_name,
        num_classes=config.model.num_classes,
        bottleneck_dim=config.model.bottleneck_dim,
        dropout=config.model.dropout,
    )

    transfer_summary: dict[str, Any] | None = None
    if config.model.transfer_from:
        transfer_summary = _load_transfer_weights(model, config.model.transfer_from)

    optimizer = AdamW(
        [
            {"params": model.encoder.parameters(), "lr": config.train.bert_lr},
            {"params": model.head_parameters(), "lr": config.train.head_lr},
        ],
        weight_decay=config.train.weight_decay,
    )

    model, optimizer, loaders["train"], loaders["val"], loaders["test"] = (
        accelerator.prepare(
            model,
            optimizer,
            loaders["train"],
            loaders["val"],
            loaders["test"],
        )
    )

    num_update_steps_per_epoch = max(
        1,
        math.ceil(
            len(loaders["train"]) / max(1, config.train.gradient_accumulation_steps),
        ),
    )
    total_steps = max(1, num_update_steps_per_epoch * config.train.num_epochs)
    warmup_steps = int(total_steps * config.train.warmup_ratio)
    scheduler = get_linear_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )

    class_weights, class_weighting = _resolve_class_weights(
        config,
        splits["train"].labels,
        accelerator.device,
    )
    data_report["class_weighting"] = class_weighting

    criterion: nn.CrossEntropyLoss | None = None
    if config.data.label_conversion == "argmax":
        criterion = nn.CrossEntropyLoss(weight=class_weights)

    best_val_loss = float("inf")
    best_path = checkpoints_dir / "best_model.pt"
    early_stop_count = 0
    history: list[dict[str, float]] = []
    global_step = 0

    if accelerator.is_main_process:
        logger.info(
            "分類器学習開始: epochs=%d, train_batches=%d, val_batches=%d, test_batches=%d, device=%s, mixed_precision=%s",
            config.train.num_epochs,
            len(loaders["train"]),
            len(loaders["val"]),
            len(loaders["test"]),
            accelerator.device,
            config.train.mixed_precision,
        )

    for epoch in range(1, config.train.num_epochs + 1):
        model.train()
        train_losses: list[float] = []
        optimizer.zero_grad(set_to_none=True)

        train_iter, train_bar = _build_progress_bar(
            loaders["train"],
            total=len(loaders["train"]),
            desc=f"Epoch {epoch}/{config.train.num_epochs} [train]",
            enabled=accelerator.is_main_process,
        )

        for batch in train_iter:
            labels = cast("torch.Tensor", batch.pop("labels"))
            soft_labels = cast("torch.Tensor | None", batch.pop("soft_labels", None))

            with accelerator.accumulate(model):
                logits = cast("torch.Tensor", model(**batch))
                loss = _compute_loss(
                    logits=logits,
                    labels=labels,
                    soft_labels=soft_labels,
                    criterion=criterion,
                    label_conversion=config.data.label_conversion,
                    temperature=config.data.soft_label_temperature,
                )

                accelerator.backward(loss)

                if accelerator.sync_gradients:
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad(set_to_none=True)

                    global_step += 1
                    if (
                        writer is not None
                        and global_step % config.train.log_every_steps == 0
                    ):
                        writer.add_scalar(
                            "train/step_loss",
                            float(loss.item()),
                            global_step,
                        )
                        writer.add_scalar(
                            "train/lr_bert",
                            optimizer.param_groups[0]["lr"],
                            global_step,
                        )
                        writer.add_scalar(
                            "train/lr_head",
                            optimizer.param_groups[1]["lr"],
                            global_step,
                        )

            reduced_step_loss = accelerator.reduce(loss.detach(), reduction="mean")
            reduced_step_loss_scalar = _to_scalar(reduced_step_loss)
            train_losses.append(reduced_step_loss_scalar)
            _safe_set_postfix(
                train_bar,
                {
                    "loss": f"{reduced_step_loss_scalar:.4f}",
                    "lr_head": f"{optimizer.param_groups[1]['lr']:.2e}",
                },
            )

        if train_bar is not None:
            train_bar.close()

        val_loss, val_logits, val_labels = _run_eval(
            model,
            loaders["val"],
            accelerator,
            criterion=criterion,
            label_conversion=config.data.label_conversion,
            soft_label_temperature=config.data.soft_label_temperature,
            progress_desc=f"Epoch {epoch}/{config.train.num_epochs} [val]",
            show_progress=accelerator.is_main_process,
        )
        epoch_train_loss = float(np.mean(train_losses)) if train_losses else 0.0

        val_metrics = compute_classification_metrics(
            val_logits,
            val_labels,
            JVNV_EMOTION_LABELS,
        )

        if writer is not None:
            writer.add_scalars(
                "loss/epoch",
                {"train": epoch_train_loss, "val": val_loss},
                epoch,
            )
            writer.add_scalar("val/accuracy", val_metrics["accuracy"], epoch)
            writer.add_scalar("val/macro_f1", val_metrics["macro_f1"], epoch)
            for emotion in JVNV_EMOTION_LABELS:
                writer.add_scalar(
                    f"val/f1_{emotion}",
                    val_metrics["per_class_f1"][emotion],
                    epoch,
                )

        if accelerator.is_main_process:
            history.append(
                {
                    "epoch": float(epoch),
                    "train_loss": epoch_train_loss,
                    "val_loss": val_loss,
                    "val_accuracy": float(val_metrics["accuracy"]),
                    "val_macro_f1": float(val_metrics["macro_f1"]),
                    "global_step": float(global_step),
                },
            )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            early_stop_count = 0
            if accelerator.is_main_process:
                accelerator.save(
                    {
                        "model_type": "classifier",
                        "model_state_dict": accelerator.get_state_dict(model),
                        "model_config": asdict(config.model),
                        "tokenizer_name": config.model.pretrained_model_name,
                        "max_length": config.data.max_length,
                        "emotion_labels": JVNV_EMOTION_LABELS,
                        "label_conversion": config.data.label_conversion,
                        "config": config.to_dict(),
                    },
                    best_path,
                )
        else:
            early_stop_count += 1

        if early_stop_count >= config.train.early_stopping_patience:
            break

        if accelerator.is_main_process:
            logger.info(
                "Epoch %d/%d | train_loss=%.4f val_loss=%.4f val_acc=%.4f val_macro_f1=%.4f best_val_loss=%.4f",
                epoch,
                config.train.num_epochs,
                epoch_train_loss,
                val_loss,
                float(val_metrics["accuracy"]),
                float(val_metrics["macro_f1"]),
                best_val_loss,
            )

    accelerator.wait_for_everyone()
    checkpoint = torch.load(best_path, map_location=accelerator.device)
    accelerator.unwrap_model(model).load_state_dict(checkpoint["model_state_dict"])

    test_loss, test_logits, test_labels = _run_eval(
        model,
        loaders["test"],
        accelerator,
        criterion=criterion,
        label_conversion=config.data.label_conversion,
        soft_label_temperature=config.data.soft_label_temperature,
        progress_desc="[test]",
        show_progress=accelerator.is_main_process,
    )

    metrics = compute_classification_metrics(
        test_logits,
        test_labels,
        JVNV_EMOTION_LABELS,
    )
    metrics["test_loss"] = test_loss
    go_no_go = _go_no_go_classifier(metrics, config)

    if accelerator.is_main_process:
        logger.info(
            "分類器学習完了: test_loss=%.4f accuracy=%.4f macro_f1=%.4f go=%s",
            float(test_loss),
            float(metrics["accuracy"]),
            float(metrics["macro_f1"]),
            go_no_go["go"],
        )

    if writer is not None:
        writer.add_scalar("test/accuracy", metrics["accuracy"], 0)
        writer.add_scalar("test/macro_f1", metrics["macro_f1"], 0)
        for emotion in JVNV_EMOTION_LABELS:
            writer.add_scalar(
                f"test/f1_{emotion}",
                metrics["per_class_f1"][emotion],
                0,
            )
        writer.close()

    if not accelerator.is_main_process:
        return None

    save_effective_config(config, output_dir / "effective_config.yaml")

    with (reports_dir / "data_report.json").open("w", encoding="utf-8") as file:
        json.dump(data_report, file, ensure_ascii=False, indent=2)

    with (reports_dir / "training_history.json").open("w", encoding="utf-8") as file:
        json.dump(history, file, ensure_ascii=False, indent=2)

    with (reports_dir / "evaluation.json").open("w", encoding="utf-8") as file:
        json.dump(
            {
                "metrics": metrics,
                "go_no_go": go_no_go,
                "class_weighting": class_weighting,
                "transfer": transfer_summary,
            },
            file,
            ensure_ascii=False,
            indent=2,
        )

    return {
        "output_dir": str(output_dir),
        "checkpoint": str(best_path),
        "device": str(accelerator.device),
        "data_report": data_report,
        "history": history,
        "metrics": metrics,
        "go_no_go": go_no_go,
        "class_weighting": class_weighting,
        "transfer": transfer_summary,
    }
