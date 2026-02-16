import json
import math
import random
from dataclasses import asdict
from pathlib import Path
from typing import Any, cast

import numpy as np
import torch
import torch.nn.functional as F
from accelerate import Accelerator
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter
from transformers import AutoTokenizer, get_linear_schedule_with_warmup

from emotionbridge.config import Phase0Config, save_effective_config
from emotionbridge.constants import (
    EMOTION_LABELS,
    LOW_VARIANCE_EMOTION_LABELS,
    MAJOR_EMOTION_LABELS,
)
from emotionbridge.data import (
    build_data_report,
    build_phase0_splits,
    estimate_unk_ratio,
    plot_confusion_matrix,
    plot_per_emotion_error,
)
from emotionbridge.model import TextEmotionRegressor
from emotionbridge.training.metrics import compute_regression_metrics


class EmotionTextDataset(Dataset):
    def __init__(self, texts: list[str], targets: np.ndarray) -> None:
        self._texts = texts
        self._targets = targets

    def __len__(self) -> int:
        return len(self._texts)

    def __getitem__(self, index: int) -> dict[str, Any]:
        return {
            "text": self._texts[index],
            "target": self._targets[index],
        }


class TextBatchCollator:
    def __init__(self, tokenizer: Any, max_length: int) -> None:
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __call__(self, batch: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        texts = [sample["text"] for sample in batch]
        targets = np.asarray([sample["target"] for sample in batch], dtype=np.float32)
        encoded = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        encoded["labels"] = torch.tensor(targets, dtype=torch.float32)
        return encoded


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _weighted_mse(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    weights: torch.Tensor | None,
) -> torch.Tensor:
    if weights is None:
        return F.mse_loss(predictions, targets)
    squared_error = torch.square(predictions - targets)
    return (squared_error * weights).mean()


def _resolve_loss_weights(
    config: Phase0Config,
    train_targets: np.ndarray,
    device: torch.device,
) -> tuple[torch.Tensor | None, dict[str, Any]]:
    if train_targets.ndim != 2 or train_targets.shape[1] != len(EMOTION_LABELS):
        msg = (
            "train targets must be 2D and match emotion dims: "
            f"(*, {len(EMOTION_LABELS)})"
        )
        raise ValueError(msg)

    train_mean_intensity = train_targets.mean(axis=0).astype(np.float32)

    if config.train.emotion_weights is not None:
        weight_array = np.asarray(config.train.emotion_weights, dtype=np.float32)
        if weight_array.shape != (len(EMOTION_LABELS),):
            msg = (
                "train.emotion_weights length must match number of emotions: "
                f"{len(EMOTION_LABELS)}"
            )
            raise ValueError(msg)
        if np.any(weight_array <= 0):
            msg = "train.emotion_weights must be all positive"
            raise ValueError(msg)
        mode = "manual"
    elif config.train.emotion_weight_mode == "none":
        return (
            None,
            {
                "mode": "none",
                "weights": None,
                "mean_intensity": {
                    emotion: round(float(value), 6)
                    for emotion, value in zip(
                        EMOTION_LABELS,
                        train_mean_intensity,
                        strict=True,
                    )
                },
            },
        )
    elif config.train.emotion_weight_mode == "inverse_mean":
        safe_mean = np.maximum(
            train_mean_intensity,
            config.train.emotion_weight_epsilon,
        )
        weight_array = 1.0 / safe_mean
        if config.train.emotion_weight_normalize:
            mean_weight = max(
                float(weight_array.mean()),
                config.train.emotion_weight_epsilon,
            )
            weight_array /= mean_weight
        mode = "inverse_mean"
    else:
        msg = "train.emotion_weight_mode must be one of: none, inverse_mean"
        raise ValueError(msg)

    weight_tensor = torch.tensor(weight_array, dtype=torch.float32, device=device).view(
        1,
        -1,
    )
    summary: dict[str, Any] = {
        "mode": mode,
        "weights": {
            emotion: round(float(value), 6)
            for emotion, value in zip(EMOTION_LABELS, weight_array, strict=True)
        },
        "mean_intensity": {
            emotion: round(float(value), 6)
            for emotion, value in zip(EMOTION_LABELS, train_mean_intensity, strict=True)
        },
    }
    if mode == "inverse_mean":
        summary["epsilon"] = config.train.emotion_weight_epsilon
        summary["normalize"] = config.train.emotion_weight_normalize
    return weight_tensor, summary


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


def _create_dataloaders(config: Phase0Config, tokenizer: Any):
    splits, metadata = build_phase0_splits(config.data)
    collator = TextBatchCollator(tokenizer=tokenizer, max_length=config.data.max_length)

    train_dataset = EmotionTextDataset(splits["train"].texts, splits["train"].targets)
    val_dataset = EmotionTextDataset(splits["val"].texts, splits["val"].targets)
    test_dataset = EmotionTextDataset(splits["test"].texts, splits["test"].targets)

    loaders = {
        "train": DataLoader(
            train_dataset,
            batch_size=config.train.batch_size,
            shuffle=True,
            num_workers=config.train.num_workers,
            pin_memory=config.train.pin_memory,
            collate_fn=collator,
        ),
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
    return loaders, splits, metadata


def _run_eval(
    model: torch.nn.Module,
    data_loader: DataLoader,
    accelerator: Accelerator,
    weights: torch.Tensor | None,
) -> tuple[float, np.ndarray, np.ndarray]:
    model.eval()
    losses: list[float] = []
    all_predictions: list[np.ndarray] = []
    all_targets: list[np.ndarray] = []

    with torch.no_grad():
        for batch in data_loader:
            labels = cast("torch.Tensor", batch.pop("labels"))

            predictions = cast("torch.Tensor", model(**batch))
            loss: torch.Tensor = _weighted_mse(predictions, labels, weights)
            reduced_loss = accelerator.reduce(loss.detach(), reduction="mean")
            losses.append(_to_scalar(reduced_loss))

            gathered_predictions, gathered_targets = accelerator.gather_for_metrics(
                (predictions.detach(), labels.detach()),
            )

            all_predictions.append(
                cast("torch.Tensor", gathered_predictions).cpu().numpy(),
            )
            all_targets.append(cast("torch.Tensor", gathered_targets).cpu().numpy())

    avg_loss = float(np.mean(losses)) if losses else 0.0
    stacked_predictions = (
        np.vstack(all_predictions)
        if all_predictions
        else np.zeros((0, len(EMOTION_LABELS)))
    )
    stacked_targets = (
        np.vstack(all_targets) if all_targets else np.zeros((0, len(EMOTION_LABELS)))
    )
    return avg_loss, stacked_predictions, stacked_targets


def _go_no_go(metrics: dict[str, Any], config: Phase0Config) -> dict[str, Any]:
    checks = {
        "macro_mse": metrics["mse_macro"] <= config.eval.go_macro_mse_max,
        "top6_min_pearson": (
            metrics["pearson_top6_min"] >= config.eval.go_top6_min_pearson
        ),
        "anger_trust_min_pearson": (
            metrics["pearson_anger_trust_min"] >= config.eval.go_anger_trust_min_pearson
        ),
        "top1_accuracy": metrics["top1_accuracy"] >= config.eval.go_top1_acc_min,
    }
    return {
        **checks,
        "go": all(checks.values()),
        "thresholds": {
            "macro_mse_max": config.eval.go_macro_mse_max,
            "top6_min_pearson": config.eval.go_top6_min_pearson,
            "anger_trust_min_pearson": config.eval.go_anger_trust_min_pearson,
            "top1_accuracy_min": config.eval.go_top1_acc_min,
        },
        "values": {
            "mse_macro": metrics["mse_macro"],
            "pearson_top6_min": metrics["pearson_top6_min"],
            "pearson_anger_trust_min": metrics["pearson_anger_trust_min"],
            "top1_accuracy": metrics["top1_accuracy"],
        },
        "emotion_groups": {
            "top6": MAJOR_EMOTION_LABELS,
            "anger_trust": LOW_VARIANCE_EMOTION_LABELS,
        },
    }


def _log_hparams(
    writer: SummaryWriter,
    config: Phase0Config,
    metrics: dict[str, Any],
) -> None:
    hparam_dict = {
        "bert_lr": config.train.bert_lr,
        "head_lr": config.train.head_lr,
        "batch_size": config.train.batch_size,
        "weight_decay": config.train.weight_decay,
        "warmup_ratio": config.train.warmup_ratio,
        "grad_accum_steps": config.train.gradient_accumulation_steps,
        "dropout": config.model.dropout,
        "bottleneck_dim": config.model.bottleneck_dim,
        "max_length": config.data.max_length,
        "filter_max_intensity_lte": config.data.filter_max_intensity_lte,
    }
    metric_dict = {
        "hparam/mse_macro": metrics["mse_macro"],
        "hparam/pearson_min": metrics["pearson_min"],
        "hparam/pearson_top6_min": metrics["pearson_top6_min"],
        "hparam/pearson_anger_trust_min": metrics["pearson_anger_trust_min"],
        "hparam/top1_accuracy": metrics["top1_accuracy"],
        "hparam/test_loss": metrics["test_loss"],
    }
    writer.add_hparams(hparam_dict, metric_dict)


def _validate_accelerate_settings(config: Phase0Config) -> None:
    if config.train.gradient_accumulation_steps < 1:
        msg = "train.gradient_accumulation_steps must be >= 1"
        raise ValueError(msg)

    if config.train.mixed_precision not in {"no", "fp16", "bf16"}:
        msg = "train.mixed_precision must be one of: no, fp16, bf16"
        raise ValueError(msg)

    if config.train.device not in {"cpu", "cuda"}:
        msg = "train.device must be either 'cpu' or 'cuda'"
        raise ValueError(msg)

    if config.train.emotion_weight_mode not in {"none", "inverse_mean"}:
        msg = "train.emotion_weight_mode must be one of: none, inverse_mean"
        raise ValueError(msg)

    if config.train.emotion_weight_epsilon <= 0:
        msg = "train.emotion_weight_epsilon must be > 0"
        raise ValueError(msg)


def train_phase0(config: Phase0Config) -> dict[str, Any] | None:
    _validate_accelerate_settings(config)
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
    loaders, splits, metadata = _create_dataloaders(config, tokenizer)
    data_report = build_data_report(splits, metadata)
    data_report["unk_ratio_train"] = estimate_unk_ratio(
        tokenizer,
        splits["train"].texts,
    )

    model = TextEmotionRegressor(
        pretrained_model_name=config.model.pretrained_model_name,
        bottleneck_dim=config.model.bottleneck_dim,
        dropout=config.model.dropout,
    )

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

    weights, loss_weighting = _resolve_loss_weights(
        config,
        splits["train"].targets,
        accelerator.device,
    )
    data_report["loss_weighting"] = loss_weighting

    best_val_loss = float("inf")
    best_path = checkpoints_dir / "best_model.pt"
    early_stop_count = 0
    history: list[dict[str, float]] = []
    global_step = 0

    for epoch in range(1, config.train.num_epochs + 1):
        model.train()
        train_losses: list[float] = []
        optimizer.zero_grad(set_to_none=True)

        for batch in loaders["train"]:
            labels = cast("torch.Tensor", batch.pop("labels"))

            with accelerator.accumulate(model):
                predictions = cast("torch.Tensor", model(**batch))
                loss: torch.Tensor = _weighted_mse(predictions, labels, weights)

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
            train_losses.append(_to_scalar(reduced_step_loss))

        val_loss, val_preds, val_targets = _run_eval(
            model,
            loaders["val"],
            accelerator,
            weights,
        )
        epoch_train_loss = float(np.mean(train_losses)) if train_losses else 0.0

        if writer is not None:
            writer.add_scalars(
                "loss/epoch",
                {"train": epoch_train_loss, "val": val_loss},
                epoch,
            )

        val_metrics = compute_regression_metrics(val_preds, val_targets)
        if writer is not None:
            writer.add_scalar("val/mse_macro", val_metrics["mse_macro"], epoch)
            writer.add_scalar("val/pearson_min", val_metrics["pearson_min"], epoch)
            writer.add_scalar(
                "val/pearson_top6_min",
                val_metrics["pearson_top6_min"],
                epoch,
            )
            writer.add_scalar(
                "val/pearson_anger_trust_min",
                val_metrics["pearson_anger_trust_min"],
                epoch,
            )
            writer.add_scalar("val/top1_accuracy", val_metrics["top1_accuracy"], epoch)
            for emotion in EMOTION_LABELS:
                writer.add_scalar(
                    f"val/mse_{emotion}",
                    val_metrics["mse_per_emotion"][emotion],
                    epoch,
                )
                writer.add_scalar(
                    f"val/pearson_{emotion}",
                    val_metrics["pearson_per_emotion"][emotion],
                    epoch,
                )

        if accelerator.is_main_process:
            history.append(
                {
                    "epoch": float(epoch),
                    "train_loss": epoch_train_loss,
                    "val_loss": val_loss,
                    "global_step": float(global_step),
                },
            )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            early_stop_count = 0
            if accelerator.is_main_process:
                accelerator.save(
                    {
                        "model_state_dict": accelerator.get_state_dict(model),
                        "model_config": asdict(config.model),
                        "tokenizer_name": config.model.pretrained_model_name,
                        "max_length": config.data.max_length,
                        "emotion_labels": EMOTION_LABELS,
                        "config": config.to_dict(),
                    },
                    best_path,
                )
        else:
            early_stop_count += 1

        if early_stop_count >= config.train.early_stopping_patience:
            break

    accelerator.wait_for_everyone()
    checkpoint = torch.load(best_path, map_location=accelerator.device)
    accelerator.unwrap_model(model).load_state_dict(checkpoint["model_state_dict"])

    test_loss, test_predictions, test_targets = _run_eval(
        model,
        loaders["test"],
        accelerator,
        weights,
    )
    metrics = compute_regression_metrics(test_predictions, test_targets)
    metrics["test_loss"] = test_loss
    go_no_go = _go_no_go(metrics, config)

    if writer is not None:
        writer.add_scalar("test/mse_macro", metrics["mse_macro"], 0)
        writer.add_scalar("test/pearson_min", metrics["pearson_min"], 0)
        writer.add_scalar("test/pearson_top6_min", metrics["pearson_top6_min"], 0)
        writer.add_scalar(
            "test/pearson_anger_trust_min",
            metrics["pearson_anger_trust_min"],
            0,
        )
        writer.add_scalar("test/top1_accuracy", metrics["top1_accuracy"], 0)
        for emotion in EMOTION_LABELS:
            writer.add_scalar(
                f"test/mse_{emotion}",
                metrics["mse_per_emotion"][emotion],
                0,
            )
            writer.add_scalar(
                f"test/pearson_{emotion}",
                metrics["pearson_per_emotion"][emotion],
                0,
            )

        _log_hparams(writer, config, metrics)
        writer.close()

    error_analysis = _build_error_analysis(
        test_predictions,
        test_targets,
        splits["test"].texts,
    )

    if not accelerator.is_main_process:
        return None

    save_effective_config(config, output_dir / "effective_config.yaml")
    _save_hparams_yaml(config, output_dir / "hparams.yaml", loss_weighting)

    with (reports_dir / "data_report.json").open("w", encoding="utf-8") as file:
        json.dump(data_report, file, ensure_ascii=False, indent=2)

    with (reports_dir / "training_history.json").open("w", encoding="utf-8") as file:
        json.dump(history, file, ensure_ascii=False, indent=2)

    with (reports_dir / "evaluation.json").open("w", encoding="utf-8") as file:
        json.dump(
            {
                "metrics": metrics,
                "go_no_go": go_no_go,
                "loss_weighting": loss_weighting,
                "error_analysis": error_analysis,
            },
            file,
            ensure_ascii=False,
            indent=2,
        )

    confusion_matrix = np.array(error_analysis["dominant_confusion_matrix"]["matrix"])
    plot_confusion_matrix(
        confusion_matrix,
        EMOTION_LABELS,
        reports_dir / "confusion_matrix.png",
    )
    plot_per_emotion_error(
        error_analysis["per_emotion_error_stats"],
        reports_dir / "per_emotion_error.png",
    )

    return {
        "output_dir": str(output_dir),
        "checkpoint": str(best_path),
        "device": str(accelerator.device),
        "data_report": data_report,
        "history": history,
        "metrics": metrics,
        "go_no_go": go_no_go,
        "loss_weighting": loss_weighting,
        "error_analysis": error_analysis,
    }


def _save_hparams_yaml(
    config: Phase0Config,
    path: Path,
    loss_weighting: dict[str, Any],
) -> None:
    import yaml

    loss_mode = str(loss_weighting.get("mode", "none"))
    effective_weights = loss_weighting.get("weights")
    hparams = {
        "model": {
            "pretrained_model_name": config.model.pretrained_model_name,
            "bottleneck_dim": config.model.bottleneck_dim,
            "dropout": config.model.dropout,
        },
        "optimizer": {
            "type": "AdamW",
            "bert_lr": config.train.bert_lr,
            "head_lr": config.train.head_lr,
            "weight_decay": config.train.weight_decay,
        },
        "scheduler": {
            "type": "linear_warmup_decay",
            "warmup_ratio": config.train.warmup_ratio,
        },
        "training": {
            "batch_size": config.train.batch_size,
            "num_epochs": config.train.num_epochs,
            "early_stopping_patience": config.train.early_stopping_patience,
            "gradient_accumulation_steps": config.train.gradient_accumulation_steps,
            "mixed_precision": config.train.mixed_precision,
            "random_seed": config.data.random_seed,
        },
        "data": {
            "dataset": config.data.dataset_name,
            "config_name": config.data.dataset_config_name,
            "label_source": config.data.label_source,
            "max_length": config.data.max_length,
            "filter_max_intensity_lte": config.data.filter_max_intensity_lte,
            "train_ratio": config.data.train_ratio,
            "val_ratio": config.data.val_ratio,
            "test_ratio": config.data.test_ratio,
        },
        "loss": {
            "type": "WeightedMSE" if loss_mode != "none" else "MSE",
            "mode": loss_mode,
            "emotion_weights": config.train.emotion_weights,
            "effective_emotion_weights": effective_weights,
            "emotion_weight_epsilon": config.train.emotion_weight_epsilon,
            "emotion_weight_normalize": config.train.emotion_weight_normalize,
        },
    }

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(hparams, f, sort_keys=False, allow_unicode=True)


def _build_error_analysis(
    predictions: np.ndarray,
    targets: np.ndarray,
    texts: list[str],
    top_k: int = 20,
) -> dict[str, Any]:
    per_sample_mse = np.mean(np.square(predictions - targets), axis=1)
    worst_indices = np.argsort(per_sample_mse)[-top_k:][::-1]

    high_error_samples = []
    for idx in worst_indices:
        idx = int(idx)
        pred_dict = {
            e: round(float(predictions[idx, i]), 4)
            for i, e in enumerate(EMOTION_LABELS)
        }
        target_dict = {
            e: round(float(targets[idx, i]), 4) for i, e in enumerate(EMOTION_LABELS)
        }
        high_error_samples.append(
            {
                "index": idx,
                "text": texts[idx][:200],
                "mse": round(float(per_sample_mse[idx]), 6),
                "predicted": pred_dict,
                "target": target_dict,
                "dominant_predicted": EMOTION_LABELS[int(np.argmax(predictions[idx]))],
                "dominant_target": EMOTION_LABELS[int(np.argmax(targets[idx]))],
            },
        )

    n_emotions = len(EMOTION_LABELS)
    confusion = np.zeros((n_emotions, n_emotions), dtype=np.int64)
    pred_dominant = np.argmax(predictions, axis=1)
    true_dominant = np.argmax(targets, axis=1)
    for p, t in zip(pred_dominant, true_dominant, strict=False):
        confusion[t, p] += 1

    confusion_dict = {
        "matrix": confusion.tolist(),
        "labels": EMOTION_LABELS,
        "rows": "true_dominant",
        "cols": "predicted_dominant",
    }

    per_emotion_errors: dict[str, dict[str, float]] = {}
    for i, emotion in enumerate(EMOTION_LABELS):
        errors = predictions[:, i] - targets[:, i]
        per_emotion_errors[emotion] = {
            "mean_error": round(float(np.mean(errors)), 6),
            "std_error": round(float(np.std(errors)), 6),
            "mean_abs_error": round(float(np.mean(np.abs(errors))), 6),
        }

    return {
        "high_error_samples": high_error_samples,
        "dominant_confusion_matrix": confusion_dict,
        "per_emotion_error_stats": per_emotion_errors,
    }
