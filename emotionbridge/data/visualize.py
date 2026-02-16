from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from emotionbridge.constants import EMOTION_LABELS

from .wrime import PreparedSplit

plt.rcParams["font.family"] = "DejaVu Sans"


def plot_intensity_histograms(
    splits: dict[str, PreparedSplit],
    output_path: str | Path,
) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(2, 4, figsize=(20, 8))
    axes = axes.flatten()

    for i, emotion in enumerate(EMOTION_LABELS):
        ax = axes[i]
        for split_name, split in splits.items():
            if split.size == 0:
                continue
            raw_values = split.raw_targets[:, i]
            counts = np.bincount(raw_values.astype(int), minlength=4)[:4]
            ratios = counts / counts.sum()
            ax.bar(
                np.arange(4) + {"train": -0.25, "val": 0.0, "test": 0.25}[split_name],
                ratios,
                width=0.25,
                label=split_name,
                alpha=0.85,
            )
        ax.set_title(emotion, fontsize=12, fontweight="bold")
        ax.set_xlabel("Intensity (raw 0-3)")
        ax.set_ylabel("Ratio")
        ax.set_xticks([0, 1, 2, 3])
        if i == 0:
            ax.legend(fontsize=8)

    fig.suptitle(
        "Emotion Intensity Distribution (after filter)",
        fontsize=14,
        fontweight="bold",
    )
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_cooccurrence_matrix(
    split: PreparedSplit,
    output_path: str | Path,
    threshold: int = 2,
) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    n_emotions = len(EMOTION_LABELS)
    cooccurrence = np.zeros((n_emotions, n_emotions), dtype=np.float64)

    active = split.raw_targets >= threshold

    for i in range(n_emotions):
        for j in range(n_emotions):
            cooccurrence[i, j] = np.sum(active[:, i] & active[:, j])

    diag = np.diag(cooccurrence).copy()
    diag[diag == 0] = 1.0
    normalized = cooccurrence / diag[:, np.newaxis]

    fig, axes = plt.subplots(1, 2, figsize=(18, 7))

    sns.heatmap(
        cooccurrence,
        annot=True,
        fmt=".0f",
        xticklabels=EMOTION_LABELS,
        yticklabels=EMOTION_LABELS,
        cmap="YlOrRd",
        ax=axes[0],
    )
    axes[0].set_title(f"Co-occurrence Count (intensity >= {threshold})", fontsize=12)

    sns.heatmap(
        normalized,
        annot=True,
        fmt=".2f",
        xticklabels=EMOTION_LABELS,
        yticklabels=EMOTION_LABELS,
        cmap="YlOrRd",
        vmin=0,
        vmax=1,
        ax=axes[1],
    )
    axes[1].set_title(f"P(col | row) (intensity >= {threshold})", fontsize=12)

    fig.suptitle("Emotion Co-occurrence (train split)", fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_emotion_means_comparison(
    splits: dict[str, PreparedSplit],
    output_path: str | Path,
) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(12, 5))
    x = np.arange(len(EMOTION_LABELS))
    width = 0.25

    for offset, (split_name, split) in enumerate(splits.items()):
        if split.size == 0:
            continue
        means = split.targets.mean(axis=0)
        ax.bar(x + (offset - 1) * width, means, width, label=split_name, alpha=0.85)

    ax.set_xticks(x)
    ax.set_xticklabels(EMOTION_LABELS, rotation=30, ha="right")
    ax.set_ylabel("Mean Normalized Intensity [0, 1]")
    ax.set_title("Mean Emotion Intensity by Split", fontsize=13, fontweight="bold")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_confusion_matrix(
    confusion: np.ndarray,
    labels: list[str],
    output_path: str | Path,
) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    row_sums = confusion.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    normalized = confusion / row_sums

    fig, axes = plt.subplots(1, 2, figsize=(18, 7))

    sns.heatmap(
        confusion,
        annot=True,
        fmt="d",
        xticklabels=labels,
        yticklabels=labels,
        cmap="Blues",
        ax=axes[0],
    )
    axes[0].set_xlabel("Predicted Dominant")
    axes[0].set_ylabel("True Dominant")
    axes[0].set_title("Dominant Emotion Confusion (Count)", fontsize=12)

    sns.heatmap(
        normalized,
        annot=True,
        fmt=".2f",
        xticklabels=labels,
        yticklabels=labels,
        cmap="Blues",
        vmin=0,
        vmax=1,
        ax=axes[1],
    )
    axes[1].set_xlabel("Predicted Dominant")
    axes[1].set_ylabel("True Dominant")
    axes[1].set_title("Dominant Emotion Confusion (Row-Normalized)", fontsize=12)

    fig.suptitle(
        "Test Set: Dominant Emotion Confusion Matrix",
        fontsize=14,
        fontweight="bold",
    )
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_per_emotion_error(
    per_emotion_errors: dict[str, dict[str, float]],
    output_path: str | Path,
) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    emotions = list(per_emotion_errors.keys())
    mean_errors = [per_emotion_errors[e]["mean_error"] for e in emotions]
    mae = [per_emotion_errors[e]["mean_abs_error"] for e in emotions]
    std_errors = [per_emotion_errors[e]["std_error"] for e in emotions]

    fig, axes = plt.subplots(1, 2, figsize=(16, 5))

    x = np.arange(len(emotions))
    colors = ["#e74c3c" if v > 0 else "#2ecc71" for v in mean_errors]
    axes[0].bar(x, mean_errors, color=colors, alpha=0.8)
    axes[0].axhline(y=0, color="black", linewidth=0.5)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(emotions, rotation=30, ha="right")
    axes[0].set_ylabel("Mean Error (pred - target)")
    axes[0].set_title("Prediction Bias by Emotion", fontsize=12)

    axes[1].bar(x, mae, color="#3498db", alpha=0.8)
    axes[1].errorbar(x, mae, yerr=std_errors, fmt="none", color="black", capsize=3)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(emotions, rotation=30, ha="right")
    axes[1].set_ylabel("Mean Absolute Error")
    axes[1].set_title("MAE by Emotion (with std)", fontsize=12)

    fig.suptitle("Test Set: Per-Emotion Error Analysis", fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
