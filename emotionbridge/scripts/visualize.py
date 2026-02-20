from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import pandas as pd


def _ensure_output_path(output_path: str | Path) -> Path:
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    return out


def plot_tsne_scatter(
    df: pd.DataFrame,
    feature_cols: list[str],
    color_col: str,
    output_path: str | Path,
    *,
    title: str,
    perplexity: float,
    random_seed: int,
    style_col: str | None = None,
    embedded: np.ndarray | None = None,
) -> None:
    if len(df) < 3:
        return

    if embedded is None:
        features = df[feature_cols].to_numpy(dtype=np.float32, copy=False)
        actual_perplexity = min(perplexity, max(2, len(df) // 4))
        reducer = TSNE(
            n_components=2,
            perplexity=actual_perplexity,
            random_state=random_seed,
            init="pca",
            learning_rate="auto",
            max_iter=1000,
        )
        embedded = reducer.fit_transform(features)
    if embedded.shape[0] != len(df):
        msg = "embedded row count must match df length"
        raise ValueError(msg)

    plot_df = df[[color_col]].copy()
    plot_df["tsne_x"] = embedded[:, 0]
    plot_df["tsne_y"] = embedded[:, 1]
    if style_col is not None and style_col in df.columns:
        plot_df[style_col] = df[style_col]

    out = _ensure_output_path(output_path)
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.scatterplot(
        data=plot_df,
        x="tsne_x",
        y="tsne_y",
        hue=color_col,
        style=style_col if style_col in plot_df.columns else None,
        alpha=0.8,
        s=35,
        ax=ax,
    )
    ax.set_title(title)
    ax.set_xlabel("t-SNE 1")
    ax.set_ylabel("t-SNE 2")
    fig.tight_layout()
    fig.savefig(out, dpi=160, bbox_inches="tight")
    plt.close(fig)


def plot_pca_scatter(
    df: pd.DataFrame,
    feature_cols: list[str],
    color_col: str,
    output_path: str | Path,
    *,
    title: str,
    random_seed: int,
    style_col: str | None = None,
    embedded: np.ndarray | None = None,
    explained_variance_ratio: np.ndarray | None = None,
) -> None:
    if len(df) < 3:
        return

    if embedded is None or explained_variance_ratio is None:
        features = df[feature_cols].to_numpy(dtype=np.float32, copy=False)
        reducer = PCA(n_components=2, random_state=random_seed)
        embedded = reducer.fit_transform(features)
        ratio = reducer.explained_variance_ratio_
    else:
        ratio = explained_variance_ratio

    if embedded.shape[0] != len(df):
        msg = "embedded row count must match df length"
        raise ValueError(msg)

    plot_df = df[[color_col]].copy()
    plot_df["pca_x"] = embedded[:, 0]
    plot_df["pca_y"] = embedded[:, 1]
    if style_col is not None and style_col in df.columns:
        plot_df[style_col] = df[style_col]

    out = _ensure_output_path(output_path)
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.scatterplot(
        data=plot_df,
        x="pca_x",
        y="pca_y",
        hue=color_col,
        style=style_col if style_col in plot_df.columns else None,
        alpha=0.8,
        s=35,
        ax=ax,
    )
    ax.set_title(title)
    ax.set_xlabel(f"PC1 ({ratio[0] * 100:.1f}%)")
    ax.set_ylabel(f"PC2 ({ratio[1] * 100:.1f}%)")
    fig.tight_layout()
    fig.savefig(out, dpi=160, bbox_inches="tight")
    plt.close(fig)


def plot_pca_3d_scatter(
    df: pd.DataFrame,
    feature_cols: list[str],
    color_col: str,
    output_path: str | Path,
    *,
    title: str,
    random_seed: int,
) -> None:
    features = df[feature_cols].to_numpy(dtype=np.float32, copy=False)
    if len(df) < 3:
        return

    reducer = PCA(n_components=3, random_state=random_seed)
    embedded = reducer.fit_transform(features)

    categories = sorted(df[color_col].dropna().unique().tolist())
    palette = sns.color_palette("tab10", n_colors=max(1, len(categories)))
    color_map = {label: palette[index] for index, label in enumerate(categories)}

    out = _ensure_output_path(output_path)
    fig = plt.figure(figsize=(11, 9))
    ax = fig.add_subplot(111, projection="3d")
    for category in categories:
        mask = df[color_col] == category
        ax.scatter(
            embedded[mask, 0],
            embedded[mask, 1],
            embedded[mask, 2],
            alpha=0.75,
            s=28,
            label=category,
            color=color_map[category],
        )

    ratio = reducer.explained_variance_ratio_
    ax.set_title(title)
    ax.set_xlabel(f"PC1 ({ratio[0] * 100:.1f}%)")
    ax.set_ylabel(f"PC2 ({ratio[1] * 100:.1f}%)")
    ax.set_zlabel(f"PC3 ({ratio[2] * 100:.1f}%)")
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(out, dpi=160, bbox_inches="tight")
    plt.close(fig)


def plot_tier_boxplots(
    df: pd.DataFrame,
    feature_cols: list[str],
    group_col: str,
    output_path: str | Path,
    *,
    title: str,
) -> None:
    if not feature_cols:
        return

    melted = df[[group_col, *feature_cols]].melt(
        id_vars=[group_col],
        var_name="feature",
        value_name="value",
    )
    out = _ensure_output_path(output_path)
    fig, ax = plt.subplots(figsize=(14, 7))
    sns.boxplot(data=melted, x="feature", y="value", hue=group_col, ax=ax)
    ax.set_title(title)
    ax.set_xlabel("Feature")
    ax.set_ylabel("Value")
    ax.tick_params(axis="x", rotation=35)
    fig.tight_layout()
    fig.savefig(out, dpi=160, bbox_inches="tight")
    plt.close(fig)


def plot_correlation_heatmap(
    corr_df: pd.DataFrame,
    output_path: str | Path,
    *,
    title: str,
    vmin: float = -1.0,
    vmax: float = 1.0,
) -> None:
    out = _ensure_output_path(output_path)
    fig, ax = plt.subplots(figsize=(16, 8))
    sns.heatmap(
        corr_df,
        cmap="coolwarm",
        center=0.0,
        vmin=vmin,
        vmax=vmax,
        ax=ax,
    )
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(out, dpi=160, bbox_inches="tight")
    plt.close(fig)
