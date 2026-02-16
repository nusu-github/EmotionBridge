"""埋め込み空間の可視化・定量評価モジュール。"""

import json
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from scipy.spatial.distance import cosine
from sklearn.manifold import TSNE
from sklearn.metrics import calinski_harabasz_score, silhouette_score

logger = logging.getLogger(__name__)

# 感情ラベル→色のマッピング (6色)
EMOTION_PALETTE = {
    "anger": "#e74c3c",
    "disgust": "#8e44ad",
    "fear": "#2c3e50",
    "happy": "#f39c12",
    "sad": "#3498db",
    "surprise": "#2ecc71",
}

# 話者→マーカーのマッピング
SPEAKER_MARKERS = {
    "F1": "o",
    "F2": "s",
    "M1": "^",
    "M2": "D",
}


def _auto_display_name(layer_name: str, dim: int) -> str:
    """レイヤー名と次元から表示名を自動生成する。"""
    return f"{layer_name} ({dim}D)"


def reduce_and_plot(
    embeddings_dict: dict[str, np.ndarray],
    labels: list[str],
    output_dir: str | Path,
    speakers: list[str] | None = None,
    tsne_perplexity: float = 30.0,
    umap_n_neighbors: int = 15,
    umap_min_dist: float = 0.1,
    random_seed: int = 42,
) -> None:
    """N層 x 2手法 (t-SNE, UMAP) の比較プロットを生成する。

    Args:
        embeddings_dict: レイヤー名→埋め込み配列のdict (例: {"feats": [N,1024], "logits": [N,9]})
        labels: 感情ラベルのリスト (長さN)
        output_dir: プロット出力先
        speakers: 話者ラベルのリスト (長さN, Noneならマーカー区別なし)
        tsne_perplexity: t-SNEのperplexity
        umap_n_neighbors: UMAPのn_neighbors
        umap_min_dist: UMAPのmin_dist
        random_seed: ランダムシード

    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    labels_arr = np.array(labels)
    unique_emotions = sorted(set(labels))

    layer_names = list(embeddings_dict.keys())
    n_layers = len(layer_names)

    # --- t-SNE ---
    logger.info("t-SNE 次元削減を実行中...")
    fig_tsne, axes_tsne = plt.subplots(1, n_layers, figsize=(8 * n_layers, 7))
    if n_layers == 1:
        axes_tsne = [axes_tsne]
    for i, layer in enumerate(layer_names):
        emb = embeddings_dict[layer]
        display_name = _auto_display_name(layer, emb.shape[1])
        perp = min(tsne_perplexity, max(5, emb.shape[0] // 4))
        if emb.shape[1] <= 2:
            reduced = (
                emb[:, :2]
                if emb.shape[1] == 2
                else np.column_stack([emb[:, 0], np.zeros(len(emb))])
            )
        else:
            tsne = TSNE(
                n_components=2,
                perplexity=perp,
                max_iter=1000,
                random_state=random_seed,
            )
            reduced = tsne.fit_transform(emb)

        _scatter_plot(
            axes_tsne[i],
            reduced,
            labels_arr,
            unique_emotions,
            speakers,
            display_name,
        )
    fig_tsne.suptitle("t-SNE Embedding Comparison", fontsize=16, y=1.02)
    fig_tsne.tight_layout()
    fig_tsne.savefig(out / "tsne_comparison.png", dpi=150, bbox_inches="tight")
    plt.close(fig_tsne)
    logger.info("t-SNE プロット保存: %s", out / "tsne_comparison.png")

    # --- UMAP ---
    try:
        import umap

        logger.info("UMAP 次元削減を実行中...")
        fig_umap, axes_umap = plt.subplots(1, n_layers, figsize=(8 * n_layers, 7))
        if n_layers == 1:
            axes_umap = [axes_umap]
        for i, layer in enumerate(layer_names):
            emb = embeddings_dict[layer]
            display_name = _auto_display_name(layer, emb.shape[1])
            n_neighbors = min(umap_n_neighbors, emb.shape[0] - 1)
            if emb.shape[1] <= 2:
                reduced = (
                    emb[:, :2]
                    if emb.shape[1] == 2
                    else np.column_stack([emb[:, 0], np.zeros(len(emb))])
                )
            else:
                reducer = umap.UMAP(
                    n_neighbors=n_neighbors,
                    min_dist=umap_min_dist,
                    random_state=random_seed,
                )
                reduced = reducer.fit_transform(emb)

            _scatter_plot(
                axes_umap[i],
                reduced,
                labels_arr,
                unique_emotions,
                speakers,
                display_name,
            )
        fig_umap.suptitle("UMAP Embedding Comparison", fontsize=16, y=1.02)
        fig_umap.tight_layout()
        fig_umap.savefig(out / "umap_comparison.png", dpi=150, bbox_inches="tight")
        plt.close(fig_umap)
        logger.info("UMAP プロット保存: %s", out / "umap_comparison.png")
    except ImportError:
        logger.warning(
            "umap-learn がインストールされていません。UMAPプロットをスキップします。",
        )


def _scatter_plot(
    ax: plt.Axes,
    reduced: np.ndarray,
    labels: np.ndarray,
    unique_emotions: list[str],
    speakers: list[str] | None,
    title: str,
) -> None:
    """散布図を描画する。"""
    if speakers is not None:
        speakers_arr = np.array(speakers)
        unique_speakers = sorted(set(speakers))
        for emotion in unique_emotions:
            for speaker in unique_speakers:
                mask = (labels == emotion) & (speakers_arr == speaker)
                if not mask.any():
                    continue
                ax.scatter(
                    reduced[mask, 0],
                    reduced[mask, 1],
                    c=EMOTION_PALETTE.get(emotion, "#999999"),
                    marker=SPEAKER_MARKERS.get(speaker, "o"),
                    s=30,
                    alpha=0.7,
                    label=f"{emotion} ({speaker})",
                )
    else:
        for emotion in unique_emotions:
            mask = labels == emotion
            ax.scatter(
                reduced[mask, 0],
                reduced[mask, 1],
                c=EMOTION_PALETTE.get(emotion, "#999999"),
                s=30,
                alpha=0.7,
                label=emotion,
            )

    ax.set_title(title, fontsize=13)
    ax.set_xlabel("Dim 1")
    ax.set_ylabel("Dim 2")
    # 凡例: 感情のみ (話者×感情だと多すぎるため)
    handles, lbls = ax.get_legend_handles_labels()
    # 感情ごとに1つだけ表示
    seen = set()
    unique_handles, unique_lbls = [], []
    for h, lbl in zip(handles, lbls, strict=True):
        emotion_key = lbl.split(" (")[0]
        if emotion_key not in seen:
            seen.add(emotion_key)
            unique_handles.append(h)
            unique_lbls.append(emotion_key)
    ax.legend(unique_handles, unique_lbls, loc="best", fontsize=9)


def compute_metrics(
    embeddings_dict: dict[str, np.ndarray],
    labels: list[str],
) -> dict:
    """定量評価メトリクスを計算する。

    Returns:
        dict: {
            "layer_name": {
                "silhouette_score": float,
                "calinski_harabasz_index": float,
            },
            "cosine_distance_matrix": {
                "layer_name": {
                    "emotion_pair": float
                }
            }
        }

    """
    labels_arr = np.array(labels)
    unique_emotions = sorted(set(labels))
    n_classes = len(unique_emotions)

    metrics: dict = {"per_layer": {}, "cosine_distance_matrix": {}}

    for layer in embeddings_dict:
        emb = embeddings_dict[layer]
        layer_metrics: dict = {}

        # シルエットスコア (2クラス以上 & サンプル数がクラス数より十分多い場合)
        if n_classes >= 2 and emb.shape[0] > n_classes:
            try:
                layer_metrics["silhouette_score"] = float(
                    silhouette_score(emb, labels_arr),
                )
            except ValueError:
                layer_metrics["silhouette_score"] = None
        else:
            layer_metrics["silhouette_score"] = None

        # Calinski-Harabaszインデックス
        if n_classes >= 2 and emb.shape[0] > n_classes:
            try:
                layer_metrics["calinski_harabasz_index"] = float(
                    calinski_harabasz_score(emb, labels_arr),
                )
            except ValueError:
                layer_metrics["calinski_harabasz_index"] = None
        else:
            layer_metrics["calinski_harabasz_index"] = None

        metrics["per_layer"][layer] = layer_metrics

        # 感情ペア間コサイン距離
        centroids = {}
        for emotion in unique_emotions:
            mask = labels_arr == emotion
            centroids[emotion] = emb[mask].mean(axis=0)

        distance_matrix: dict[str, float] = {}
        for i, e1 in enumerate(unique_emotions):
            for e2 in unique_emotions[i + 1 :]:
                pair_key = f"{e1}-{e2}"
                distance_matrix[pair_key] = float(cosine(centroids[e1], centroids[e2]))

        metrics["cosine_distance_matrix"][layer] = distance_matrix

    return metrics


def plot_distance_matrix(
    metrics: dict,
    output_dir: str | Path,
    layer: str = "projected",
) -> None:
    """感情ペア間コサイン距離のヒートマップを描画する。"""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    distance_data = metrics["cosine_distance_matrix"].get(layer, {})
    if not distance_data:
        logger.warning("距離データが空です: layer=%s", layer)
        return

    # 感情ラベルを収集
    emotions = sorted({e for pair in distance_data for e in pair.split("-")})
    n = len(emotions)
    matrix = np.zeros((n, n))

    emo_to_idx = {e: i for i, e in enumerate(emotions)}
    for pair, dist in distance_data.items():
        e1, e2 = pair.split("-")
        i, j = emo_to_idx[e1], emo_to_idx[e2]
        matrix[i, j] = dist
        matrix[j, i] = dist

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(
        matrix,
        xticklabels=emotions,
        yticklabels=emotions,
        annot=True,
        fmt=".3f",
        cmap="YlOrRd",
        ax=ax,
    )
    ax.set_title(f"Cosine Distance Matrix — {layer}")
    fig.tight_layout()
    fig.savefig(out / "emotion_distance_matrix.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("距離行列プロット保存: %s", out / "emotion_distance_matrix.png")


def save_report(metrics: dict, output_path: str | Path) -> None:
    """メトリクスをJSON形式で保存する。"""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    logger.info("メトリクスレポート保存: %s", path)
