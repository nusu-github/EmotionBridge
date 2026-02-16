"""emotion2vec+ 埋め込み抽出器。

FunASR経由でemotion2vec_plus_largeをロードし、
1024D埋め込み (feats) と 9クラスlogits (scores) を抽出する。

emotion2vec+ の9感情クラス:
  0: angry, 1: disgusted, 2: fearful, 3: happy, 4: neutral,
  5: other, 6: sad, 7: surprised, 8: unknown
"""

import logging

import numpy as np

logger = logging.getLogger(__name__)

# emotion2vec+ の感情ラベル (index順)
EMOTION2VEC_LABELS = [
    "angry",
    "disgusted",
    "fearful",
    "happy",
    "neutral",
    "other",
    "sad",
    "surprised",
    "unknown",
]


class Emotion2vecExtractor:
    """emotion2vec+ ベースの埋め込み抽出器。"""

    def __init__(
        self,
        model_id: str = "emotion2vec/emotion2vec_plus_large",
        device: str = "cuda",
    ) -> None:
        from funasr import AutoModel

        logger.info("emotion2vec+ モデルをロード中: %s", model_id)
        self.model = AutoModel(
            model=model_id,
            hub="hf",
            device=device,
            disable_update=True,
        )
        logger.info("emotion2vec+ ロード完了")

    def extract(
        self,
        audio_paths: list[str],
        batch_size: int = 16,
    ) -> dict[str, np.ndarray]:
        """音声ファイルリストから埋め込みとlogitsを抽出する。

        Args:
            audio_paths: WAVファイルパスのリスト
            batch_size: バッチサイズ (FunASR内部で使用)

        Returns:
            dict with keys:
                "feats": 1024D 埋め込み [N, 1024]
                "logits": 9クラスsoftmaxスコア [N, 9]

        """
        all_feats = []
        all_logits = []

        n_total = len(audio_paths)
        for start in range(0, n_total, batch_size):
            end = min(start + batch_size, n_total)
            batch_paths = audio_paths[start:end]

            logger.info("バッチ処理中: %d-%d / %d", start + 1, end, n_total)

            results = self.model.generate(
                batch_paths,
                granularity="utterance",
                extract_embedding=True,
            )

            for r in results:
                all_feats.append(r["feats"])
                all_logits.append(np.array(r["scores"], dtype=np.float32))

        return {
            "feats": np.stack(all_feats, axis=0),
            "logits": np.stack(all_logits, axis=0),
        }
