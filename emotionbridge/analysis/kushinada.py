"""くしなだ (kushinada-hubert-large) 埋め込み抽出器。

下流モデル（JTES 4感情分類）のチェックポイントから projector / post_net を
再構成し、3層の埋め込みを抽出する。

s3prlチェックポイント形式:
  - トップレベルキー: Config, Args, Upstream, Downstream, Optimizer, Step, ...
  - Downstream state_dict キー:
      projector.weight, projector.bias
      model.post_net.linear.weight, model.post_net.linear.bias
"""

import logging
from pathlib import Path

import numpy as np
import torch
import torchaudio
from torch import nn
from transformers import HubertModel

logger = logging.getLogger(__name__)

# s3prlチェックポイント内のデフォルト fold (fold1)
_DEFAULT_HF_CKPT_PATH = (
    "s3prl/result/downstream/kushinada-hubert-large-jtes-er_fold1/dev-best.ckpt"
)


def download_downstream_ckpt(
    repo_id: str,
    local_path: str | Path,
    hf_filename: str = _DEFAULT_HF_CKPT_PATH,
) -> Path:
    """HuggingFace Hubからくしなだ下流チェックポイントをダウンロードする。

    Args:
        repo_id: HuggingFace リポジトリID (例: imprt/kushinada-hubert-large-jtes-er)
        local_path: ローカル保存先パス
        hf_filename: リポジトリ内のファイルパス

    Returns:
        ダウンロードされたファイルのパス

    """
    from huggingface_hub import hf_hub_download

    local_path = Path(local_path)
    if local_path.exists():
        logger.info("チェックポイントが既に存在: %s", local_path)
        return local_path

    logger.info("チェックポイントをダウンロード中: %s/%s", repo_id, hf_filename)
    cached = hf_hub_download(repo_id=repo_id, filename=hf_filename)

    # キャッシュからローカルパスにコピー
    local_path.parent.mkdir(parents=True, exist_ok=True)
    import shutil

    shutil.copy2(cached, local_path)
    logger.info("チェックポイントを保存: %s", local_path)
    return local_path


class KushinadaExtractor:
    """くしなだHuBERTベースの埋め込み抽出器。"""

    def __init__(
        self,
        upstream_name: str,
        downstream_ckpt_path: str,
        device: str = "cuda",
        downstream_repo_id: str = "imprt/kushinada-hubert-large-jtes-er",
    ) -> None:
        self.device = torch.device(device)

        # 上流モデル: HuBERT
        logger.info("上流モデルをロード中: %s", upstream_name)
        self.hubert: HubertModel = HubertModel.from_pretrained(upstream_name).to(
            self.device,
        )
        self.hubert.eval()
        hubert_dim = self.hubert.config.hidden_size  # 1024

        # 下流チェックポイントの取得（ローカルに無ければダウンロード）
        ckpt_path = Path(downstream_ckpt_path)
        if not ckpt_path.exists():
            logger.info(
                "ローカルにチェックポイントが見つかりません。HFからダウンロードします...",
            )
            ckpt_path = download_downstream_ckpt(downstream_repo_id, ckpt_path)

        logger.info("下流チェックポイントをロード中: %s", ckpt_path)
        ckpt = torch.load(ckpt_path, map_location=self.device, weights_only=False)

        # s3prl形式: Downstream キー配下に state_dict
        state_dict = self._extract_downstream_state_dict(ckpt)

        logger.info("チェックポイントのキー: %s", list(state_dict.keys())[:20])

        # projector (Linear 1024→256) のウェイトを探索
        proj_weight, proj_bias = self._find_layer_weights(
            state_dict,
            "projector",
            hubert_dim,
            256,
        )
        self.projector = nn.Linear(hubert_dim, 256).to(self.device)
        self.projector.load_state_dict({"weight": proj_weight, "bias": proj_bias})
        self.projector.eval()
        logger.info("projector ロード完了: Linear(%d, 256)", hubert_dim)

        # post_net (Linear 256→4) のウェイトを探索
        post_weight, post_bias = self._find_layer_weights(
            state_dict,
            "post_net",
            256,
            4,
        )
        self.post_net = nn.Linear(256, 4).to(self.device)
        self.post_net.load_state_dict({"weight": post_weight, "bias": post_bias})
        self.post_net.eval()
        logger.info("post_net ロード完了: Linear(256, 4)")

        # リサンプラ (48kHz→16kHz)
        self.resampler = torchaudio.transforms.Resample(
            orig_freq=48000,
            new_freq=16000,
        ).to(self.device)

        logger.info("KushinadaExtractor 初期化完了")

    @staticmethod
    def _extract_downstream_state_dict(ckpt: dict) -> dict:
        """s3prl/PyTorch Lightningチェックポイントから下流モデルのstate_dictを抽出。"""
        # s3prl runner形式: トップレベルに 'Downstream' キー
        if "Downstream" in ckpt:
            return ckpt["Downstream"]

        # PyTorch Lightning形式: 'state_dict' キー
        if "state_dict" in ckpt:
            sd = ckpt["state_dict"]
            # Downstream. プレフィックスを除去
            stripped = {}
            for k, v in sd.items():
                if k.startswith("Downstream."):
                    stripped[k[len("Downstream.") :]] = v
                else:
                    stripped[k] = v
            return stripped

        # フラットなstate_dict
        return ckpt

    @staticmethod
    def _find_layer_weights(
        state_dict: dict,
        layer_name: str,
        in_features: int,
        out_features: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """state_dictから指定レイヤーのウェイトを探索する。

        s3prl emotion downstreamの典型的なキー:
          - projector.weight / projector.bias
          - model.post_net.linear.weight / model.post_net.linear.bias
          - model.post_net.0.weight / model.post_net.0.bias
        """
        # パターン1: 直接名前で検索 (projector.weight など)
        candidates = [
            f"{layer_name}.",
            f"model.{layer_name}.",
            f"model.{layer_name}.linear.",
            f"model.{layer_name}.0.",
            f"downstream.{layer_name}.",
        ]
        for prefix in candidates:
            w_key = f"{prefix}weight"
            b_key = f"{prefix}bias"
            if w_key in state_dict and b_key in state_dict:
                w = state_dict[w_key]
                if w.shape == (out_features, in_features):
                    return w, state_dict[b_key]

        # パターン2: 部分一致で検索
        for key in state_dict:
            if layer_name in key and key.endswith(".weight"):
                w = state_dict[key]
                if w.shape == (out_features, in_features):
                    b_key = key.replace(".weight", ".bias")
                    if b_key in state_dict:
                        return w, state_dict[b_key]

        # パターン3: 形状ベースで検索
        for key, tensor in state_dict.items():
            if not isinstance(tensor, torch.Tensor):
                continue
            if not key.endswith(".weight"):
                continue
            if tensor.shape == (out_features, in_features):
                b_key = key.replace(".weight", ".bias")
                if b_key in state_dict:
                    logger.warning(
                        "%s のウェイトを形状ベースで検出: %s",
                        layer_name,
                        key,
                    )
                    return tensor, state_dict[b_key]

        msg = (
            f"{layer_name} (shape: [{out_features}, {in_features}]) が"
            f"チェックポイントに見つかりません。"
            f"利用可能なキー: {list(state_dict.keys())[:30]}"
        )
        raise KeyError(msg)

    def _load_audio(self, audio_path: str) -> torch.Tensor:
        """音声ファイルを読み込み、16kHzモノラルに変換する。"""
        waveform, sr = torchaudio.load(audio_path)

        # ステレオ→モノラル
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)

        waveform = waveform.to(self.device)

        # リサンプリング
        if sr != 16000:
            if sr != 48000:
                resampler = torchaudio.transforms.Resample(
                    orig_freq=sr,
                    new_freq=16000,
                ).to(self.device)
                waveform = resampler(waveform)
            else:
                waveform = self.resampler(waveform)

        return waveform.squeeze(0)  # [T]

    @torch.no_grad()
    def extract(
        self,
        audio_paths: list[str],
        batch_size: int = 16,
    ) -> dict[str, np.ndarray]:
        """音声ファイルリストから3層の埋め込みを抽出する。

        Args:
            audio_paths: WAVファイルパスのリスト
            batch_size: バッチサイズ

        Returns:
            dict with keys:
                "hubert_pooled": HuBERT last_hidden_state の時間平均 [N, 1024]
                "projected": projector適用後の時間平均 [N, 256]
                "logits": post_net適用後 [N, 4]

        """
        all_hubert_pooled = []
        all_projected = []
        all_logits = []

        n_total = len(audio_paths)
        for start in range(0, n_total, batch_size):
            end = min(start + batch_size, n_total)
            batch_paths = audio_paths[start:end]

            logger.info("バッチ処理中: %d-%d / %d", start + 1, end, n_total)

            # 音声ロード & パディング
            waveforms = [self._load_audio(p) for p in batch_paths]
            lengths = [w.shape[0] for w in waveforms]
            max_len = max(lengths)

            # ゼロパディングでバッチ化
            padded = torch.zeros(len(waveforms), max_len, device=self.device)
            attention_mask = torch.zeros(
                len(waveforms),
                max_len,
                device=self.device,
                dtype=torch.long,
            )
            for i, (w, wlen) in enumerate(zip(waveforms, lengths, strict=True)):
                padded[i, :wlen] = w
                attention_mask[i, :wlen] = 1

            # HuBERT forward
            outputs = self.hubert(padded, attention_mask=attention_mask)
            hidden_states = outputs.last_hidden_state  # [B, T', 1024]

            # HuBERTはCNNで時間軸を縮小するため、出力長に合わせた有効フレーム数を推定
            t_out = hidden_states.shape[1]
            valid_frames = [max(1, int(t_out * wlen / max_len)) for wlen in lengths]

            # 層1: HuBERT hidden statesの時間平均 [B, 1024]
            hubert_pooled_list = []
            for i, vf in enumerate(valid_frames):
                pooled = hidden_states[i, :vf, :].mean(dim=0)
                hubert_pooled_list.append(pooled)
            hubert_pooled = torch.stack(hubert_pooled_list)

            # 層2: projectorをフレーム単位で適用→時間平均 [B, 256]
            # s3prlの処理順序: projector(per-frame) → pooling
            # Linear変換と平均は交換可能なので hubert_pooled に適用しても等価
            projected = self.projector(hubert_pooled)

            # 層3: post_net適用 [B, 4]
            logits = self.post_net(projected)

            all_hubert_pooled.append(hubert_pooled.cpu().numpy())
            all_projected.append(projected.cpu().numpy())
            all_logits.append(logits.cpu().numpy())

        return {
            "hubert_pooled": np.concatenate(all_hubert_pooled, axis=0),
            "projected": np.concatenate(all_projected, axis=0),
            "logits": np.concatenate(all_logits, axis=0),
        }
