# EmotionBridge

Phase 0「テキスト感情エンコーダ」の初期実装です。設計書 `EB-P0-BD-001` をベースに、以下を実装しています。

- WRIME Ver1 の読込・前処理（客観ラベル抽出 / フィルタ / 正規化 / 分割）
- BERT + 回帰ヘッドによる 8 次元感情ベクトル推定
- 訓練ループ（感情別重み付きMSE, AdamW, warmup+decay, Early Stopping, checkpoint）
- 評価指標（MSE, Pearson r, Top-1 Accuracy）と感情グループ別 Go/No-Go 判定
- 推論 API `EmotionEncoder`（単一 / バッチ）

## セットアップ

```bash
uv sync
```

## CLI

### 1) データ分析

```bash
uv run python main.py analyze-data --config configs/phase0.yaml
```

### 2) 訓練

```bash
uv run python main.py train --config configs/phase0.yaml
```

`Accelerate` を使う場合（推奨）:

```bash
uv run accelerate config
uv run accelerate launch main.py train --config configs/phase0.yaml
```

`configs/phase0.yaml` の `train` セクションで以下を調整できます。

- `gradient_accumulation_steps`: 勾配累積ステップ数
- `mixed_precision`: `"no" | "fp16" | "bf16"`
- `emotion_weight_mode`: `"inverse_mean" | "none"`
- `emotion_weight_epsilon`: 逆数計算の下限値（ゼロ割防止）
- `emotion_weight_normalize`: 重みの平均を 1 に正規化するか
- `emotion_weights`: 手動重み（指定時は自動重みより優先）

Go/No-Go 判定（`eval` セクション）:

- `go_macro_mse_max`: マクロ MSE の上限（全感情共通）
- `go_top6_min_pearson`: 上位 6 感情（`joy/sadness/anticipation/surprise/fear/disgust`）の最小 Pearson 下限
- `go_anger_trust_min_pearson`: `anger/trust` の最小 Pearson 下限
- `go_top1_acc_min`: Top-1 Accuracy の下限

出力先（既定）:

- `artifacts/phase0/checkpoints/best_model.pt`
- `artifacts/phase0/reports/data_report.json`
- `artifacts/phase0/reports/training_history.json`
- `artifacts/phase0/reports/evaluation.json`

### 3) 推論

```bash
uv run python main.py encode \
  --checkpoint artifacts/phase0/checkpoints/best_model.pt \\
  --text "今日は楽しかった！"

# 連続軸（Arousal/Valence）を直接出力
uv run python main.py encode \
  --checkpoint artifacts/phase0/checkpoints/best_model.pt \
  --text "今日は楽しかった！" \
  --output-format av2d

# 8D と連続軸を同時出力
uv run python main.py encode \
  --checkpoint artifacts/phase0/checkpoints/best_model.pt \
  --text "今日は楽しかった！" \
  --output-format both
```

## Python API

```python
from emotionbridge import EmotionEncoder

encoder = EmotionEncoder("artifacts/phase0/checkpoints/best_model.pt", device="cuda")
vec = encoder.encode("今日は楽しかった！")
av = encoder.encode_av("今日は楽しかった！")
batch = encoder.encode_batch(["嬉しい", "少し不安"])
batch_av = encoder.encode_batch_av(["嬉しい", "少し不安"])
```

出力次元順序は固定で、`[joy, sadness, anticipation, surprise, anger, fear, disgust, trust]` です。
連続軸は `[arousal, valence]` 順で出力されます。

### 4) Phase 2: 三つ組スコア付与（Approach B+）

Phase 1 で生成済みの `triplet_dataset.parquet` を入力として、
emotion2vec+ で `feats` / `logits` を抽出し、
共通6感情マッピングによる `ser_score` を付与します。

```bash
uv run python main.py score-triplets --config configs/phase2_triplet.yaml
```

出力（既定）:

- `artifacts/phase2/triplets/triplet_dataset_scored.parquet`
- `artifacts/phase2/triplets/ser_embeddings.npz`
- `artifacts/phase2/triplets/metadata.json`

### 5) Phase 3: 韻律特徴空間検証（V-01 / V-02 / V-03）

実験スクリプトは `emotionbridge/scripts/` 配下に追加しています。
セットアップと実行順序は以下を参照してください。

V-03 では、`prepare_direct_matching` により感情ごとの推奨5D制御プロファイル
（韻律特徴空間ベース）を生成できます。

- `configs/experiment_config.yaml`
- `docs/phase3/prosody_validation_setup.md`


