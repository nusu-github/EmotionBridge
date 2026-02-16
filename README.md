# EmotionBridge

Phase 0「テキスト感情エンコーダ」の初期実装です。設計書 `EB-P0-BD-001` をベースに、以下を実装しています。

- WRIME Ver1 の読込・前処理（客観ラベル抽出 / フィルタ / 正規化 / 分割）
- BERT + 回帰ヘッドによる 8 次元感情ベクトル推定
- 訓練ループ（MSE, AdamW, warmup+decay, Early Stopping, checkpoint）
- 評価指標（MSE, Pearson r, Top-1 Accuracy）と Go/No-Go 判定
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
```

## Python API

```python
from emotionbridge import EmotionEncoder

encoder = EmotionEncoder("artifacts/phase0/checkpoints/best_model.pt", device="cuda")
vec = encoder.encode("今日は楽しかった！")
batch = encoder.encode_batch(["嬉しい", "少し不安"])
```

出力次元順序は固定で、`[joy, sadness, anticipation, surprise, anger, fear, disgust, trust]` です。

