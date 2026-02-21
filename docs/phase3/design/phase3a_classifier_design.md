# Phase 3a 詳細設計: テキスト感情分類器（EB3-C01）

> **対象要求**: EB3-C01（Phase 3 USDM）、EB-P0-01（全体USDM）
> **設計対象**: Phase 0 再設計 — 8D Plutchik回帰 → 6クラス感情分類
> **作成日**: 2026-02-18
> **ステータス**: 初版
> **追記（2026-02-21）**: 現行実装は soft label学習固定。argmaxラベル学習、転移学習、class weight、Phase 0単体Go/No-Goは廃止済み。

---

## 1. 設計の背景と目的

### 1.1 現状（旧Phase 0）

- **モデル**: `TextEmotionRegressor` — BERT [CLS] → Dropout → Linear(768, 256) → ReLU → Dropout → Linear(256, 8) → Sigmoid
- **出力**: 8D連続値ベクトル（Plutchikの8感情: joy, sadness, anticipation, surprise, anger, fear, disgust, trust）
- **損失**: 重み付きMSE（inverse_mean方式）
- **問題**: trustのPearson相関0.36でNo-Go判定。anticipation/trustはJVNVに対応する感情カテゴリが存在せず構造的に不要

### 1.2 新設計の目的

- 出力を6クラス分類（anger, disgust, fear, happy, sad, surprise）に変更
- JVNVの感情ラベルと1対1対応を実現
- Softmax出力により確率分布（合計1.0）を直接得る
- 不要な感情軸（trust, anticipation）の干渉を排除

### 1.3 関連仕様ID

| 仕様ID | 内容 |
|--------|------|
| EB3-C01-01-001 | joy→happy, sadness→sad, anger→anger, fear→fear, disgust→disgust, surprise→surpriseの直接対応 |
| EB3-C01-01-002 | anticipation, trustを除外 |
| EB3-C01-01-003 | **決定**: WRIMEの連続強度→6感情soft labelを本採用し、学習はsoft targetで統一 |
| EB3-C01-02-001 | sbintuitions/modernbert-ja-70m を使用 |
| EB3-C01-02-002 | 6クラス分類ヘッド（Softmax） |
| EB-P0-01-04-002 | Phase 0単体の固定閾値Go/No-Goは廃止し、確率分布指標を継続監視 |

---

## 2. WRIMEラベル変換方式（確定仕様）

WRIMEコーパスは8感情（Plutchik）の客観強度（avg_readers、0--3の連続値）を持つ。EmotionBridgeでは、これを6感情（JVNV順）へ写像した後、**soft label（確率分布）を唯一の教師信号**として使用する。

### 2.1 確定方針

- 学習データは hard label（argmax）を保持しない
- soft labelは強度ベクトルの正規化により生成し、温度パラメータで分布の鋭さを制御する
- 損失関数は soft target に対する cross entropy を採用する

### 2.2 データ分割

- train/val/test は stratified split を維持する
- 層化キーは hard label ではなく **soft labelクラスタID** とする
- 層化が成立しない場合は fail-fast で停止する

### 2.3 評価指標

- mean KL divergence
- mean cross entropy
- Brier score
- key emotion MAE / per-emotion MAE

Phase 0単体では固定閾値によるGo/No-Go判定を行わず、指標を記録して継続監視する。

---

## 3. モデルアーキテクチャ

### 3.1 既存 TextEmotionRegressor からの変更箇所

| 層 | 旧（regressor.py） | 新（classifier） |
|----|---------------------|-------------------|
| BERT encoder | `AutoModel.from_pretrained(...)` | **変更なし** |
| Dropout1 | `nn.Dropout(dropout)` | **変更なし** |
| FC1 | `nn.Linear(768, 256)` | **変更なし** |
| ReLU | `nn.ReLU()` | **変更なし** |
| Dropout2 | `nn.Dropout(dropout)` | **変更なし** |
| FC2 | `nn.Linear(256, 8)` | `nn.Linear(256, 6)` |
| 出力活性化 | `nn.Sigmoid()` | **削除**（LogSoftmax/CrossEntropyが内部処理） |

### 3.2 新クラス設計: TextEmotionClassifier

新ファイル `emotionbridge/model/classifier.py` を作成する。

```python
class TextEmotionClassifier(nn.Module):
    def __init__(
        self,
        pretrained_model_name: str,
        num_classes: int = 6,
        bottleneck_dim: int = 256,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.encoder = AutoModel.from_pretrained(pretrained_model_name)
        hidden_size = int(self.encoder.config.hidden_size)

        self.dropout1 = nn.Dropout(dropout)
        self.fc1 = nn.Linear(hidden_size, bottleneck_dim)
        self.relu = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)
        self.fc2 = nn.Linear(bottleneck_dim, num_classes)
        # 注意: forward()はlogitsを返す。Softmaxは推論時にのみ適用
        # CrossEntropyLossがlogitsを直接受け取るため

    def forward(self, input_ids, attention_mask, token_type_ids=None):
        # ... BERTエンコード（既存と同一）...
        cls_embedding = outputs.last_hidden_state[:, 0, :]
        x = self.dropout1(cls_embedding)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout2(x)
        logits = self.fc2(x)
        return logits  # (batch_size, num_classes)

    def predict_proba(self, input_ids, attention_mask, token_type_ids=None):
        """推論時に確率分布を返す"""
        logits = self.forward(input_ids, attention_mask, token_type_ids)
        return torch.softmax(logits, dim=-1)

    def head_parameters(self):
        """分類ヘッドのパラメータ（differential LR用）"""
        modules = [self.dropout1, self.fc1, self.relu, self.dropout2, self.fc2]
        for module in modules:
            yield from module.parameters()
```

### 3.3 設計判断

- **logits出力**: `forward()`はlogitsを返し、`nn.CrossEntropyLoss`が内部的にlog_softmax + NLLLossを計算する。これにより数値的に安定する
- **predict_proba()**: 推論時に`torch.softmax()`を適用して確率分布を得る。`EmotionEncoder`から呼び出す
- **num_classes引数**: デフォルト6だが引数化して柔軟性を持たせる
- **既存のTextEmotionRegressorは残す**: 旧Phase 0のチェックポイントロードに必要。削除はしない

### 3.4 既存BERT重みの再利用方針

| 方式 | 説明 | 推奨度 |
|------|------|--------|
| **スクラッチ学習** | HuggingFace事前学習BERTから分類ヘッドを学習 | 推奨（ベースライン） |
| **旧Phase 0からの転移** | 旧チェックポイントのBERTエンコーダ重みをロードし、分類ヘッドのみ再初期化 | 実験的に比較 |

転移学習の効果見込み:
- 旧Phase 0はWRIMEデータで8D感情回帰を学習済み → BERT部分は感情関連の特徴を既に抽出可能
- ただし出力層の構造が根本的に異なる（回帰→分類）ため、エンコーダの特徴表現がそのまま最適とは限らない
- 実験として旧重みからのfine-tuningを試み、HuggingFace事前学習からの学習と比較する

---

## 4. 学習パイプラインの変更

### 4.1 損失関数

```python
# 方式A: hard label
criterion = nn.CrossEntropyLoss(weight=class_weights)
loss = criterion(logits, labels)  # labels: LongTensor (batch_size,)

# 方式C: soft label
def soft_cross_entropy(logits, soft_targets, temperature=1.0):
    log_probs = F.log_softmax(logits / temperature, dim=-1)
    loss = -(soft_targets * log_probs).sum(dim=-1).mean()
    return loss
```

### 4.2 クラス重み付き CrossEntropyLoss

クラス不均衡に対処するため、`nn.CrossEntropyLoss(weight=class_weights)`を使用する。重み計算方式:

```python
# inverse frequency方式
class_counts = np.bincount(train_labels, minlength=6)
class_weights = 1.0 / np.maximum(class_counts, 1)
class_weights = class_weights / class_weights.mean()  # 正規化
class_weights_tensor = torch.tensor(class_weights, dtype=torch.float32)
```

既存の`emotion_weight_mode: inverse_mean`に相当する仕組みをクラス頻度ベースに置き換える。

### 4.3 WRIMEデータフィルタリングとクラス分布推定

#### フィルタリングルール

既存ルール（`filter_max_intensity_lte: 1`）を維持: 8感情すべての客観強度が1以下のサンプルを除外。

これは6感情のみで見ても同様に適用する（8感情の最大が1以下なら、当然6感情の最大も1以下）。

#### クラス分布の実測（argmax方式, 2026-02-18）

`filter_max_intensity_lte=1` 適用後（N=18,237）の6感情argmax分布は以下。

| 感情 | 件数 | 比率 |
|------|------|------|
| anger | 2,490 | 13.65% |
| disgust | 1,521 | 8.34% |
| fear | 2,039 | 11.18% |
| happy | 7,035 | 38.58% |
| sad | 3,122 | 17.12% |
| surprise | 2,030 | 11.13% |

補足:

- 最大/最小クラス比は `7035 / 1521 = 4.63`（中程度の不均衡）
- train/val/testの層化後比率はほぼ同一（train, val, testで小数第3位まで一致）

**評価**: happy優勢はあるが、anger/fear/disgustが「極少数」ではないため、Go/No-Go達成可能性は十分ある。

**運用方針（初期）**:

1. クラス重み付き損失（`inverse_frequency`）を有効化
2. `WeightedRandomSampler` は初期OFF（学習が不安定な場合のみON）
3. フィルタ閾値緩和は初回学習結果を見て判断

### 4.4 データ分割の層化キー

```python
# 6感情のargmax（変換後ラベル）で層化
six_emotion_cols = ["joy", "sadness", "anger", "fear", "disgust", "surprise"]
stratify_key = np.argmax(filtered_raw_targets[:, six_indices], axis=1)
```

既存の`build_phase0_splits()`で`stratify_key = np.argmax(filtered_raw_targets, axis=1)`を8D→6Dに変更する。

### 4.5 評価指標

| 指標 | 計算方法 | 用途 |
|------|----------|------|
| accuracy | `(pred == true).mean()` | 全体精度 |
| macro F1 | `sklearn.metrics.f1_score(average='macro')` | クラス不均衡に頑健な全体指標 |
| 感情別F1 | `sklearn.metrics.f1_score(average=None)` | 各感情の個別性能 |
| 感情別precision/recall | 同上 | エラー分析用 |
| confusion matrix | `sklearn.metrics.confusion_matrix()` | 誤分類パターンの分析 |

新たに`compute_classification_metrics()`関数を作成する（既存の`compute_regression_metrics()`は旧Phase 0用に残す）。

```python
def compute_classification_metrics(
    logits: np.ndarray,      # (N, 6)
    true_labels: np.ndarray,  # (N,) int
    label_names: list[str],   # JVNV_EMOTION_LABELS
) -> dict[str, Any]:
    pred_labels = np.argmax(logits, axis=1)
    return {
        "accuracy": float(np.mean(pred_labels == true_labels)),
        "macro_f1": float(f1_score(true_labels, pred_labels, average="macro")),
        "per_class_f1": {
            name: float(f1)
            for name, f1 in zip(label_names, f1_score(true_labels, pred_labels, average=None))
        },
        "per_class_precision": { ... },
        "per_class_recall": { ... },
        "confusion_matrix": confusion_matrix(true_labels, pred_labels).tolist(),
    }
```

### 4.6 Go/No-Go基準

USDM仕様（EB-P0-01-04-002）に基づく:

| 条件 | 閾値 | 備考 |
|------|------|------|
| マクロF1 | >= 0.40 | 6クラス全体 |
| anger 個別F1 | >= 0.50 | 韻律分離が明確な3感情 |
| happy 個別F1 | >= 0.50 | 同上 |
| sad 個別F1 | >= 0.50 | 同上 |

```python
def _go_no_go_classifier(metrics: dict, config) -> dict:
    checks = {
        "macro_f1": metrics["macro_f1"] >= config.eval.go_macro_f1_min,
        "anger_f1": metrics["per_class_f1"]["anger"] >= config.eval.go_key_emotion_f1_min,
        "happy_f1": metrics["per_class_f1"]["happy"] >= config.eval.go_key_emotion_f1_min,
        "sad_f1": metrics["per_class_f1"]["sad"] >= config.eval.go_key_emotion_f1_min,
    }
    return {**checks, "go": all(checks.values()), ...}
```

---

## 5. 推論インターフェース

### 5.1 EmotionEncoderの変更

現在の`EmotionEncoder`（`inference/encoder.py`）は`TextEmotionRegressor`を使用し、8D Sigmoid出力を返す。新設計では:

```python
class EmotionEncoder:
    def __init__(self, checkpoint_path: str, device: str = "cuda"):
        # チェックポイントの種類を自動判別
        checkpoint = torch.load(checkpoint_path, map_location=device)

        model_type = checkpoint.get("model_type", "regressor")  # 後方互換

        if model_type == "classifier":
            self.model = TextEmotionClassifier(...)
            self._is_classifier = True
            self._label_names = checkpoint["emotion_labels"]
            # JVNV_EMOTION_LABELS: ["anger", "disgust", "fear", "happy", "sad", "surprise"]
        else:
            self.model = TextEmotionRegressor(...)
            self._is_classifier = False
            self._label_names = checkpoint.get("emotion_labels", EMOTION_LABELS)

        self.model.load_state_dict(checkpoint["model_state_dict"])

    def encode(self, text: str) -> np.ndarray:
        """テキストから感情ベクトルを返す。

        - classifier: 6D確率ベクトル (softmax出力, 合計1.0)
        - regressor (旧): 8D連続値ベクトル (sigmoid出力)
        """
        result = self.encode_batch([text])
        return result[0]

    def encode_batch(self, texts: list[str], batch_size: int = 32) -> np.ndarray:
        outputs = []
        with torch.no_grad():
            for start in range(0, len(texts), batch_size):
                batch_texts = texts[start : start + batch_size]
                encoded = self.tokenizer(batch_texts, ...)
                encoded = {k: v.to(self.device) for k, v in encoded.items()}

                if self._is_classifier:
                    preds = self.model.predict_proba(**encoded)
                else:
                    preds = self.model(**encoded)
                outputs.append(preds.detach().cpu().numpy())

        return np.vstack(outputs).astype(np.float32)

    @property
    def label_names(self) -> list[str]:
        return self._label_names

    @property
    def num_emotions(self) -> int:
        return len(self._label_names)
```

### 5.2 後方互換性: 旧8D出力が必要な箇所の影響分析

| モジュール | 現在の8D依存箇所 | 影響 | 対応 |
|-----------|-----------------|------|------|
| `inference/axes.py` | `emotion8d_to_av()`, `emotion8d_batch_to_av()` | Arousal/Valence変換は8D前提 | Phase 3では使用しない。Phase 3のパイプラインは6D確率→パラメータ生成器（EB3-D01）→5Dパラメータ。A/V変換は旧互換として残す |
| `inference/encoder.py` | `encode_av()`, `encode_batch_av()` | 8D出力前提のA/V変換 | classifierモードでは`encode_av()`を使用不可とする（`NotImplementedError`またはCOMMON6_CIRCUMPLEX_COORDSを使用した6D版を新設） |
| `training/trainer.py` | `train_phase0()` | 8D回帰の学習パイプライン | 旧Phase 0用に残す。新分類器用に`train_phase0_classifier()`を新設 |
| `training/metrics.py` | `compute_regression_metrics()` | 8D回帰用メトリクス | 旧用に残す。分類用の`compute_classification_metrics()`を新設 |
| `data/wrime.py` | `build_phase0_splits()` | 8D正規化ターゲット | 6クラスラベル生成用の`build_phase0_classifier_splits()`を新設 |
| `constants.py` | `EMOTION_LABELS` (8個) | 全体で参照 | `JVNV_EMOTION_LABELS` (6個)は既に定義済み。分類器関連では`JVNV_EMOTION_LABELS`を使用 |
| Phase 1パイプライン | `EmotionEncoder.encode()` → 8D | Phase 1のtriplet生成時に感情ベクトルをメタデータに記録 | Phase 1は完了済み。再実行は不要 |

### 5.3 constants.py の変更

`JVNV_EMOTION_LABELS`は既に定義済み:
```python
JVNV_EMOTION_LABELS = ["anger", "disgust", "fear", "happy", "sad", "surprise"]
```

追加が必要な定数:
```python
# WRIMEの8感情からJVNV 6感情へのマッピング
WRIME_TO_JVNV_MAPPING: dict[str, str] = {
    "joy": "happy",
    "sadness": "sad",
    "anger": "anger",
    "fear": "fear",
    "disgust": "disgust",
    "surprise": "surprise",
}

# WRIMEの8感情のうちJVNV 6感情に含まれるもののインデックス
# EMOTION_LABELS = [joy, sadness, anticipation, surprise, anger, fear, disgust, trust]
WRIME_6EMOTION_INDICES: list[int] = [0, 1, 3, 4, 5, 6]
# 対応: joy(0), sadness(1), surprise(3), anger(4), fear(5), disgust(6)
# → JVNV順 [anger, disgust, fear, happy, sad, surprise] への並び替えも定義

NUM_JVNV_EMOTIONS = len(JVNV_EMOTION_LABELS)

# Go/No-Go基準で重視する感情
KEY_EMOTION_LABELS = ["anger", "happy", "sad"]
```

---

## 6. 設定ファイル

### 6.1 新設定ファイル: configs/phase0_classifier.yaml

```yaml
data:
  dataset_name: shunk031/wrime
  dataset_config_name: ver1
  text_field: sentence
  label_source: avg_readers
  max_length: 128
  use_official_split: false
  filter_max_intensity_lte: 1
  stratify_after_filter: true
  train_ratio: 0.8
  val_ratio: 0.1
  test_ratio: 0.1
  random_seed: 42
  # 新規パラメータ
  label_conversion: argmax  # argmax | soft_label
  soft_label_temperature: 1.0  # soft_label方式の場合のみ

model:
  pretrained_model_name: sbintuitions/modernbert-ja-70m
  bottleneck_dim: 256
  dropout: 0.1
  num_classes: 6
  # 旧Phase 0重みからの転移学習
  transfer_from: null  # artifacts/phase0/checkpoints/best_model.pt を指定すると転移学習

train:
  output_dir: artifacts/phase0_v2
  batch_size: 32
  num_epochs: 10
  bert_lr: 0.00002
  head_lr: 0.001
  weight_decay: 0.01
  warmup_ratio: 0.1
  early_stopping_patience: 3
  device: cuda
  num_workers: 2
  pin_memory: true
  log_every_steps: 50
  gradient_accumulation_steps: 2
  mixed_precision: fp16
    # soft label学習固定のため class weight / sampler は設定しない
```

### 6.2 Config dataclass: Phase0ClassifierConfig

```python
@dataclass(slots=True)
class ClassifierDataConfig(DataConfig):
    """分類タスク用のデータ設定（DataConfigを継承）"""
    soft_label_temperature: float = 1.0

@dataclass(slots=True)
class ClassifierModelConfig(ModelConfig):
    """分類タスク用のモデル設定"""
    num_classes: int = 6

@dataclass(slots=True)
class ClassifierTrainConfig:
    """分類タスク用の学習設定"""
    output_dir: str = "artifacts/phase0_v2"
    batch_size: int = 32
    num_epochs: int = 10
    bert_lr: float = 2e-5
    head_lr: float = 1e-3
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1
    early_stopping_patience: int = 3
    device: str = "cuda"
    num_workers: int = 2
    pin_memory: bool = True
    log_every_steps: int = 50
    gradient_accumulation_steps: int = 1
    mixed_precision: str = "no"

@dataclass(slots=True)
class Phase0ClassifierConfig:
    data: ClassifierDataConfig = field(default_factory=ClassifierDataConfig)
    model: ClassifierModelConfig = field(default_factory=ClassifierModelConfig)
    train: ClassifierTrainConfig = field(default_factory=ClassifierTrainConfig)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
```

### 6.3 load_config() の拡張

既存の`load_config()`に分類器設定の自動判別を追加:

```python
def load_config(config_path):
    raw = yaml.safe_load(...)

    if "voicevox" in raw:
        return Phase1Config(...)

    # 分類器設定の判別: num_classes / soft_label_temperature / bert_lr / head_lr
    model_raw = raw.get("model", {})
    data_raw = raw.get("data", {})
    train_raw = raw.get("train", {})
    if (
        "num_classes" in model_raw
        or "soft_label_temperature" in data_raw
        or "bert_lr" in train_raw
        or "head_lr" in train_raw
    ):
        return Phase0ClassifierConfig(...)

    return Phase0Config(...)
```

---

## 7. ファイル構成の変更サマリ

### 新規作成ファイル

| ファイル | 内容 |
|----------|------|
| `emotionbridge/model/classifier.py` | `TextEmotionClassifier` クラス |
| `emotionbridge/training/classifier_trainer.py` | `train_phase0_classifier()` 関数 |
| `emotionbridge/training/classification_metrics.py` | `compute_classification_metrics()` 関数 |
| `emotionbridge/data/wrime_classifier.py` | `build_phase0_classifier_splits()` — 6クラスラベル生成 |
| `configs/phase0_classifier.yaml` | 分類器設定ファイル |
| `configs/phase0_classifier_smoke.yaml` | スモークテスト用 |

### 変更ファイル

| ファイル | 変更内容 |
|----------|----------|
| `emotionbridge/constants.py` | `WRIME_TO_JVNV_MAPPING`, `NUM_JVNV_EMOTIONS`, `KEY_EMOTION_LABELS` 追加 |
| `emotionbridge/config.py` | `Phase0ClassifierConfig` 追加、`load_config()` 拡張 |
| `emotionbridge/inference/encoder.py` | classifier/regressorの自動判別、`predict_proba()`対応 |
| `emotionbridge/model/__init__.py` | `TextEmotionClassifier` のエクスポート追加 |
| `main.py` | `train` コマンドの分類器対応、新コマンド追加 |

### 変更しないファイル（後方互換性のため維持）

| ファイル | 理由 |
|----------|------|
| `emotionbridge/model/regressor.py` | 旧Phase 0チェックポイントの読み込みに必要 |
| `emotionbridge/training/trainer.py` | 旧Phase 0の学習パイプラインを維持 |
| `emotionbridge/training/metrics.py` | 旧Phase 0の回帰メトリクスを維持 |
| `emotionbridge/data/wrime.py` | 旧Phase 0のデータローダを維持 |
| `emotionbridge/inference/axes.py` | 8D→A/V変換は将来的に6D版への移行も可能だが現時点では変更不要 |
| `configs/phase0.yaml` | 旧Phase 0設定を維持 |

---

## 8. チェックポイント形式

### 8.1 保存形式

```python
checkpoint = {
    "model_type": "classifier",  # 新規追加。"regressor"との判別用
    "model_state_dict": model.state_dict(),
    "model_config": {
        "pretrained_model_name": "sbintuitions/modernbert-ja-70m",
        "bottleneck_dim": 256,
        "dropout": 0.1,
        "num_classes": 6,
    },
    "tokenizer_name": "sbintuitions/modernbert-ja-70m",
    "max_length": 128,
    "emotion_labels": JVNV_EMOTION_LABELS,  # 8D→6Dに変更
    "label_conversion": "argmax",  # 使用したラベル変換方式
    "config": config.to_dict(),
}
```

### 8.2 旧チェックポイントとの互換性

`EmotionEncoder`が`model_type`キーの有無で自動判別:
- `model_type == "classifier"` → `TextEmotionClassifier`をロード
- `model_type`キーなし（旧形式） → `TextEmotionRegressor`をロード（後方互換）

---

## 9. 決定済み事項と継続Open Issues

### 9.1 本ターンで確定した事項

| # | 事項 | 決定内容 | 反映先 |
|---|------|----------|--------|
| 1 | WRIMEラベル変換方式 (EB3-C01-01-003) | 本採用はargmax。soft labelは補助比較実験として実施 | 2.3, 2.4 |
| 2 | クラス不均衡の実態 | 実測で中程度不均衡（最大/最小比 4.63）。初期対策はclass weight ON / sampler OFF | 4.3 |

### 9.2 継続Open Issues

| # | 事項 | 理由 | 影響 | 決定時期 |
|---|------|------|------|----------|
| 1 | **旧Phase 0重みからの転移学習の効果** | BERTエンコーダの特徴表現が分類タスクでも有効かは実験で確認が必要 | 学習効率と最終性能に影響。効果がなければスクラッチ学習のみでよい | Phase 3a実装時 |
| 2 | **Go/No-Go基準の最終妥当性** | 初回学習結果（macro F1 / key emotion F1）を見て微調整要否を判断する必要がある | Phase 3移行判断に直結 | Phase 3a実装時 |
| 3 | **encode_av()の6D対応** | 6D確率ベクトルからArousal/Valenceを算出する必要があるか（Phase 3のパイプラインでは直接使わないが、分析・可視化で有用な可能性） | 推論APIの完全性 | Phase 3b以降 |

---

## 10. 実装優先順序

1. **データ分析** — WRIMEの6感情argmax分布を確認し、クラス不均衡の程度を定量化
2. **モデル** — `TextEmotionClassifier`の実装
3. **データ** — `build_phase0_classifier_splits()`の実装（argmax方式）
4. **学習** — `train_phase0_classifier()`の実装
5. **評価** — `compute_classification_metrics()`とGo/No-Go判定
6. **推論** — `EmotionEncoder`の分類器対応
7. **実験** — Exp-A（argmax）でベースライン確立
8. **比較実験** — Exp-C1/C2（soft label）、転移学習の効果検証
9. **最終決定** — EB3-C01-01-003の確定、Go/No-Go判定
