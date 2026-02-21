# EmotionBridge

テキストを入力するだけで、感情表現付きの音声を自動生成する変換エンジン。テキスト感情分類と韻律特徴マッチングに基づき、TTS の制御パラメータを自動決定する。

## 概要

EmotionBridge は日本語テキストから感情を分類し、その結果に基づいて VOICEVOX TTS の制御パラメータを自動決定して感情音声を合成する。

```text
テキスト
  --> Phase 0 感情分類（BERT 6クラス）
    --> DeterministicMixer（6D --> 5D制御パラメータ）
      --> VOICEVOX TTS（スタイル選択 + 韻律制御）
        --> (任意) DSP後処理（WORLDベース声質制御）
          --> 感情音声
```

### フェーズ構成

| フェーズ | 内容                                                                                                                                       |
| -------- | ------------------------------------------------------------------------------------------------------------------------------------------ |
| Phase 0  | 日本語テキスト感情分類器。BERT + 分類ヘッド、WRIME データセットで学習。6感情クラス（anger, disgust, fear, happy, sad, surprise）           |
| Phase 1  | VOICEVOX TTS 統合。音声サンプル生成パイプライン。5D制御パラメータ（pitch_shift, pitch_range, speed, energy, pause_weight）                 |
| Phase 3  | 韻律特徴ワークフロー。JVNV/VOICEVOX の eGeMAPS 抽出・正規化・クロスドメイン整合・マッチングにより教師表を作成し、DeterministicMixer を学習 |
| Bridge   | 統合推論パイプライン。テキスト入力から感情分類・パラメータ生成・スタイル選択・音声合成までを一括実行                                       |

## セットアップ

```bash
# 依存関係のインストール（uv パッケージマネージャを使用）
uv sync
```

Bridge パイプラインおよび Phase 1 の音声合成には VOICEVOX Engine の起動が必要:

```bash
# VOICEVOX Engine をデフォルトポート（50021）で起動しておく
# https://voicevox.hiroshiba.jp/
```

## デモ音声

`demo/` に感情別のサンプル音声を収録している。Bridge パイプラインの出力を確認する場合はこちらを参照。

| ファイル     | テキスト                 | 分類結果 | 確信度 | スタイル | 主な制御パラメータ           |
| ------------ | ------------------------ | -------- | ------ | -------- | ---------------------------- |
| happy.wav    | 今日は本当に楽しかった！ | happy    | 99.9%  | あまあま | speed+0.28, pitch+0.23       |
| anger.wav    | もう絶対に許さないからね | anger    | 58.7%  | ノーマル | pitch_range+0.22, pitch-0.20 |
| sad.wav      | あの人がいなくなって…    | sad      | 99.7%  | セクシー | energy-0.68, speed-0.23      |
| fear.wav     | 怖い…誰かいるの…？       | fear     | 97.2%  | セクシー | energy-0.51, pause+0.38      |
| surprise.wav | えっ！？嘘でしょ！？     | surprise | 98.0%  | セクシー | pause+0.39, energy-0.37      |
| disgust.wav  | あんなことするなんて…    | disgust  | 88.0%  | ノーマル | energy-0.44, pitch-0.28      |

> デモ音声は VOICEVOX:ずんだもん を使用して生成されている。

## クイックスタート

テキストから感情音声を生成する最短例:

```bash
uv run python main.py bridge \
  --text "今日は楽しかった！" \
  --output output.wav \
  --character zundamon
```

感情分類結果、制御パラメータ、選択されたスタイル情報が JSON で出力され、音声ファイルが保存される。

DSP後処理を有効化する場合:

```bash
uv run python main.py bridge \
  --text "今日は楽しかった！" \
  --output output_dsp.wav \
  --character zundamon \
  --enable-dsp
```

## CLI リファレンス

### Phase 0: テキスト感情分類器

```bash
# 訓練
uv run python main.py train --config configs/classifier.yaml

# Accelerate 使用（推奨）
uv run accelerate launch main.py train --config configs/classifier.yaml

# データ分析
uv run python main.py analyze-data --config configs/classifier.yaml

# 推論
uv run python main.py encode \
  --checkpoint artifacts/classifier/checkpoints/best_model \
  --text "今日は楽しかった！"
```

Phase 0 データ前処理仕様（2026-02 リファクタ後）:
- WRIME ラベル入力は `avg_readers` のネスト構造のみ受け付ける（flatキー互換は廃止）。
- train/val/test 分割は常に stratified split を使用し、成立しない場合は即時エラーで停止する。

### Phase 1: VOICEVOX 音声サンプル生成（要 VOICEVOX Engine）

```bash
# サンプル生成
uv run python main.py generate-samples --config configs/audio_gen.yaml

# 利用可能なキャラクター一覧
uv run python main.py list-speakers
```

`configs/audio_gen*.yaml` の `voicevox.base_url` で接続先を指定する（`http://` / `https://` 両対応）。

```yaml
voicevox:
  base_url: https://your-voicevox.example.com
```

### Phase 3: 韻律特徴パイプライン

実行順序に注意。上から順に実行する。

```bash
# JVNV データセット準備
uv run python -m emotionbridge.scripts.prepare_jvnv --config configs/experiment_config.yaml

# eGeMAPS 特徴量抽出
uv run python -m emotionbridge.scripts.extract_egemaps --config configs/experiment_config.yaml --source jvnv
uv run python -m emotionbridge.scripts.extract_egemaps --config configs/experiment_config.yaml --source voicevox

# 特徴量正規化
uv run python -m emotionbridge.scripts.normalize_features --config configs/experiment_config.yaml --source jvnv
uv run python -m emotionbridge.scripts.normalize_features --config configs/experiment_config.yaml --source voicevox

# 特徴応答性評価（feature_weights.json 出力）
uv run python -m emotionbridge.scripts.evaluate_responsiveness --config configs/experiment_config.yaml

# クロスドメイン整合
uv run python -m emotionbridge.scripts.align_domains --config configs/experiment_config.yaml

# 感情パラメータマッチング
uv run python -m emotionbridge.scripts.match_emotion_params --config configs/experiment_config.yaml

# ドメインギャップ評価
uv run python -m emotionbridge.scripts.evaluate_domain_gap --config configs/experiment_config.yaml

# 教師表作成
uv run python -m emotionbridge.scripts.prepare_generator_teacher

# Generator 訓練
uv run python -m emotionbridge.scripts.train_generator --config configs/generator.yaml

# スタイルマッピング構築
uv run python -m emotionbridge.scripts.build_style_mapping --config configs/experiment_config.yaml

# スタイルマッピング手動調整（dry-run）
uv run python -m emotionbridge.scripts.adjust_style_mapping \
  --mapping-path artifacts/prosody/style_mapping.json \
  --character zundamon \
  --set anger=7,happy=1 \
  --default-style-id 3 \
  --dry-run

# スタイルマッピング手動調整（本適用）
uv run python -m emotionbridge.scripts.adjust_style_mapping \
  --mapping-path artifacts/prosody/style_mapping.json \
  --character zundamon \
  --set anger=7,happy=1 \
  --default-style-id 3
```

### Bridge: 統合推論（要 VOICEVOX Engine）

```bash
uv run python main.py bridge \
  --text "今日は楽しかった！" \
  --output output.wav \
  --character zundamon \
  --classifier-checkpoint artifacts/classifier/checkpoints/best_model \
  --generator-checkpoint artifacts/generator/checkpoints/best_generator.pt \
  --style-mapping artifacts/prosody/style_mapping.json \
  --enable-dsp \
  --dsp-features-path artifacts/prosody/v01/jvnv_egemaps_normalized.parquet \
  --dsp-f0-extractor harvest
```

主要オプション:

| オプション             | 既定値                                                  | 説明                                                           |
| ---------------------- | ------------------------------------------------------- | -------------------------------------------------------------- |
| `--character`          | `zundamon`                                              | スタイルマッピング内のキャラクターキー                         |
| `--fallback-threshold` | `0.3`                                                   | 感情確信度がこの値未満の場合デフォルトスタイルにフォールバック |
| `--enable-dsp`         | `False`                                                 | WORLDベースのDSP後処理を有効化                                 |
| `--dsp-features-path`  | `artifacts/prosody/v01/jvnv_egemaps_normalized.parquet` | EmotionDSPMapper 初期化用の特徴量parquet                       |
| `--dsp-f0-extractor`   | `dio`                                                   | DSP段のWORLD解析で使うF0抽出法（`dio` または `harvest`）       |
| `--device`             | `cuda`                                                  | 推論デバイス（`cuda` または `cpu`）                            |
| `--voicevox-url`       | `http://127.0.0.1:50021`                                | VOICEVOX Engine URL（`http://` / `https://`）                  |

`bridge` のJSON出力には以下のDSP関連項目が追加される:

- `dsp_params`: 適用したDSP制御量（未適用時は `null`）
- `dsp_applied`: DSPが適用されたかどうか
- `dsp_seed`: 決定論シード（未適用時は `null`）

### 主観評価

```bash
# 評価刺激と回答テンプレートを生成
uv run python -m emotionbridge.scripts.prepare_subjective_eval \
  --dataset-path artifacts/audio_gen_multistyle_smoke/dataset/triplet_dataset.parquet \
  --output-dir artifacts/prosody/subjective_eval/pilot_v01 \
  --character zundamon \
  --classifier-checkpoint artifacts/classifier/checkpoints/best_model \
  --generator-checkpoint artifacts/generator/checkpoints/best_generator.pt \
  --style-mapping artifacts/prosody/style_mapping.json

# 回答CSV集計（responses/*.csv 配置後）
uv run python -m emotionbridge.scripts.analyze_subjective_eval \
  --eval-dir artifacts/prosody/subjective_eval/pilot_v01
```

### 定量評価（Roundtrip）

`demo/v2`（baseline）と`demo/v2-dsp`（candidate）を比較して、PESQ / MCD / F0 RMSE を算出する。

```bash
uv run python -m emotionbridge.scripts.evaluate_roundtrip \
  --baseline-manifest demo/v2/manifest.json \
  --candidate-manifest demo/v2-dsp/manifest.json \
  --output-dir artifacts/prosody/roundtrip_eval/v2_dsp
```

出力ファイル:

- `roundtrip_metrics.json`: 全指標・Go/No-Go判定・ファイル別詳細
- `roundtrip_report.md`: 人間可読サマリ
- `per_file_metrics.csv`: ファイル単位メトリクス

Go/No-Go閾値:

- `PESQ mean >= 3.5`
- `MCD mean <= 6.0 dB`
- `F0 RMSE mean <= 5.0 Hz`

Harvest比較（`demo/v2-dsp-harvest`）:

```bash
uv run python -m emotionbridge.scripts.evaluate_roundtrip \
  --baseline-manifest demo/v2/manifest.json \
  --candidate-manifest demo/v2-dsp-harvest/manifest.json \
  --output-dir artifacts/prosody/roundtrip_eval/v2_dsp_harvest
```

## Python API

### EmotionEncoder（感情分類）

```python
from emotionbridge import EmotionEncoder

encoder = EmotionEncoder("artifacts/classifier/checkpoints/best_model", device="cuda")
probs = encoder.encode("今日は楽しかった！")          # numpy array (6,)
batch = encoder.encode_batch(["嬉しい", "少し不安"])  # numpy array (N, 6)
```

### Bridge Pipeline（統合推論）

```python
from emotionbridge.inference import create_pipeline

pipeline = await create_pipeline(
    classifier_checkpoint="artifacts/classifier/checkpoints/best_model",
    generator_checkpoint="artifacts/generator/checkpoints/best_generator.pt",
    style_mapping="artifacts/prosody/style_mapping.json",
    voicevox_url="https://your-voicevox.example.com",
    character="zundamon",
    fallback_threshold=0.3,
    enable_dsp=True,
    dsp_features_path="artifacts/prosody/v01/jvnv_egemaps_normalized.parquet",
    dsp_f0_extractor="harvest",
    device="cuda",
)

result = await pipeline.synthesize(
    text="今日は楽しかった！",
    output_path="output.wav",
)

print(result.dominant_emotion)  # "happy"
print(result.control_params)   # {"pitch_shift": 0.12, ...}
print(result.dsp_params)       # {"jitter_amount": 0.18, ...} or None
print(result.style_name)       # "うれしい"

await pipeline.close()
```

## プロジェクト構成

```text
emotionbridge/
├── cli.py                  # メインCLI
├── config.py               # Classifier/AudioGen 設定
├── constants.py            # 感情ラベル・制御パラメータ定数
├── data/                   # WRIME データ処理
├── model/                  # DeterministicMixer, ParameterGenerator
├── training/               # Phase 0 訓練, generator_trainer
├── inference/              # EmotionEncoder, bridge_pipeline
├── generation/             # Phase 1 生成パイプライン
├── tts/                    # VOICEVOX クライアント・アダプタ
└── scripts/                # Phase 3 スクリプト群
configs/                    # YAML設定ファイル
artifacts/                  # 訓練成果物・チェックポイント
docs/                       # 設計書・USDM仕様書
```

## ライセンス

本プロジェクトのソースコードのライセンスは未定。

本プロジェクトは以下の外部リソースに依存しており、それぞれ固有のライセンス・利用規約が適用される。利用前に各ライセンスの確認を推奨する。

- **VOICEVOX Engine / キャラクター音声**: [VOICEVOX 利用規約](https://voicevox.hiroshiba.jp/)（ずんだもんの音声は「VOICEVOX:ずんだもん」のクレジット表記により商用・非商用利用可能）
- **WRIME データセット**: [CC BY-NC-ND 4.0](https://creativecommons.org/licenses/by-nc-nd/4.0/)（非商用・改変不可）
- **JVNV コーパス**: [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/)
- **sbintuitions/modernbert-ja-70m**: [MIT](https://choosealicense.com/licenses/mit/)
- **openSMILE (eGeMAPS)**: 研究・教育用途はオープンソース。商用利用には [audEERING 商用ライセンス](https://www.audeering.com/) が必要
