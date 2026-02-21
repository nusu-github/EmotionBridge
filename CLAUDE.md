# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

EmotionBridge は日本語テキストから感情を分類し、韻律特徴マッチングに基づく制御パラメータ生成と VOICEVOX TTS を組み合わせて感情音声を合成する変換エンジン。

```
テキスト → 感情分類（6クラス） → DeterministicMixer（6D→5D） → スタイル選択 → VOICEVOX → (任意)DSP後処理 → 感情音声
```

フェーズ構成:

- **Phase 0 (Classifier)**: BERT + 分類ヘッド、WRIME → 6感情クラス（anger, disgust, fear, happy, sad, surprise）の softmax 確率を出力
- **Phase 1 (Audio Gen)**: VOICEVOX 音声サンプル生成パイプライン — (text, control_params, audio) triplet を作成
- **Phase 3 (Prosody)**: JVNV/VOICEVOX の eGeMAPS 抽出・クロスドメイン整合・マッチング → 教師表 → DeterministicMixer
- **Bridge**: 統合推論パイプライン — text → classifier → DeterministicMixer → style selection → VOICEVOX → (optional) DSP → WAV

## Commands

```bash
# 依存関係インストール（uv パッケージマネージャ）
uv sync

# Lint / Format / Type check
uv run ruff check .
uv run ruff format .
uv run ty check

# --- Phase 0: 感情分類器 ---
uv run python main.py train --config configs/classifier.yaml
uv run accelerate launch main.py train --config configs/classifier.yaml  # 推奨
uv run python main.py analyze-data --config configs/classifier.yaml
uv run python main.py encode \
  --checkpoint artifacts/classifier/checkpoints/best_model \
  --text "今日は楽しかった！"

# --- Phase 1: 音声サンプル生成（要 VOICEVOX Engine） ---
uv run python main.py generate-samples --config configs/audio_gen.yaml
uv run python main.py generate-samples --config configs/audio_gen_smoke.yaml  # quick test
uv run python main.py list-speakers

# --- Phase 3: 韻律特徴ワークフロー（上から順に実行） ---
uv run python -m emotionbridge.scripts.prepare_jvnv --config configs/experiment_config.yaml
uv run python -m emotionbridge.scripts.extract_egemaps --config configs/experiment_config.yaml --source jvnv
uv run python -m emotionbridge.scripts.extract_egemaps --config configs/experiment_config.yaml --source voicevox
uv run python -m emotionbridge.scripts.normalize_features --config configs/experiment_config.yaml --source jvnv
uv run python -m emotionbridge.scripts.normalize_features --config configs/experiment_config.yaml --source voicevox
uv run python -m emotionbridge.scripts.evaluate_responsiveness --config configs/experiment_config.yaml
uv run python -m emotionbridge.scripts.align_domains --config configs/experiment_config.yaml
uv run python -m emotionbridge.scripts.match_emotion_params --config configs/experiment_config.yaml
uv run python -m emotionbridge.scripts.evaluate_domain_gap --config configs/experiment_config.yaml
uv run python -m emotionbridge.scripts.prepare_generator_teacher
uv run python -m emotionbridge.scripts.train_generator --config configs/generator.yaml
uv run python -m emotionbridge.scripts.build_style_mapping --config configs/experiment_config.yaml

# --- スタイルマッピング手動調整 ---
uv run python -m emotionbridge.scripts.adjust_style_mapping \
  --mapping-path artifacts/prosody/style_mapping.json \
  --character zundamon \
  --set anger=7,happy=1 \
  --default-style-id 3 \
  --dry-run  # 本適用時は --dry-run を外す

# --- Bridge: 統合推論（要 VOICEVOX Engine） ---
uv run python main.py bridge \
  --text "今日は楽しかった！" \
  --output output.wav \
  --character zundamon
# DSP後処理有効化:
uv run python main.py bridge \
  --text "今日は楽しかった！" \
  --output output_dsp.wav \
  --character zundamon \
  --enable-dsp \
  --dsp-f0-extractor harvest

# --- 評価 ---
# 主観評価（刺激生成 → 回答CSV集計）
uv run python -m emotionbridge.scripts.prepare_subjective_eval \
  --dataset-path artifacts/audio_gen/dataset/triplet_dataset.parquet \
  --output-dir artifacts/prosody/subjective_eval/pilot_v01 \
  --character zundamon \
  --classifier-checkpoint artifacts/classifier/checkpoints/best_model \
  --generator-checkpoint artifacts/generator/checkpoints/best_generator.pt \
  --style-mapping artifacts/prosody/style_mapping.json
uv run python -m emotionbridge.scripts.analyze_subjective_eval \
  --eval-dir artifacts/prosody/subjective_eval/pilot_v01

# 定量評価（Roundtrip: PESQ / MCD / F0 RMSE）
uv run python -m emotionbridge.scripts.evaluate_roundtrip \
  --baseline-manifest demo/v2/manifest.json \
  --candidate-manifest demo/v2-dsp/manifest.json \
  --output-dir artifacts/prosody/roundtrip_eval/v2_dsp
```

正式なテストスイートはない。Smoke config（`classifier_smoke.yaml`, `audio_gen_smoke.yaml`, `experiment_smoke.yaml`, `generator_smoke.yaml`）と CLI 手動実行でテストする。

## Tooling

Ruff: `line-length = 100`。`E501` は無視。`RUF001`/`RUF002`/`RUF003`（全角文字系）も無視（日本語テキスト処理のため）。`PLC0415`（lazy import）も許容。`ty check` は明示的な設定なしでデフォルト動作。

## Architecture

### Two-space Design

**感情空間**（6D softmax 確率）と**制御空間**（5D TTS パラメータ `[-1, +1]`）を分離。感情空間は固定で、TTS アダプタ（`VoicevoxAdapter`）のみを差し替えることで TTS エンジン非依存を実現する。

### Emotion Labels and Constants

**WRIME 8D** (legacy, `_WRIME_LABELS` in `data/wrime_classifier.py`): `[joy, sadness, anticipation, surprise, anger, fear, disgust, trust]` — データ読み込み専用の private 定数。

**JVNV 6D** (current, `JVNV_EMOTION_LABELS`): `[anger, disgust, fear, happy, sad, surprise]` — 全モジュールで使用。`_WRIME_TO_JVNV_INDICES = [4, 6, 5, 0, 1, 3]` で 8D→6D 変換。

制御空間 (`CONTROL_PARAM_NAMES`): `[pitch_shift, pitch_range, speed, energy, pause_weight]`

DSP制御空間 (`DSP_PARAM_NAMES`): `[jitter_amount, shimmer_amount, aperiodicity_shift, spectral_tilt_shift]`

感情円環モデル座標 (`COMMON6_CIRCUMPLEX_COORDS`): valence-arousal 2D 空間上の各感情座標。DSP マッピングで使用。

Go/No-Go 評価で重視: `KEY_EMOTION_LABELS = ["anger", "happy", "sad"]`

すべて `emotionbridge/constants.py` で定義。ラベル順序変更は全モジュールに影響するため厳禁。

### Configuration System

`configs/*.yaml` → `config.py` dataclasses。`load_config()` のキー判定ロジック:
- `voicevox` キー → `AudioGenConfig`
- `num_classes` / `label_conversion` / `class_weight_mode` キー → `ClassifierConfig`
- いずれもなし → `ValueError`

Phase 3 スクリプトは `emotionbridge/scripts/common.py::load_experiment_config` で独自に設定を読む（`ExperimentConfig` → `PathsConfig`, `V01Config`, `V02Config`, `V03Config` 等のネスト構造）。

### Phase 0: Classifier Data Flow

```
WRIME (shunk031/wrime on HF Hub)
  → data/wrime_classifier.py: batched preprocessing, filter (max_intensity > 1), 8D→6D変換, stratify split
  → training/classifier_trainer.py:
      EmotionTrainer (HF Trainer 拡張) + ClassifierBatchCollator
      CrossEntropy + inverse_frequency class weighting
      label_conversion: "argmax"(default) or "soft_label"(KL divergence + temperature scaling)
  → AutoModelForSequenceClassification (HF 標準モデル)
  → artifacts/classifier/checkpoints/best_model (HF 標準形式で保存)
  → inference/encoder.py: EmotionEncoder → encode() returns numpy (6,)
```

分類器はカスタムモデルクラスではなく `AutoModelForSequenceClassification` を直接使用。`model/` には `DeterministicMixer` と `ParameterGenerator` のみ残る。

Differential learning rates: `_split_model_parameters()` で BERT backbone (`bert_lr: 2e-5`) と classification head (`head_lr: 1e-3`) を分離。

### Phase 1: Audio Gen Data Flow

```
load_jvnv_texts() → JVNV transcription から感情分類付きテキスト選択
  → GridSampler.sample() → LHS (128 samples) or Grid in 5D [-1, +1]^5
  → VoicevoxClient.audio_query() → VoicevoxAdapter.apply(query, control_vec) → synthesis()
  → AudioValidator.validate() → TripletRecord
  → artifacts/audio_gen/dataset/triplet_dataset.parquet
```

Parquet 列規約: `emotion_*`, `ctrl_*`, `vv_*`。列名変更は下流に影響。音声パスは相対保存。Pipeline は `generation_progress.json` で checkpoint/resume 対応。Phase 1 は VOICEVOX Engine 起動が前提（default: `http://localhost:50021`）。

### Bridge Pipeline

`emotionbridge/inference/bridge_pipeline.py`:

```
EmotionEncoder(6D probs) → DeterministicMixer(5D params) → RuleBasedStyleSelector(style_id) → VOICEVOX → (optional) EmotionDSPProcessor → WAV
```

- **DeterministicMixer** (`model/generator.py`): `tanh(emotion_probs @ teacher_matrix)` — 6×5 教師行列の線形混合。学習パラメータなし
- **ParameterGenerator** (`model/generator.py`): Linear(6→hidden)→ReLU→Dropout→Linear(hidden→5)→Tanh — NN版の代替
- **RuleBasedStyleSelector**: `style_mapping.json` に基づき感情→VOICEVOX スタイルをマッピング
- チェックポイントの `model_type` フィールドで DeterministicMixer / ParameterGenerator を自動判別
- 信頼度が `fallback_threshold`（default: 0.3）未満 → デフォルトスタイル＋ゼロ ControlVector にフォールバック
- ファクトリ: `create_pipeline()` (async)

### DSP Post-processing Layer

`emotionbridge/dsp/`: WORLD 解析/再合成ベースの声質制御。Bridge パイプラインで `--enable-dsp` 時に適用。

- **EmotionDSPMapper** (`dsp/mapper.py`): 感情確率 6D → `DSPControlVector` 4D への変換。JVNV eGeMAPS 特徴量（jitter, shimmer, HNR, spectral flux 等）の感情別統計から制御量を算出
- **EmotionDSPProcessor** (`dsp/processor.py`): WORLD (pyworld) で F0/SP/AP を解析し、jitter・shimmer・aperiodicity・spectral tilt を操作して再合成。F0 抽出は `dio` または `harvest` を選択可能
- **DSPControlVector** (`dsp/types.py`): 4D制御ベクトル `[jitter_amount, shimmer_amount, aperiodicity_shift, spectral_tilt_shift]`
- 全パラメータがゼロに近い場合（`<= 1e-8`）は処理をスキップ
- 決定論的再現性のため seed ベースの RNG を使用

### TTS Layer

`emotionbridge/tts/`: `VoicevoxClient`（async HTTP）、`VoicevoxAdapter`、`ControlVector`、`AudioQuery` 等。`VoicevoxAdapter.apply()` は immutable — 新しい `AudioQuery` を返し、元を変更しない。抽象基底 `TTSAdapter` を実装しており、別 TTS エンジンへの差し替えポイント。

### Training Artifacts

```
artifacts/
├── classifier/checkpoints/best_model, reports/
├── audio_gen/audio/, dataset/triplet_dataset.parquet
├── generator/checkpoints/best_generator.pt
└── prosody/v01/ (JVNV), v02/ (VOICEVOX), v03/ (matching), style_mapping.json
```

### Accelerate Integration

HuggingFace Accelerate で mixed precision (`fp16`/`bf16`)、gradient accumulation、分散学習をサポート。`accelerator.prepare()` でラップ、`accelerator.gather()` でマルチプロセスメトリクス集約、checkpoint 保存は `accelerator.is_main_process` ガード。

## Important Invariants

- WRIME ラベル入力は `avg_readers` ネスト構造のみを受け付ける（flatキー互換は廃止）
- train/val/test 分割は常に stratified split。成立しない場合は fail-fast で停止する
- WRIME ラベルは相対強度を softmax 風に正規化して soft-label を生成可能。分類器の標準ラベルは argmax で生成
- 制御空間は常に 5D `ControlVector` / `[-1, 1]`。TTS 固有値変換は `VoicevoxAdapter` が担当
- DSP 制御空間は 4D `DSPControlVector`。`EmotionDSPMapper` が感情→DSPパラメータへ変換
- 各フェーズは `save_effective_config()` で実行時設定を成果物へ保存する前提。出力構造を変える変更は慎重に
- CLI 拡張時は `emotionbridge/cli.py` の parser 定義と command dispatch の両方を更新する
- 分類器チェックポイントは HF 標準形式（`config.json`, `model.safetensors` 等）。`AutoModelForSequenceClassification.from_pretrained()` でロード

## Language

日本語テキスト処理。BERT: `sbintuitions/modernbert-ja-70m`。コード中のコメント・ドキュメントは日本語。
