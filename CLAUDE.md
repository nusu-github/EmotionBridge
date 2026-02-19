# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

EmotionBridge は日本語テキストから感情を分類し、韻律特徴マッチングに基づく制御パラメータ生成と VOICEVOX TTS を組み合わせて感情音声を合成する変換エンジン。

```
テキスト → 感情分類（6クラス） → DeterministicMixer（6D→5D） → スタイル選択 → VOICEVOX → 感情音声
```

フェーズ構成:

- **Phase 0 (Classifier)**: BERT + 分類ヘッド、WRIME → 6感情クラス（anger, disgust, fear, happy, sad, surprise）の softmax 確率を出力
- **Phase 1 (Audio Gen)**: VOICEVOX 音声サンプル生成パイプライン — (text, control_params, audio) triplet を作成
- **Phase 3 (Prosody)**: JVNV/VOICEVOX の eGeMAPS 抽出・クロスドメイン整合・マッチング → 教師表 → DeterministicMixer
- **Bridge**: 統合推論パイプライン — text → classifier → DeterministicMixer → style selection → VOICEVOX → WAV

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

# --- Bridge: 統合推論（要 VOICEVOX Engine） ---
uv run python main.py bridge \
  --text "今日は楽しかった！" \
  --output output.wav \
  --character zundamon
```

正式なテストスイートはない。Smoke config（`classifier_smoke.yaml`, `audio_gen_smoke.yaml`, `experiment_smoke.yaml`）と CLI 手動実行でテストする。

## Architecture

### Two-space Design

**感情空間**（6D softmax 確率）と**制御空間**（5D TTS パラメータ `[-1, +1]`）を分離。感情空間は固定で、TTS アダプタ（`VoicevoxAdapter`）のみを差し替えることで TTS エンジン非依存を実現する。

### Emotion Labels (2 sets)

**WRIME 8D** (legacy, `EMOTION_LABELS`): `[joy, sadness, anticipation, surprise, anger, fear, disgust, trust]`

**JVNV 6D** (current, `JVNV_EMOTION_LABELS`): `[anger, disgust, fear, happy, sad, surprise]` — 全モジュールで使用。`WRIME_TO_JVNV_INDICES = [4, 6, 5, 0, 1, 3]` で 8D→6D 変換。

制御空間 (`CONTROL_PARAM_NAMES`): `[pitch_shift, pitch_range, speed, energy, pause_weight]`

Go/No-Go 評価で重視: `KEY_EMOTION_LABELS = ["anger", "happy", "sad"]`

すべて `emotionbridge/constants.py` で定義。ラベル順序変更は全モジュールに影響するため厳禁。

### Configuration System

`configs/*.yaml` → `config.py` dataclasses。`load_config()` のキー判定ロジック:
- `voicevox` キー → `AudioGenConfig`
- `num_classes` / `label_conversion` / `class_weight_mode` キー → `ClassifierConfig`
- いずれもなし → `ValueError`

Phase 3 スクリプトは `emotionbridge/scripts/common.py::load_experiment_config` で独自に設定を読む。

### Phase 0: Classifier Data Flow

```
WRIME (shunk031/wrime on HF Hub)
  → data/wrime.py: filter (max_intensity > 1), normalize (/3.0), 8D→6D変換, stratify split
  → training/classifier_trainer.py: CrossEntropy + inverse_frequency class weighting
  → model/classifier.py: BERT [CLS] → Dropout → Linear(768→256) → ReLU → Dropout → Linear(256→6)
  → forward() → logits, predict_proba() → softmax (6D確率)
  → artifacts/classifier/checkpoints/best_model
  → inference/encoder.py: EmotionEncoder → encode() returns numpy (6,)
```

Differential learning rates: BERT backbone (`bert_lr: 2e-5`) vs classification head (`head_lr: 1e-3`)。

### Phase 1: Audio Gen Data Flow

```
TextSelector.select() → stratified sampling from WRIME
  → GridSampler.sample() → LHS (128 samples) or Grid in 5D [-1, +1]^5
  → VoicevoxClient.audio_query() → VoicevoxAdapter.apply(query, control_vec) → synthesis()
  → AudioValidator.validate() → TripletRecord
  → artifacts/audio_gen/dataset/triplet_dataset.parquet
```

Parquet 列規約: `emotion_*`, `ctrl_*`, `vv_*`。列名変更は下流に影響。音声パスは相対保存。Pipeline は `generation_progress.json` で checkpoint/resume 対応。Phase 1 は VOICEVOX Engine 起動が前提（default: `http://localhost:50021`）。

### Bridge Pipeline

`emotionbridge/inference/bridge_pipeline.py`:

```
EmotionEncoder(6D probs) → DeterministicMixer(5D params) → RuleBasedStyleSelector(style_id) → VOICEVOX → WAV
```

- **DeterministicMixer** (`model/generator.py`): `tanh(emotion_probs @ teacher_matrix)` — 6×5 教師行列の線形混合。学習パラメータなし
- **ParameterGenerator** (`model/generator.py`): Linear(6→hidden)→ReLU→Dropout→Linear(hidden→5)→Tanh — NN版の代替
- **RuleBasedStyleSelector**: `style_mapping.json` に基づき感情→VOICEVOX スタイルをマッピング
- 信頼度が `fallback_threshold`（default: 0.3）未満の場合、デフォルトスタイル＋ニュートラル制御にフォールバック
- ファクトリ: `create_pipeline()` (async)

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

- WRIME ラベルは `LABEL_SCALE_MAX=3.0` で [0,1] に正規化。分類器は argmax でクラスラベルに変換
- 制御空間は常に 5D `ControlVector` / `[-1, 1]`。TTS 固有値変換は `VoicevoxAdapter` が担当
- 各フェーズは `save_effective_config()` で実行時設定を成果物へ保存する前提。出力構造を変える変更は慎重に
- CLI 拡張時は `emotionbridge/cli.py` の parser 定義と command dispatch の両方を更新する

## Language

日本語テキスト処理。BERT: `sbintuitions/modernbert-ja-70m`。コード中のコメント・ドキュメントは日本語。
