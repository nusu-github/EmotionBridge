# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

EmotionBridge is a learning-based conversion engine that bridges text emotion analysis and emotional speech synthesis. Inspired by CLIP's architecture, it aligns "text emotion space" and "audio emotion space" in a shared latent space. The project is developed in phases:

- **Phase 0**: Japanese text emotion encoder — BERT + regression head trained on WRIME dataset, outputs 8D emotion vectors (Plutchik's basic emotions)
- **Phase 1**: VOICEVOX TTS integration and audio sample generation pipeline
- **Phase 2** (current): SER embedding validation & audio emotion encoder — emotion2vec+ で音声感情空間を構築し、Phase 0 のテキスト感情空間とアライメントする

## Commands

```bash
# Install dependencies (uses uv package manager)
uv sync

# Train Phase 0 model
uv run python main.py train --config configs/phase0.yaml

# Train with Accelerate (recommended for GPU/distributed)
uv run accelerate launch main.py train --config configs/phase0.yaml

# Data analysis and visualization
uv run python main.py analyze-data --config configs/phase0.yaml

# Inference
uv run python main.py encode \
  --checkpoint artifacts/phase0/checkpoints/best_model.pt \
  --text "今日は楽しかった！"

# Lint
uv run ruff check .

# Format
uv run ruff format .

# Type check
uv run ty check
```

## Architecture

### Two-space Design (Core Concept)

The system separates **shared emotion latent space** (semantic alignment, d-dimensional) from **control space** (5D TTS parameter knobs: pitch_shift, pitch_range, speed, energy, pause_weight). This separation is what enables TTS-agnostic operation — the semantic space stays fixed while only TTS adapters are swapped.

### Phase 0 Data Flow

```
WRIME dataset (shunk031/wrime on HF Hub)
  → emotionbridge/data/wrime.py: load, filter (max_intensity > 1), normalize (/3.0), stratify, split
  → emotionbridge/training/trainer.py: TextBatchCollator tokenizes on-the-fly, trains with weighted MSE
  → emotionbridge/model/regressor.py: BERT [CLS] → Dropout → Linear(768→256) → ReLU → Dropout → Linear(256→8) → Sigmoid
  → artifacts/phase0/checkpoints/best_model.pt
  → emotionbridge/inference/encoder.py: EmotionEncoder loads checkpoint, returns 8D numpy arrays
```

### Key Design Decisions

- **Differential learning rates**: BERT backbone (2e-5) vs regression head (1e-3), configured via `TrainConfig.bert_lr` / `head_lr`
- **Emotion weighting**: `inverse_mean` mode auto-weights loss by inverse of per-emotion mean intensity to compensate for imbalanced emotions (anger/trust are rare)
- **Go/No-Go gates**: Evaluation has tiered thresholds — stricter for the 6 major emotions, relaxed for anger/trust (`EvalConfig`)
- **Output range**: Sigmoid activation produces [0, 1] values; raw WRIME labels (0–3) are normalized by `LABEL_SCALE_MAX = 3.0`

### Emotion Label Order (fixed)

`[joy, sadness, anticipation, surprise, anger, fear, disgust, trust]` — this order is hardcoded in `constants.py` and used throughout all modules. `anger` and `trust` are classified as `LOW_VARIANCE_EMOTION_LABELS` with relaxed evaluation thresholds.

### Configuration System

`configs/phase0.yaml` → `config.py` dataclasses (`Phase0Config` containing `DataConfig`, `ModelConfig`, `TrainConfig`, `EvalConfig`). Loaded via `load_config()`. Effective config is snapshot-saved to `artifacts/` at training start.

### Training Artifacts

All outputs go to `artifacts/phase0/` (configurable via `train.output_dir`):
- `checkpoints/best_model.pt` — model state_dict + config metadata + tokenizer name
- `reports/` — data_report.json, training_history.json, evaluation.json (includes Go/No-Go result and error analysis)
- `tensorboard/` — TensorBoard event files (`tensorboard --logdir artifacts/phase0/tensorboard`)

### Accelerate Integration

Training uses HuggingFace Accelerate for mixed precision (`fp16`/`bf16`), gradient accumulation, and distributed training. The trainer wraps model/optimizer/dataloaders with `accelerator.prepare()` and uses `accelerator.gather()` for multi-process metric computation. Checkpoint saving is guarded by `accelerator.is_main_process`.

### Phase 2: SER Model Decision

- **採用モデル**: `emotion2vec/emotion2vec_plus_large` (FunASR ベース, ~300M params)
- **使用レイヤー**: **feats (1024D)** — logits (9D) ではなく中間特徴量を使用
- **理由**: ランキング学習で必要な「微妙なパラメータ差による感情ニュアンスの変化」を捉えるには、logits の 9 次元では情報量が不足する。feats (1024D) なら射影層が必要な情報を選択的に抽出できる
- **却下モデル**: kushinada-hubert-large — JTES 4感情分類に過適合しており、6感情の分離が不十分だった (シルエットスコア 0.009〜0.049)
- **検証コーパス**: JVNV v1 (4話者 × 6感情 × 2セッション, 1615サンプル, 48kHz)
- **検証結果** (emotion2vec+): feats シルエットスコア 0.056, CH Index 219.8

## Language

The project targets Japanese text processing. BERT model: `tohoku-nlp/bert-base-japanese-whole-word-masking`. Tokenizer requires `fugashi` + `ipadic` (Japanese morphological analysis). Code comments and documentation are in Japanese.
