# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

EmotionBridge is a learning-based conversion engine that bridges text emotion analysis and emotional speech synthesis. Inspired by CLIP's architecture, it aligns "text emotion space" and "audio emotion space" in a shared latent space. The project is developed in phases:

- **Phase 0**: Japanese text emotion encoder — BERT + regression head trained on WRIME dataset, outputs 8D emotion vectors (Plutchik's basic emotions)
- **Phase 1**: VOICEVOX TTS integration and audio sample generation pipeline — generates (text, control_params, audio) triplets
- **Phase 3** (current): Prosody feature workflow（JVNV/VOICEVOXのeGeMAPS抽出・正規化・マッチング）

## Commands

```bash
# Install dependencies (uses uv package manager)
uv sync

# Lint / Format / Type check
uv run ruff check .
uv run ruff format .
uv run ty check

# --- Phase 0: Text Emotion Encoder ---
uv run python main.py train --config configs/phase0.yaml
uv run accelerate launch main.py train --config configs/phase0.yaml  # recommended
uv run python main.py analyze-data --config configs/phase0.yaml
uv run python main.py encode \
  --checkpoint artifacts/phase0/checkpoints/best_model.pt \
  --text "今日は楽しかった！"

# --- Phase 1: Audio Generation ---
uv run python main.py generate-samples --config configs/phase1.yaml
uv run python main.py generate-samples --config configs/phase1_smoke.yaml  # quick test (16 samples)
uv run python main.py list-speakers                                        # VOICEVOX speakers
uv run python main.py synthesize --text "嬉しい" --output out.wav          # single synthesis

# --- Phase 3: Prosody feature workflow ---
uv run python -m emotionbridge.scripts.prepare_jvnv --config configs/experiment_config.yaml
uv run python -m emotionbridge.scripts.extract_egemaps --config configs/experiment_config.yaml --source jvnv
uv run python -m emotionbridge.scripts.extract_egemaps --config configs/experiment_config.yaml --source voicevox
uv run python -m emotionbridge.scripts.normalize_features --config configs/experiment_config.yaml --source jvnv
uv run python -m emotionbridge.scripts.normalize_features --config configs/experiment_config.yaml --source voicevox
uv run python -m emotionbridge.scripts.evaluate_responsiveness --config configs/experiment_config.yaml  # feature_weights.json出力
uv run python -m emotionbridge.scripts.align_domains --config configs/experiment_config.yaml            # クロスドメイン整合
uv run python -m emotionbridge.scripts.match_emotion_params --config configs/experiment_config.yaml     # aligned + weighted distance
```

No formal test suite exists. Testing is done via smoke configs (`phase1_smoke.yaml`, `experiment_smoke.yaml`) and manual CLI invocation.

## Architecture

### Two-space Design (Core Concept)

The system separates **shared emotion latent space** (semantic alignment, d-dimensional) from **control space** (5D TTS parameter knobs: pitch_shift, pitch_range, speed, energy, pause_weight). This separation enables TTS-agnostic operation — the semantic space stays fixed while only TTS adapters are swapped.

### Emotion Label Order (fixed)

`[joy, sadness, anticipation, surprise, anger, fear, disgust, trust]` — hardcoded in `constants.py`, used throughout all modules. `anger` and `trust` are `LOW_VARIANCE_EMOTION_LABELS` with relaxed evaluation thresholds. Control space: `[pitch_shift, pitch_range, speed, energy, pause_weight]` (also in `constants.py`).

### Configuration System

`configs/*.yaml` → `config.py` dataclasses. `load_config()` auto-detects config type by key presence:
- `voicevox` → `Phase1Config`
- Otherwise → `Phase0Config`

Phase 0: `Phase0Config` = `DataConfig` + `ModelConfig` + `TrainConfig` + `EvalConfig`
Phase 1: `Phase1Config` = `VoicevoxConfig` + `ControlSpaceConfig` + `GridConfig` + `TextSelectionConfig` + `ValidationConfig` + `GenerationConfig`
Phase 3 scripts use `configs/experiment_config.yaml` with `emotionbridge/scripts/common.py::load_experiment_config`

### Phase 0 Data Flow

```
WRIME dataset (shunk031/wrime on HF Hub)
  → data/wrime.py: load, filter (max_intensity > 1), normalize (/3.0), stratify, split
  → training/trainer.py: TextBatchCollator tokenizes on-the-fly, trains with weighted MSE
  → model/regressor.py: BERT [CLS] → Dropout → Linear(768→256) → ReLU → Dropout → Linear(256→8) → Sigmoid
  → artifacts/phase0/checkpoints/best_model.pt
  → inference/encoder.py: EmotionEncoder loads checkpoint, returns 8D numpy arrays
```

Key design decisions:
- **Differential learning rates**: BERT backbone (2e-5) vs regression head (1e-3), via `TrainConfig.bert_lr` / `head_lr`
- **Emotion weighting**: `inverse_mean` mode auto-weights loss by inverse of per-emotion mean intensity
- **Go/No-Go gates**: Tiered thresholds — stricter for 6 major emotions, relaxed for anger/trust (`EvalConfig`)
- **Output range**: Sigmoid → [0, 1]; raw WRIME labels (0–3) normalized by `LABEL_SCALE_MAX = 3.0`

### Phase 1 Data Flow

```
TextSelector.select() — stratified sampling from WRIME splits (texts_per_emotion per group)
  → GridSampler.sample(text_id) — LHS (128 samples) or Full Grid in 5D control space [-1, +1]^5
  → VoicevoxClient.audio_query() → VoicevoxAdapter.apply(query, control_vec) → synthesis()
     Adapter maps [-1, +1] → VOICEVOX params: speedScale, pitchScale, intonationScale, volumeScale, pause*
  → AudioValidator.validate() — file size, duration, RMS amplitude, sample rate
  → TripletRecord → triplet_dataset.parquet (text_id, text, emotion_*8, ctrl_*5, audio_path, vv_*7)
```

Pipeline (`generation/pipeline.py`) supports checkpoint/resume via `generation_progress.json`. Async batch synthesis with `asyncio.Semaphore` concurrency control. Phase 1 requires a running VOICEVOX Engine (default: `http://localhost:50021`).

### Training Artifacts

```
artifacts/
├── phase0/
│   ├── checkpoints/best_model.pt
│   ├── reports/{data_report,training_history,evaluation}.json
│   └── tensorboard/
├── phase1/
│   ├── audio/{text_id:04d}/{task_id}.wav
│   ├── dataset/triplet_dataset.parquet, metadata.json
│   ├── checkpoints/generation_progress.json
│   └── reports/generation_report.json
└── phase3/
  ├── v01/  # JVNV preprocessing/extraction/normalization
  ├── v02/  # VOICEVOX extraction/normalization
  └── v03/  # matching results and reports
```

### Accelerate Integration

Training uses HuggingFace Accelerate for mixed precision (`fp16`/`bf16`), gradient accumulation, and distributed training. The trainer wraps model/optimizer/dataloaders with `accelerator.prepare()` and uses `accelerator.gather()` for multi-process metric computation. Checkpoint saving is guarded by `accelerator.is_main_process`.

## Language

The project targets Japanese text processing. BERT model: `tohoku-nlp/bert-base-japanese-whole-word-masking`. Tokenizer requires `fugashi` + `ipadic` (Japanese morphological analysis). Code comments and documentation are in Japanese.
