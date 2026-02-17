# Copilot Instructions for EmotionBridge

## まず把握する全体像
- このリポジトリは Phase 0→1→2 の逐次パイプライン。入口は `main.py` → `emotionbridge/cli.py`。
- Phase 0: WRIME テキストから 8D 感情ベクトルを学習（`emotionbridge/data/wrime.py`, `emotionbridge/training/trainer.py`, `emotionbridge/model/regressor.py`）。
- Phase 1: Phase 0 の checkpoint でテキストを encode し、VOICEVOX で音声を生成して triplet dataset を作成（`emotionbridge/generation/pipeline.py`）。
- Phase 2: Phase 1 dataset を読み、emotion2vec+ で埋め込み/logitsを抽出し `ser_score` を付与（`emotionbridge/alignment/triplet_builder.py`）。

## 重要な不変条件（壊しやすい）
- 感情ラベル順は固定: `[joy, sadness, anticipation, surprise, anger, fear, disgust, trust]`（`emotionbridge/constants.py`）。
- `anger`, `trust` は低分散扱いで評価閾値が別（`LOW_VARIANCE_EMOTION_LABELS`, `emotionbridge/training/metrics.py`, `emotionbridge/training/trainer.py`）。
- WRIME ラベルは `LABEL_SCALE_MAX=3.0` で [0,1] に正規化。モデル出力は Sigmoid 前提（`emotionbridge/data/wrime.py`, `emotionbridge/model/regressor.py`）。
- 制御空間は常に 5D `ControlVector` / [-1,1]。TTS 固有値への変換は `VoicevoxAdapter` が担当（`emotionbridge/tts/types.py`, `emotionbridge/tts/adapter.py`）。

## 設定・CLIの作法
- 設定ロードはトップレベルキーで型判定（`emotionbridge/config.py::load_config`）:
  - `voicevox` → `Phase1Config`
  - `jvnv_root` → `Phase2ValidationConfig`
  - `phase1_dataset_path` → `Phase2TripletConfig`
- CLI 拡張時は `emotionbridge/cli.py` の parser 定義と command dispatch の両方を更新する。
- 各フェーズは `save_effective_config()` で実行時設定を成果物へ保存する前提。出力構造を変える変更は慎重に行う。

## 開発/実行フロー（実際に使うコマンド）
- 依存導入: `uv sync`
- Phase 0 学習: `uv run python main.py train --config configs/phase0.yaml`
- 分散/混合精度学習: `uv run accelerate launch main.py train --config configs/phase0.yaml`
- Phase 1 生成（VOICEVOX 起動が前提）: `uv run python main.py generate-samples --config configs/phase1.yaml`
- Smoke 実行: `configs/phase1_smoke.yaml`, `configs/phase2_triplet_smoke.yaml`
- Phase 2 検証: `uv run python main.py validate-ser --config configs/phase2_validation.yaml`
- Phase 2 スコア付与: `uv run python main.py score-triplets --config configs/phase2_triplet.yaml`
- 静的チェック: `uv run ruff check .`, `uv run ruff format .`, `uv run ty check`

## データ連携ポイント
- Phase 1 Parquet の列規約は `emotion_*`, `ctrl_*`, `vv_*`（`emotionbridge/generation/dataset.py`）。列名変更は下流処理に影響する。
- `ser_score` は Phase 1 では `None`、Phase 2 で更新される（`TripletRecord`）。
- 音声パスは相対保存が基本。Phase 2 は `phase1_output_dir` と dataset 位置の両方から解決を試みる（`Phase2TripletScorer._resolve_audio_path`）。

## 外部依存の前提
- 日本語 tokenizer 依存: `fugashi`, `ipadic`, `unidic-lite`（`pyproject.toml`）。
- VOICEVOX 連携は非同期 REST クライアント実装（`emotionbridge/tts/voicevox_client.py`）。
- emotion2vec+ 抽出は FunASR `AutoModel` を使用し、`feats(1024D)` と `logits(9D)` を返す（`emotionbridge/analysis/emotion2vec.py`）。