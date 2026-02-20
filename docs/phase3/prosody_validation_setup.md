# Phase 3: 韻律特徴空間検証セットアップ

本ドキュメントは、実験計画書の V-01 / V-02 / V-03 をこのリポジトリ上で実行するための準備手順をまとめたものです。

## 1. 前提

- Python 環境をセットアップ済み（`uv sync`）
- JVNV コーパスを `data/jvnv_v1` 配下に配置済み
- Phase 1 の生成データ（`artifacts/audio_gen/dataset/triplet_dataset.parquet` と音声）を用意済み

`openSMILE` の Python バインディングが未導入の場合は、次を実行してください。

```bash
uv add opensmile
```

## 2. 設定ファイル

実験設定は [configs/experiment_config.yaml](../../configs/experiment_config.yaml) を編集してください。

主な設定項目:

- `paths.jvnv_root`: JVNV ルート
- `paths.triplet_dataset_path`: Phase 1 parquet
- `v01.output_dir`, `v02.output_dir`, `v03.output_dir`: 出力先
- `evaluation.silhouette_go_threshold`: V-01 ゲート閾値

## 3. 実行順序

### V-01: JVNV 感情分離性検証

```bash
uv run python -m emotionbridge.scripts.prepare_jvnv --config configs/experiment_config.yaml
uv run python -m emotionbridge.scripts.extract_egemaps --config configs/experiment_config.yaml --source jvnv --jvnv-audio-key processed
uv run python -m emotionbridge.scripts.normalize_features --config configs/experiment_config.yaml --source jvnv
uv run python -m emotionbridge.scripts.evaluate_separation --config configs/experiment_config.yaml
```

連続軸ベース判定（Conditional Go 判定）を追加する場合:

```bash
uv run python -m emotionbridge.scripts.evaluate_continuous_axes --config configs/experiment_config.yaml
```

NV 除外影響比較を行う場合（任意）:

```bash
uv run python -m emotionbridge.scripts.extract_egemaps --config configs/experiment_config.yaml --source jvnv --jvnv-audio-key raw
uv run python -m emotionbridge.scripts.normalize_features --config configs/experiment_config.yaml --source jvnv --input-path artifacts/prosody/v01/jvnv_egemaps_with_nv_raw.parquet --output-path artifacts/prosody/v01/jvnv_egemaps_with_nv_normalized.parquet --params-path artifacts/prosody/v01/jvnv_with_nv_normalization_params.json
uv run python -m emotionbridge.scripts.evaluate_separation --config configs/experiment_config.yaml --with-nv-input-path artifacts/prosody/v01/jvnv_egemaps_with_nv_normalized.parquet
```

### V-02: VOICEVOX 応答性検証

```bash
uv run python -m emotionbridge.scripts.extract_egemaps --config configs/experiment_config.yaml --source voicevox
uv run python -m emotionbridge.scripts.normalize_features --config configs/experiment_config.yaml --source voicevox
uv run python -m emotionbridge.scripts.evaluate_responsiveness --config configs/experiment_config.yaml --gate-policy feature_only
```

`feature_and_av` を使うと A/V 方向整合も必須にできます（診断ではなく厳格判定したい場合）。

### V-03: ドメインギャップ検証

```bash
uv run python -m emotionbridge.scripts.evaluate_domain_gap --config configs/experiment_config.yaml
uv run python -m emotionbridge.scripts.match_emotion_params --config configs/experiment_config.yaml
uv run python -m emotionbridge.scripts.prepare_direct_matching --config configs/experiment_config.yaml
```

## 4. 出力物

- V-01: `artifacts/prosody/v01/`
  - `jvnv_manifest.parquet`
  - `jvnv_egemaps_raw.parquet`
  - `jvnv_egemaps_normalized.parquet`
  - `v01_metrics.json`, `v01_separation_report.md`
- V-02: `artifacts/prosody/v02/`
  - `voicevox_egemaps_raw.parquet`
  - `voicevox_egemaps_normalized.parquet`
  - `v02_responsiveness_metrics.json`, `v02_responsiveness_report.md`
- V-03: `artifacts/prosody/v03/`
  - `v03_domain_gap_metrics.json`, `v03_domain_gap_report.md`
  - `emotion_param_matches.parquet`, `emotion_param_matches.json`
  - `direct_matching_profiles.parquet`, `direct_matching_profiles.json`, `direct_matching_profiles_report.md`

## 5. 非公開データ方針

抽出済み特徴量・正規化パラメータ・スコアテーブル・可視化画像は配布対象外とし、コードと手順のみ公開する運用を想定しています。
