# Phase 3a-1 詳細設計: JVNV韻律プロファイル構築 + VOICEVOX韻律応答空間

> **対象要求**: EB3-A01（JVNV韻律プロファイル構築）、EB3-B01（VOICEVOX韻律応答空間構築）
> **作成日**: 2026-02-18
> **ステータス**: 初版

---

## 1. ギャップ分析

### 1.1 EB3-A01（JVNV韻律プロファイル構築）

| 仕様ID | 仕様要約 | 既存実装 | カバー状況 | ギャップ詳細 |
|--------|---------|---------|-----------|-------------|
| EB3-A01-01-001 | NV区間を無音化（切り詰めではない） | `prepare_jvnv.py` `_mask_intervals()`: 該当区間をゼロ埋め | **充足** | なし |
| EB3-A01-01-002 | NVラベル付与区間すべてを対象 | `prepare_jvnv.py` `_parse_nv_intervals()`: `nv`タグ含む行+数値2トークン行を解析 | **充足** | なし |
| EB3-A01-02-001 | eGeMAPSv02 88Dフルセット | `extract_egemaps.py`: `opensmile.FeatureSet.eGeMAPSv02`, `Functionals` | **充足** | 88特徴量が `egemaps__` プレフィクスで保存される |
| EB3-A01-02-002 | F0セミトーンスケール（追加変換なし） | openSMILEのデフォルト出力をそのまま使用 | **充足** | なし |
| EB3-A01-02-003 | openSMILE使用 | `extract_egemaps.py`: `import opensmile` | **充足** | なし |
| EB3-A01-03-001 | 話者4名ごとにmu/sigma算出、z-score | `normalize_features.py` `_normalize()`: `group_col="speaker"` でグループ化、`ddof=0` | **充足** | なし |
| EB3-A01-03-002 | 88D全次元に適用 | `normalize_features.py`: `_collect_feature_columns()` で `egemaps__` 全列を収集 | **充足** | 分散ゼロ列は自動除外されるが、実績上 `dropped_zero_variance_features: []` であり88D全次元が有効 |
| EB3-A01-04-001 | 感情別重心（算術平均） | `match_emotion_params.py` L151-153: `j_subset[feature_cols].mean(axis=0)` | **充足** | 重心計算は実装済み。ただしマッチングスクリプト内部で一時的に計算されるのみで、独立した永続化はない |
| EB3-A01-04-002 | 共分散行列または各次元標準偏差の保持 | **未実装** | **ギャップ** | 現状は重心のみを使用し、分布情報（共分散行列・標準偏差）は計算も保存もされていない |

### 1.2 EB3-B01（VOICEVOX韻律応答空間構築）

| 仕様ID | 仕様要約 | 既存実装 | カバー状況 | ギャップ詳細 |
|--------|---------|---------|-----------|-------------|
| EB3-B01-01-001 | JVNV同一のeGeMAPSv02 88Dで抽出 | `extract_egemaps.py` `_extract_voicevox()`: 同一の `_build_extractor()` を使用 | **充足** | なし |
| EB3-B01-01-002 | style_idごとにmu/sigma、z-score | `normalize_features.py`: `group_col = config.v02.speaker_mode` (="style_id") | **充足** | なし |
| EB3-B01-01-003 | ずんだもん8スタイル+四国めたん6スタイル | 設定値なし。全style_idが対象 | **部分充足** | 抽出は全style_idに対して行われるため、対象キャラクタ以外も含まれる。フィルタリングは未実装だが、マッチング時にdominant_emotionマッピングでフィルタされるため実害なし |
| EB3-B01-02-001 | 正規化後88D空間ユークリッド距離 | `match_emotion_params.py` L154: `np.linalg.norm(voice_features - centroid, axis=1)` | **充足** | なし |
| EB3-B01-02-002 | 韻律距離最小k件を推奨パラメータ候補（k値T.B.D） | `match_emotion_params.py` L139,155: `config.v03.nearest_k`（デフォルト25）で上位k件取得 | **充足（k値は暫定）** | 現状k=25。k値の最終決定はPhase 3bまで保留（USDM T.B.D） |
| EB3-B01-02-003 | 推奨パラメータはk件の**中央値** | **ギャップ** | **ギャップ** | 現状は `emotion_param_matches.json` で**平均値（mean）**のみ算出。中央値（median）の計算・保存が未実装 |

### 1.3 ギャップ要約

| # | ギャップ | 関連仕様 | 影響度 | 対応方針 |
|---|---------|---------|--------|---------|
| G1 | 感情別韻律プロファイルの分布情報が未永続化 | EB3-A01-04-002 | 中 | 重心・共分散行列・各次元標準偏差をJSONで保存する処理を追加 |
| G2 | 推奨パラメータの中央値が未計算 | EB3-B01-02-003 | 高 | マッチングスクリプトに中央値計算を追加し、教師値として保存 |
| G3 | 感情別重心の独立永続化がない | EB3-A01-04-001 | 低 | G1と同時にプロファイルJSONとして書き出し |

---

## 2. EB3-A01-04: 感情別韻律プロファイルの保存形式

### 2.1 設計方針

感情別韻律プロファイルは、マッチングスクリプトから独立した成果物として永続化する。推論時にマッチングを再実行せずとも、プロファイルを直接ロードできるようにする。

### 2.2 保存ファイルパス

```
artifacts/phase3/v01/jvnv_emotion_profiles.json
```

### 2.3 データ構造

```json
{
  "version": "1.0",
  "feature_set": "eGeMAPSv02",
  "feature_count": 88,
  "normalization": "speaker_zscore",
  "source_normalized_path": "artifacts/phase3/v01/jvnv_egemaps_normalized.parquet",
  "profiles": {
    "anger": {
      "num_samples": 249,
      "centroid": {
        "egemaps__F0semitoneFrom27.5Hz_sma3nz_amean": 0.5432,
        ...
      },
      "stddev": {
        "egemaps__F0semitoneFrom27.5Hz_sma3nz_amean": 1.0123,
        ...
      },
      "covariance_diag": [0.5432, ...],
      "covariance_full_path": "artifacts/phase3/v01/covariance/anger_cov.npy"
    },
    "disgust": { ... },
    "fear": { ... },
    "happy": { ... },
    "sad": { ... },
    "surprise": { ... }
  }
}
```

### 2.4 共分散行列の保存方式

**設計判断**: 共分散行列を対角成分（JSON内 `covariance_diag`）と完全行列（`.npy`ファイル）の2形式で保存する。

- **対角成分（JSON内）**: 各次元の分散値を88要素の配列として格納。軽量で可読性が高く、次元独立の距離計算（標準化ユークリッド距離等）に十分。
- **完全共分散行列（.npyファイル）**: 88x88のnumpy配列。マハラノビス距離への拡張時に使用。ファイルサイズは約62KB/感情（float64, 88x88x8bytes）、6感情で約370KB。

```
artifacts/phase3/v01/covariance/
  anger_cov.npy      # shape: (88, 88), dtype: float64
  disgust_cov.npy
  fear_cov.npy
  happy_cov.npy
  sad_cov.npy
  surprise_cov.npy
```

**理由**: JSON単体では88x88の行列は可読性が低く、ファイルサイズも肥大化する（約5MB/感情）。`.npy`は numpy の標準フォーマットでありロード高速・サイズ効率が良い。対角成分のみをJSONに含めることで、プロファイルの概要把握とフル行列利用の両方に対応する。

### 2.5 生成タイミング

`match_emotion_params.py` の `run_matching()` 内でJVNV重心を計算する既存処理を拡張し、同時にプロファイルを生成・保存する。新規スクリプトは不要。

---

## 3. EB3-B01-02: 推奨パラメータ導出とk値選定方針

### 3.1 中央値計算の追加

現状の `match_emotion_params.py` は `control_summary` として各制御パラメータのmean/std/min/maxを保存している。USDM EB3-B01-02-003の仕様に従い、**中央値（median）**を追加する。

**変更箇所**: `match_emotion_params.py` L161-169

```python
# 追加: median を算出し control_summary に含める
controls_summary[control] = {
    "mean": float(np.mean(values)),
    "median": float(np.median(values)),  # 追加
    "std": float(np.std(values, ddof=0)),
    "min": float(np.min(values)),
    "max": float(np.max(values)),
}
```

### 3.2 教師値の保存形式

パラメータ生成器（EB3-D01）の教師データとして直接利用可能な形式で、感情別推奨パラメータを独立保存する。

**保存ファイルパス**:
```
artifacts/phase3/v03/recommended_params.json
```

**データ構造**:
```json
{
  "version": "1.0",
  "nearest_k": 25,
  "aggregation": "median",
  "emotions": {
    "anger": {
      "ctrl_pitch_shift": -0.0806,
      "ctrl_pitch_range": 0.5242,
      "ctrl_speed": 0.0145,
      "ctrl_energy": 0.5399,
      "ctrl_pause_weight": 0.0492
    },
    ...
  }
}
```

### 3.3 k値選定方針

**現状**: k=25（`V03Config.nearest_k` のデフォルト値）

**USDM方針**: k値はPhase 3bのハイパーパラメータ調整で最終決定（EB3-B01-02-002 T.B.D）。

**Phase 3aでのスタンス**: k=25を暫定値として設計を進める。ただし、k値変更時に推奨パラメータを容易に再計算できるよう以下を担保する。

1. `--nearest-k` CLIオプションは既に実装済み（`match_emotion_params.py` L44）
2. `recommended_params.json` に使用したk値を明記する（上記の `nearest_k` フィールド）
3. Phase 3bでk値の感度分析を行うための実験基盤として、k={10, 15, 20, 25, 30, 50}程度のバッチ実行スクリプトの提供を推奨

**k値決定の考慮事項**（Phase 3b向けメモ）:
- kが小さすぎると外れ値の影響を受けやすく、kが大きすぎると感情間の差異が薄まる
- 中央値を使用するため平均値よりは外れ値に頑健だが、k=5以下では中央値の安定性も低下する
- VOICEVOXグリッドサーチ音声は25,416件。6感情で均等に割ると約4,236件/感情が候補となるため、k=25は全体の0.6%に相当し、十分に選択的
- 各感情のdistance_std（0.16〜0.27）がdistance_mean（3.35〜4.64）に対して小さいことから、k=25でも近傍の同質性は確保されている

---

## 4. 正規化統計量の永続化と推論時再利用

### 4.1 現状の永続化状態

正規化パラメータは既にJSONで保存されている。

| ファイル | 内容 | 保存されている情報 |
|---------|------|-------------------|
| `v01/jvnv_normalization_params.json` | JVNV話者内正規化パラメータ | group_col="speaker", 話者ごとのmeans/stds (88D), valid_features |
| `v02/voicevox_normalization_params.json` | VOICEVOX style_id内正規化パラメータ | group_col="style_id", style_idごとのmeans/stds (88D), valid_features |

### 4.2 推論時の役割分担

推論パイプライン（EB3-F01）では、VOICEVOX側の正規化パラメータのみが必要となる。

- **JVNV正規化パラメータ**: オフライン教師データ構築（Phase 3a）でのみ使用。推論時には不要（JVNV韻律プロファイルは正規化後の値として永続化済み）。
- **VOICEVOX正規化パラメータ**: 推論時にラウンドトリップ評価（EB3-G01-01-001）でVOICEVOX生成音声を正規化する際に必要。ただし、基本推論フロー（テキスト→パラメータ→音声）では不要。

### 4.3 推論時に必要なアーティファクト一覧

```
推論パイプライン (EB3-F01) が参照するファイル:
  1. Phase 0 感情分類器の重み
     → artifacts/phase0/checkpoints/best_model.pt（Phase 3a-2で再学習後の新重み）
  2. 感情別推奨パラメータ
     → artifacts/phase3/v03/recommended_params.json
  3. 感情→スタイルマッピング（EB3-E01）
     → artifacts/phase3/v03/emotion_style_mapping.json（Phase 3c で作成予定）

ラウンドトリップ評価 (EB3-G01-01) で追加参照するファイル:
  4. JVNV感情プロファイル
     → artifacts/phase3/v01/jvnv_emotion_profiles.json
  5. VOICEVOX正規化パラメータ
     → artifacts/phase3/v02/voicevox_normalization_params.json
```

### 4.4 正規化パラメータの不変性保証

正規化パラメータはオフラインで算出した統計量（mu, sigma）であり、推論時に更新してはならない。正規化パラメータJSONにはバージョン情報とソースデータのメタデータを含めることで、パラメータとデータの対応関係を追跡可能にする。

現状のJSONには `input_path`, `num_rows`, `source` が含まれており、追跡性は確保されている。追加のバージョニングは不要と判断する。

---

## 5. 推論時の新規VOICEVOX音声の正規化フロー

### 5.1 ユースケース

ラウンドトリップ評価（EB3-G01-01-001）において、推論パイプラインが生成したVOICEVOX音声の韻律特徴量を抽出し、JVNV感情プロファイルとの距離を計算する場合。

### 5.2 正規化フロー

```
新規VOICEVOX音声
  │
  ├─ (1) openSMILE eGeMAPSv02 で 88D 特徴量を抽出
  │
  ├─ (2) voicevox_normalization_params.json から
  │      該当 style_id の mu/sigma をロード
  │
  ├─ (3) z-score 正規化: z = (x - mu) / sigma
  │      ※ sigma = 0 の次元は v01/v02 構築時に除外済み（valid_features で保証）
  │
  └─ (4) 正規化済み 88D ベクトルと JVNV 感情プロファイル重心の
         ユークリッド距離を計算
```

### 5.3 実装上の注意点

1. **style_id の一致**: 推論時に使用する VOICEVOX style_id が `voicevox_normalization_params.json` の `group_params` に存在しない場合はエラーとする。オフラインでグリッドサーチを実施していない style_id に対する正規化パラメータは存在しないため、フォールバックは行わない。

2. **valid_features の適用**: 正規化パラメータJSON内の `valid_features` リストを使用して、正規化対象の次元を限定する。抽出された88次元のうち `valid_features` に含まれない次元は除外する（現状は全88次元が有効だが、将来的な変更への備え）。

3. **新規実装の配置**: 推論時の正規化処理は `emotionbridge/inference/` 配下に新規モジュール（例: `prosody_normalizer.py`）として実装する。既存の `normalize_features.py` はバッチ処理用スクリプトであり、推論時の単一音声正規化とは責務が異なるため、再利用ではなく共通ロジックの抽出を推奨する。

### 5.4 共通ロジックの抽出方針

`normalize_features.py` の `_normalize()` はDataFrame全体を対象とするバッチ処理。推論時は単一音声（1行）を処理するため、以下の共通処理を抽出する。

```python
# emotionbridge/prosody/normalizer.py（新規）

class ProsodyNormalizer:
    """正規化パラメータを保持し、単一ベクトルの正規化を行う"""

    @classmethod
    def from_json(cls, params_path: Path, group_key: str) -> ProsodyNormalizer:
        """保存済みパラメータJSONから特定グループの正規化器を構築"""
        ...

    def normalize(self, raw_features: dict[str, float]) -> np.ndarray:
        """88D生特徴量 → 正規化済み88Dベクトル"""
        ...
```

---

## 6. データフロー図

### 6.1 全体データフロー（既存→新規の境界）

```
┌─────────────────────────────────────────────────────────────────────┐
│ オフライン（Phase 3a 構築）                                         │
│                                                                     │
│ [JVNV音声]                          [Phase 1 グリッドサーチ音声]    │
│     │                                       │                       │
│     ▼                                       ▼                       │
│ prepare_jvnv.py ─── 既存             extract_egemaps.py ─── 既存   │
│ (NV区間除外)                         (eGeMAPS抽出: VOICEVOX)       │
│     │                                       │                       │
│     ▼                                       ▼                       │
│ extract_egemaps.py ─── 既存          normalize_features.py ─── 既存│
│ (eGeMAPS抽出: JVNV)                 (style_id内 z-score)           │
│     │                                       │                       │
│     ▼                                       ▼                       │
│ normalize_features.py ─── 既存              │                       │
│ (話者内 z-score)                            │                       │
│     │                                       │                       │
│     ├──────────────┐                        │                       │
│     ▼              ▼                        ▼                       │
│ ┌────────────┐ ┌──────────────┐    ┌──────────────────────┐       │
│ │感情プロ    │ │共分散行列    │    │match_emotion_params.py│       │
│ │ファイル    │ │(.npy)        │    │      ─── 既存+改修   │       │
│ │JSON ★新規 │ │  ★新規      │    │(+median, +profiles)  │       │
│ └────────────┘ └──────────────┘    └──────────┬───────────┘       │
│                                               │                     │
│                                    ┌──────────┴───────────┐        │
│                                    ▼                      ▼        │
│                           ┌──────────────┐  ┌──────────────────┐  │
│                           │推奨パラメータ│  │マッチング結果    │  │
│                           │JSON ★新規   │  │parquet ─── 既存  │  │
│                           └──────────────┘  └──────────────────┘  │
│                                                                     │
│ ★新規: 新たに追加する出力                                          │
│ 既存+改修: 既存スクリプトへの機能追加                               │
└─────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────┐
│ 推論時（Phase 3c/3d）                                               │
│                                                                     │
│ [テキスト入力]                                                      │
│     │                                                               │
│     ▼                                                               │
│ Phase 0 感情分類器 (6D確率)                                         │
│     │                                                               │
│     ├─────────────────────┐                                         │
│     ▼                     ▼                                         │
│ パラメータ生成器      スタイル選択                                    │
│ (6D→5D)              (6D→style_id)                                  │
│     │                     │                                         │
│     └─────────┬───────────┘                                         │
│               ▼                                                     │
│         VOICEVOX合成                                                │
│               │                                                     │
│               ▼                                                     │
│         [音声出力]                                                   │
│               │                                                     │
│               ▼ (評価時のみ)                                        │
│         eGeMAPS抽出 → 正規化 → JVNV距離計算                        │
│         (voicevox_normalization_params.json を使用)                  │
└─────────────────────────────────────────────────────────────────────┘
```

### 6.2 成果物の依存関係

```
v01/jvnv_manifest.parquet
  └─→ v01/jvnv_egemaps_raw.parquet
        └─→ v01/jvnv_egemaps_normalized.parquet
              └─→ v01/jvnv_normalization_params.json
              └─→ v01/jvnv_emotion_profiles.json        ★新規
              └─→ v01/covariance/*.npy                   ★新規

v02/voicevox_egemaps_raw.parquet
  └─→ v02/voicevox_egemaps_normalized.parquet
        └─→ v02/voicevox_normalization_params.json

v03/emotion_param_matches.parquet                         既存
v03/emotion_param_matches.json                            既存（改修: +median）
v03/recommended_params.json                               ★新規
```

---

## 7. 実装変更のまとめ

### 7.1 `match_emotion_params.py` の改修内容

| 変更箇所 | 内容 | 理由 |
|---------|------|------|
| `controls_summary` へ `median` 追加 | `np.median(values)` を計算し辞書に含める | EB3-B01-02-003 |
| 感情プロファイル生成処理の追加 | 重心・標準偏差・共分散行列を計算し永続化 | EB3-A01-04-001, EB3-A01-04-002 |
| 推奨パラメータJSON出力の追加 | 感情別のmedian制御パラメータを独立JSONとして保存 | EB3-D01-005（教師データ準備） |

### 7.2 新規ファイル

| ファイル | 目的 |
|---------|------|
| `emotionbridge/prosody/normalizer.py` | 推論時の単一音声正規化ユーティリティ |

### 7.3 既存ファイルの変更なし

| ファイル | 理由 |
|---------|------|
| `prepare_jvnv.py` | EB3-A01-01 を完全に充足 |
| `extract_egemaps.py` | EB3-A01-02, EB3-B01-01-001 を完全に充足 |
| `normalize_features.py` | EB3-A01-03, EB3-B01-01-002 を完全に充足 |
| `common.py` | 変更不要 |
| `experiment_config.yaml` | 設定追加の必要なし（k値等は既存設定で対応可） |

---

## 8. すぐに決められない事項

| # | 事項 | 理由 | 影響 | 決定期限 |
|---|------|------|------|---------|
| O1 | k値の最終決定（現状k=25） | パラメータ生成器の精度に直結するハイパーパラメータであり、Phase 3bの実験結果（生成器のMAE、ラウンドトリップ距離）を見て判断すべき | k値変更時は `match_emotion_params.py` を再実行し `recommended_params.json` を再生成する必要がある。設計上、再計算コストは低い | Phase 3b完了時 |
| O2 | マハラノビス距離への切り替え判断 | EB3-A01-04-002で共分散行列を保持するのは将来拡張のため。現状のユークリッド距離でV-03の結果が文献整合的であるため、切り替えの動機が弱い。一方、感情間で分散構造が異なる場合（例: angerはF0変動大、sadはエネルギー低）にマハラノビスが有利となる可能性がある | 距離関数の変更は `match_emotion_params.py` 内の1行の変更で対応可能（`np.linalg.norm` → マハラノビス計算）。共分散行列の正則性（88Dで249〜306サンプル）は十分だが、条件数の確認が必要 | Phase 3b実験時 |
| O3 | 話者均等化（equalize_speaker_stats）の最適性 | JVNV正規化で `equalize_speaker_stats=True` により感情ラベルの均衡サブセットから統計量を算出している。これが重心品質に与える影響は未検証 | 影響は限定的（各話者の感情別サンプル数は54〜81でほぼ均衡）。現状の設定を維持し、Phase 3bで必要に応じて検証 | Phase 3b |
| O4 | VOICEVOX未対応style_idでの推論時の振る舞い | グリッドサーチを実施していないstyle_idに対する正規化パラメータが存在しない場合のフォールバック方針 | ラウンドトリップ評価でのみ問題となる。基本推論フローでは正規化不要。エラー終了で十分だが、将来的にノーマルstyle_idへのフォールバックも検討可能 | Phase 3c |
| O5 | 感情プロファイル構築の話者間統合方法 | 現状は4話者の正規化済みデータをプールして重心を計算。話者による重み付け（例: 話者あたりのサンプル数補正）は行っていない | 話者間サンプル数差は小さい（356〜490件/話者）。均等化はEB3-A01-04-001の「算術平均」の仕様に合致するため現状維持で問題なし | 決定済み（現状維持） |

---

## 9. 補足: 既存アーティファクトの確認結果

### 9.1 JVNV正規化パラメータ

- **話者数**: 4（F1, F2, M1, M2と推定）
- **有効特徴量**: 88/88（分散ゼロ次元なし）
- **正規化単位**: speaker
- **均等化**: 有効（stats_rows < num_rowsとなっている話者あり）
- **全発話数**: 1,615

### 9.2 VOICEVOX正規化パラメータ

- **正規化単位**: style_id
- **有効特徴量**: 88/88
- **全サンプル数**: 25,416

### 9.3 マッチング結果

- **共通特徴量**: 88次元
- **k値**: 25
- **感情別距離**: anger(4.64) > fear(3.85) > sad(3.78) > happy(3.76) > surprise(3.65) > disgust(3.35)
- **注目点**: angerが最も距離が大きい = VOICEVOXで再現困難な感情の可能性。disgustが最も近い = 低エネルギー・低ピッチの韻律がVOICEVOXで再現しやすい
