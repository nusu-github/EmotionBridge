# Phase 1 データ生成パイプライン 詳細設計

## 1. 概要

Phase 1 はテキスト感情ベクトルと TTS 制御パラメータの対応関係を学習するためのデータを自動生成するフェーズである。Phase 0 で訓練済みの感情エンコーダ（8D感情ベクトル出力）と VOICEVOX TTS エンジンを組み合わせ、「テキスト x 制御パラメータ → 音声」の三つ組データセットを構築する。

### 前提条件

- Phase 0 の学習済みチェックポイント（`artifacts/phase0/checkpoints/best_model.pt`）が Go 判定を通過していること
- VOICEVOX Engine がローカルで起動していること（デフォルト `http://127.0.0.1:50021`）

---

## 2. パイプラインフロー図

```
[WRIME Dataset]
      |
      v
+------------------------------+
| 1. テキスト選定              |
|   build_phase0_splits()      |
|   → 感情分布に基づくフィルタ |
|   → 層化サンプリング         |
+------------------------------+
      |
      v
+------------------------------+
| 2. 感情ベクトル推論          |
|   EmotionEncoder.encode_batch|
|   → text_emotion_vec[8]      |
+------------------------------+
      |
      v
+------------------------------+
| 3. パラメータグリッド生成    |
|   GridSampler                |
|   → control_params[5]        |
|   → LHS / Full Grid          |
+------------------------------+
      |
      v
+-----------------------------------+
| 4. 音声生成（バッチ）             |
|   VoicevoxClient                  |
|     POST /audio_query             |
|     → AudioQuery 書き換え          |
|     POST /synthesis               |
|   → WAV ファイル保存              |
|   チェックポイント/再開対応       |
+-----------------------------------+
      |
      v
+------------------------------+
| 5. 品質検証                  |
|   無音検出, ファイルサイズ   |
|   破損チェック               |
+------------------------------+
      |
      v
+------------------------------+
| 6. データセット組み立て      |
|   Parquet + メタデータ JSON  |
|   → TripletDataset           |
+------------------------------+
```

---

## 3. 制御空間の定義

### 3.1 5D 制御パラメータ

| パラメータ名   | 内部名称        | EmotionBridge 範囲 | VOICEVOX AudioQuery フィールド | VOICEVOX 値域       | マッピング方式                    |
|---------------|----------------|--------------------|-----------------------------|--------------------|---------------------------------|
| pitch_shift   | `pitch_shift`  | [-1.0, +1.0]      | `pitchScale`                | [-0.15, +0.15]     | 線形: `v * 0.15`               |
| pitch_range   | `pitch_range`  | [-1.0, +1.0]      | `intonationScale`           | [0.0, 2.0]         | 線形: `(v + 1) * 1.0`          |
| speed         | `speed`        | [-1.0, +1.0]      | `speedScale`                | [0.5, 2.0]         | 線形: `(v + 1) * 0.75 + 0.5`   |
| energy        | `energy`       | [-1.0, +1.0]      | `volumeScale`               | [0.3, 2.0]         | 線形: `(v + 1) * 0.85 + 0.3`   |
| pause_weight  | `pause_weight` | [-1.0, +1.0]      | `pauseLengthScale`          | [0.2, 3.0]         | 線形: `(v + 1) * 1.4 + 0.2`    |

### 3.2 制御パラメータとVOICEVOX値の相互変換

```python
@dataclass(slots=True, frozen=True)
class ControlParamMapping:
    """EmotionBridge正規化空間 [-1, +1] と VOICEVOX AudioQuery値の相互変換"""
    name: str
    voicevox_field: str
    vv_min: float   # VOICEVOX側の最小値
    vv_max: float   # VOICEVOX側の最大値

    def to_voicevox(self, normalized: float) -> float:
        """[-1, +1] → VOICEVOX値"""
        return self.vv_min + (normalized + 1.0) / 2.0 * (self.vv_max - self.vv_min)

    def from_voicevox(self, vv_value: float) -> float:
        """VOICEVOX値 → [-1, +1]"""
        return (vv_value - self.vv_min) / (self.vv_max - self.vv_min) * 2.0 - 1.0

CONTROL_PARAM_MAPPINGS: dict[str, ControlParamMapping] = {
    "pitch_shift":  ControlParamMapping("pitch_shift",  "pitchScale",        -0.15, 0.15),
    "pitch_range":  ControlParamMapping("pitch_range",  "intonationScale",    0.0,  2.0),
    "speed":        ControlParamMapping("speed",        "speedScale",         0.5,  2.0),
    "energy":       ControlParamMapping("energy",       "volumeScale",        0.3,  2.0),
    "pause_weight": ControlParamMapping("pause_weight", "pauseLengthScale",   0.2,  3.0),
}

CONTROL_PARAM_NAMES: list[str] = [
    "pitch_shift", "pitch_range", "speed", "energy", "pause_weight"
]
NUM_CONTROL_PARAMS = 5
```

---

## 4. パラメータグリッド設計

### 4.1 離散化戦略

各パラメータを [-1.0, +1.0] の範囲で等間隔に離散化する。

| 段階数 | グリッド値                                      | 全組み合わせ数 |
|--------|------------------------------------------------|---------------|
| 5段階  | [-1.0, -0.5, 0.0, 0.5, 1.0]                   | 5^5 = 3,125   |
| 7段階  | [-1.0, -0.67, -0.33, 0.0, 0.33, 0.67, 1.0]    | 7^5 = 16,807  |

### 4.2 サンプリング戦略: Latin Hypercube Sampling (LHS)

全数探索（Full Grid）はテキスト数が増えると組み合わせが爆発する。LHS を採用し、5D空間を効率的にカバーする。

- **デフォルト**: 各テキストにつき LHS で **128サンプル**を生成
- 全数探索モード: 開発時の検証用に 5段階 Full Grid（3,125組み合わせ）も選択可能
- LHS の seed は `DataConfig.random_seed` と text_id から決定論的に導出

```python
from dataclasses import dataclass
from enum import Enum

class SamplingStrategy(str, Enum):
    LHS = "lhs"           # Latin Hypercube Sampling
    FULL_GRID = "full_grid"  # 全数探索

@dataclass(slots=True)
class GridConfig:
    strategy: SamplingStrategy = SamplingStrategy.LHS
    lhs_samples_per_text: int = 128
    grid_steps: int = 5           # FULL_GRID時の各軸の段階数
    random_seed: int = 42
```

### 4.3 GridSampler インターフェース

```python
import numpy as np

class GridSampler:
    """5D制御パラメータ空間のサンプリング"""

    def __init__(self, config: GridConfig) -> None:
        self._config = config
        self._rng = np.random.default_rng(config.random_seed)

    def sample(self, text_id: int) -> np.ndarray:
        """
        指定テキストに対する制御パラメータ群を生成する。

        Args:
            text_id: テキストの一意識別子（LHSシード導出に使用）

        Returns:
            shape (n_samples, 5) の ndarray, 各値は [-1.0, +1.0]
        """
        ...

    def sample_lhs(self, n_samples: int, seed: int) -> np.ndarray:
        """
        Latin Hypercube Sampling で n_samples 点を 5D 空間から生成。

        Args:
            n_samples: サンプル数
            seed: 乱数シード

        Returns:
            shape (n_samples, 5) の ndarray, 各値は [-1.0, +1.0]
        """
        ...

    def sample_full_grid(self) -> np.ndarray:
        """
        全数探索グリッドを生成。

        Returns:
            shape (grid_steps^5, 5) の ndarray
        """
        ...

    @property
    def total_samples_per_text(self) -> int:
        """テキスト1件あたりのパラメータ組み合わせ数"""
        ...
```

---

## 5. テキストサンプル選定

### 5.1 選定基準

WRIMEデータセットから Phase 0 のフィルタリング後のデータを基に、以下の基準でサンプルを選定する。

1. **dominant emotion の均衡**: 8感情の dominant emotion（argmax）でグループ化し、各グループから均等にサンプリング
2. **感情強度の多様性**: dominant emotion の値が [0.33, 0.67, 1.0]（正規化後）の3レンジに分散
3. **テキスト長の多様性**: 短文（~20文字）、中文（20~60文字）、長文（60文字~）を含む
4. **重複排除**: 類似度が高いテキストは除外（簡易的に完全一致のみ）

### 5.2 サンプル数の設計

| パラメータ              | デフォルト値 | 根拠                                         |
|------------------------|-------------|---------------------------------------------|
| テキスト数             | 200         | 8感情 x 25テキスト/感情、多様性を確保       |
| パラメータ/テキスト    | 128 (LHS)   | 5D空間を十分にカバーしつつ実行可能な規模     |
| 総音声サンプル数       | 25,600      | 200 x 128 = 学習に十分なデータ量            |
| VOICEVOX話者数         | 1 (初期)    | 単一話者で制御パラメータの効果を分離         |

### 5.3 TextSelector インターフェース

```python
from dataclasses import dataclass

@dataclass(slots=True)
class SelectedText:
    text_id: int
    text: str
    emotion_vec: np.ndarray          # shape (8,), Phase 0 エンコーダ出力
    dominant_emotion: str            # EMOTION_LABELS のいずれか
    dominant_intensity: float        # [0.0, 1.0]
    source_split: str                # "train" | "val" | "test"

@dataclass(slots=True)
class TextSelectionConfig:
    num_texts: int = 200
    texts_per_emotion: int = 25
    intensity_bins: int = 3          # dominant intensity を何段階に分けるか
    min_text_length: int = 5
    max_text_length: int = 200
    random_seed: int = 42

class TextSelector:
    """WRIMEデータセットから学習用テキストを選定する"""

    def __init__(
        self,
        config: TextSelectionConfig,
        encoder: EmotionEncoder,
    ) -> None:
        self._config = config
        self._encoder = encoder

    def select(
        self,
        splits: dict[str, PreparedSplit],
    ) -> list[SelectedText]:
        """
        WRIMEの全splitからテキストを選定する。

        Args:
            splits: build_phase0_splits() の出力

        Returns:
            選定されたテキストのリスト（text_id は 0始まりの連番）
        """
        ...

    def _stratified_sample(
        self,
        texts: list[str],
        targets: np.ndarray,
        source_split: str,
    ) -> list[SelectedText]:
        """dominant emotion と intensity に基づく層化サンプリング"""
        ...
```

---

## 6. VOICEVOX クライアント

### 6.1 API フロー

```
1. GET  /speakers → 利用可能話者リスト取得
2. POST /audio_query?speaker={style_id}&text={text} → AudioQuery取得
3. AudioQuery のパラメータを制御値で上書き:
     pitchScale       ← ControlParamMapping.to_voicevox(pitch_shift)
     intonationScale  ← ControlParamMapping.to_voicevox(pitch_range)
     speedScale       ← ControlParamMapping.to_voicevox(speed)
     volumeScale      ← ControlParamMapping.to_voicevox(energy)
     pauseLengthScale ← ControlParamMapping.to_voicevox(pause_weight)
4. POST /synthesis?speaker={style_id} body=AudioQuery → WAV バイナリ
5. WAV をファイルに保存
```

### 6.2 VoicevoxClient インターフェース

```python
from dataclasses import dataclass
from pathlib import Path

@dataclass(slots=True)
class VoicevoxConfig:
    host: str = "127.0.0.1"
    port: int = 50021
    default_style_id: int = 3       # ずんだもん（ノーマル）
    timeout_seconds: float = 30.0
    max_retries: int = 3
    retry_delay_seconds: float = 1.0

    @property
    def base_url(self) -> str:
        return f"http://{self.host}:{self.port}"

@dataclass(slots=True)
class SynthesisResult:
    audio_path: Path
    duration_seconds: float          # 音声の長さ
    file_size_bytes: int
    sample_rate: int                 # 通常 24000
    is_valid: bool                   # 品質チェック通過フラグ
    error: str | None = None

class VoicevoxClient:
    """VOICEVOX Engine HTTP API クライアント"""

    def __init__(self, config: VoicevoxConfig) -> None:
        self._config = config

    async def get_speakers(self) -> list[dict]:
        """利用可能な話者スタイル一覧を取得"""
        ...

    async def synthesize(
        self,
        text: str,
        control_params: np.ndarray,    # shape (5,), [-1, +1]
        style_id: int | None = None,
        output_path: Path | None = None,
    ) -> SynthesisResult:
        """
        テキストと制御パラメータから音声を合成する。

        1. /audio_query でベースクエリ取得
        2. 制御パラメータで AudioQuery フィールドを上書き
        3. /synthesis で WAV 生成
        4. 品質チェック（無音検出、ファイルサイズ）

        Args:
            text: 合成するテキスト
            control_params: 5D制御パラメータ [-1, +1]
            style_id: VOICEVOX話者スタイルID（Noneならデフォルト）
            output_path: 保存先パス（Noneなら自動生成）

        Returns:
            SynthesisResult
        """
        ...

    async def _create_audio_query(
        self,
        text: str,
        style_id: int,
    ) -> dict:
        """POST /audio_query → AudioQuery dict"""
        ...

    def _apply_control_params(
        self,
        audio_query: dict,
        control_params: np.ndarray,
    ) -> dict:
        """AudioQuery dict に制御パラメータを適用"""
        ...

    async def _synthesize_audio(
        self,
        audio_query: dict,
        style_id: int,
    ) -> bytes:
        """POST /synthesis → WAV bytes"""
        ...

    async def health_check(self) -> bool:
        """VOICEVOX Engine の稼働確認"""
        ...
```

---

## 7. 音声生成パイプライン

### 7.1 GenerationPipeline インターフェース

```python
from dataclasses import dataclass, field
from pathlib import Path
from enum import Enum

class PipelineState(str, Enum):
    INITIALIZED = "initialized"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"

@dataclass(slots=True)
class GenerationConfig:
    output_dir: str = "artifacts/phase1"
    audio_subdir: str = "audio"
    max_concurrent_requests: int = 4     # VOICEVOX への同時リクエスト数
    checkpoint_interval: int = 100       # N サンプルごとにチェックポイント保存
    skip_existing: bool = True           # 既存ファイルをスキップ（再開時）
    voicevox: VoicevoxConfig = field(default_factory=VoicevoxConfig)
    grid: GridConfig = field(default_factory=GridConfig)
    text_selection: TextSelectionConfig = field(default_factory=TextSelectionConfig)

@dataclass(slots=True)
class GenerationProgress:
    total_tasks: int = 0
    completed_tasks: int = 0
    failed_tasks: int = 0
    skipped_tasks: int = 0              # skip_existing で飛ばしたもの
    elapsed_seconds: float = 0.0
    state: PipelineState = PipelineState.INITIALIZED

    @property
    def progress_ratio(self) -> float:
        if self.total_tasks == 0:
            return 0.0
        return (self.completed_tasks + self.skipped_tasks) / self.total_tasks

    @property
    def estimated_remaining_seconds(self) -> float:
        if self.completed_tasks == 0:
            return float("inf")
        rate = self.elapsed_seconds / self.completed_tasks
        remaining = self.total_tasks - self.completed_tasks - self.skipped_tasks
        return rate * remaining

class GenerationPipeline:
    """テキスト x 制御パラメータ → 音声 の一括生成パイプライン"""

    def __init__(
        self,
        config: GenerationConfig,
        encoder: EmotionEncoder,
    ) -> None:
        self._config = config
        self._encoder = encoder
        self._client = VoicevoxClient(config.voicevox)
        self._sampler = GridSampler(config.grid)
        self._selector = TextSelector(config.text_selection, encoder)
        self._progress = GenerationProgress()

    async def run(
        self,
        splits: dict[str, PreparedSplit],
    ) -> GenerationReport:
        """
        パイプライン全体を実行する。

        1. テキスト選定
        2. 各テキストに対するパラメータグリッド生成
        3. 非同期バッチ音声合成
        4. 品質チェック
        5. データセット組み立て
        6. レポート生成

        Args:
            splits: build_phase0_splits() の出力

        Returns:
            GenerationReport
        """
        ...

    async def _generate_batch(
        self,
        tasks: list[GenerationTask],
    ) -> list[GenerationResult]:
        """
        semaphore で同時実行数を制御しつつ非同期バッチ生成。

        Args:
            tasks: 生成タスクのリスト

        Returns:
            結果のリスト
        """
        ...

    def _save_checkpoint(self) -> None:
        """進捗状態をチェックポイントファイルに保存"""
        ...

    def _load_checkpoint(self) -> GenerationProgress | None:
        """既存のチェックポイントを読み込む（再開用）"""
        ...

    @property
    def progress(self) -> GenerationProgress:
        return self._progress
```

### 7.2 生成タスクの定義

```python
@dataclass(slots=True)
class GenerationTask:
    """音声生成の1単位"""
    task_id: str                         # "{text_id:04d}_{param_idx:04d}"
    text_id: int
    text: str
    emotion_vec: np.ndarray              # shape (8,)
    control_params: np.ndarray           # shape (5,), [-1, +1]
    style_id: int
    output_path: Path

@dataclass(slots=True)
class GenerationResult:
    task: GenerationTask
    synthesis: SynthesisResult | None    # None なら生成失敗
    error: str | None = None
```

### 7.3 非同期実行モデル

```python
import asyncio

async def _generate_batch(
    self,
    tasks: list[GenerationTask],
) -> list[GenerationResult]:
    semaphore = asyncio.Semaphore(self._config.max_concurrent_requests)

    async def _run_one(task: GenerationTask) -> GenerationResult:
        async with semaphore:
            try:
                result = await self._client.synthesize(
                    text=task.text,
                    control_params=task.control_params,
                    style_id=task.style_id,
                    output_path=task.output_path,
                )
                return GenerationResult(task=task, synthesis=result)
            except Exception as e:
                return GenerationResult(task=task, synthesis=None, error=str(e))

    results = await asyncio.gather(*[_run_one(t) for t in tasks])
    return list(results)
```

---

## 8. 三つ組データセットスキーマ

### 8.1 レコード構造

```python
@dataclass(slots=True)
class TripletRecord:
    """三つ組データセットの1レコード"""
    # テキスト情報
    text_id: int
    text: str
    # Phase 0 感情ベクトル
    emotion_vec: list[float]             # len=8, [0, 1]
    dominant_emotion: str
    # 制御パラメータ
    control_params: list[float]          # len=5, [-1, +1]
    # 音声情報
    audio_path: str                      # 相対パス (output_dir 基準)
    audio_duration_seconds: float
    audio_file_size_bytes: int
    sample_rate: int
    # VOICEVOX メタデータ
    style_id: int
    voicevox_params: dict[str, float]    # AudioQuery のスケール値（変換後）
    # 品質フラグ
    is_valid: bool
    generation_timestamp: str            # ISO 8601
```

### 8.2 ストレージ形式

**メインデータ**: Apache Parquet（列指向圧縮、pandas/polars との相性が良い）

```
artifacts/phase1/
  dataset/
    triplet_dataset.parquet        # メインデータ (全レコード)
    metadata.json                  # データセットメタデータ
  audio/
    0000/                          # text_id 0000 のディレクトリ
      0000_0000.wav                # text_id=0000, param_idx=0000
      0000_0001.wav
      ...
    0001/
      0001_0000.wav
      ...
  checkpoints/
    generation_progress.json       # チェックポイント
  reports/
    generation_report.json         # 生成統計レポート
    quality_report.json            # 品質チェック結果
```

### 8.3 Parquet スキーマ

```
text_id:               int32
text:                  string (utf-8)
emotion_joy:           float32
emotion_sadness:       float32
emotion_anticipation:  float32
emotion_surprise:      float32
emotion_anger:         float32
emotion_fear:          float32
emotion_disgust:       float32
emotion_trust:         float32
dominant_emotion:      string
ctrl_pitch_shift:      float32
ctrl_pitch_range:      float32
ctrl_speed:            float32
ctrl_energy:           float32
ctrl_pause_weight:     float32
audio_path:            string
audio_duration_sec:    float32
audio_file_size_bytes: int32
sample_rate:           int32
style_id:              int32
vv_pitchScale:         float32
vv_intonationScale:    float32
vv_speedScale:         float32
vv_volumeScale:        float32
vv_pauseLengthScale:   float32
is_valid:              bool
generation_timestamp:  string (ISO 8601)
```

### 8.4 metadata.json

```json
{
  "version": "1.0.0",
  "phase": "phase1",
  "created_at": "2026-02-16T12:00:00+09:00",
  "config": { "...GenerationConfig全体..." },
  "statistics": {
    "num_texts": 200,
    "num_params_per_text": 128,
    "total_records": 25600,
    "valid_records": 25400,
    "invalid_records": 200,
    "total_audio_duration_hours": 17.8,
    "total_audio_size_gb": 3.1,
    "dominant_emotion_distribution": {
      "joy": 25, "sadness": 25, "...": "..."
    }
  },
  "phase0_checkpoint": "artifacts/phase0/checkpoints/best_model.pt",
  "voicevox_version": "0.21.1",
  "speaker_info": { "style_id": 3, "name": "ずんだもん", "style": "ノーマル" }
}
```

---

## 9. データ品質管理

### 9.1 品質チェック項目

| チェック項目       | 条件                    | 失敗時の処理         |
|-------------------|------------------------|---------------------|
| ファイル存在       | WAV ファイルが存在       | is_valid = False    |
| ファイルサイズ     | > 1 KB                  | is_valid = False    |
| 音声長             | > 0.1 秒               | is_valid = False    |
| 無音検出           | RMS > 閾値 (0.001)      | is_valid = False    |
| サンプルレート     | == 24000 (VOICEVOX標準) | 警告ログ            |
| WAV ヘッダ整合性  | wave モジュールで読込可  | is_valid = False    |

### 9.2 AudioValidator インターフェース

```python
from dataclasses import dataclass

@dataclass(slots=True)
class ValidationConfig:
    min_file_size_bytes: int = 1024       # 1 KB
    min_duration_seconds: float = 0.1
    min_rms_amplitude: float = 0.001
    expected_sample_rate: int = 24000

@dataclass(slots=True)
class ValidationResult:
    is_valid: bool
    file_size_bytes: int
    duration_seconds: float
    sample_rate: int
    rms_amplitude: float
    errors: list[str]                     # 失敗理由のリスト

class AudioValidator:
    """生成音声の品質検証"""

    def __init__(self, config: ValidationConfig | None = None) -> None:
        self._config = config or ValidationConfig()

    def validate(self, audio_path: Path) -> ValidationResult:
        """
        WAV ファイルの品質を検証する。

        Args:
            audio_path: WAV ファイルのパス

        Returns:
            ValidationResult
        """
        ...

    def validate_batch(
        self,
        audio_paths: list[Path],
    ) -> list[ValidationResult]:
        """複数ファイルの一括検証"""
        ...
```

### 9.3 GenerationReport

```python
@dataclass(slots=True)
class GenerationReport:
    """生成パイプラインの実行レポート"""
    total_tasks: int
    completed: int
    failed: int
    skipped: int
    invalid: int                          # 品質チェック不合格
    elapsed_seconds: float
    avg_synthesis_time_seconds: float
    total_audio_duration_seconds: float
    total_audio_size_bytes: int
    dominant_emotion_distribution: dict[str, int]
    failure_reasons: dict[str, int]       # エラー種別ごとの件数
    config_snapshot: dict                 # GenerationConfig の辞書化

    def to_dict(self) -> dict:
        """JSON シリアライズ用"""
        ...
```

---

## 10. 設定システム拡張

### 10.1 Phase1Config

既存の `config.py` の設定パターンに倣い、`Phase1Config` を定義する。

```python
# emotionbridge/config.py に追加

@dataclass(slots=True)
class VoicevoxConfig:
    host: str = "127.0.0.1"
    port: int = 50021
    default_style_id: int = 3
    timeout_seconds: float = 30.0
    max_retries: int = 3
    retry_delay_seconds: float = 1.0

@dataclass(slots=True)
class GridConfig:
    strategy: str = "lhs"                # "lhs" | "full_grid"
    lhs_samples_per_text: int = 128
    grid_steps: int = 5
    random_seed: int = 42

@dataclass(slots=True)
class TextSelectionConfig:
    num_texts: int = 200
    texts_per_emotion: int = 25
    intensity_bins: int = 3
    min_text_length: int = 5
    max_text_length: int = 200
    random_seed: int = 42

@dataclass(slots=True)
class ValidationConfig:
    min_file_size_bytes: int = 1024
    min_duration_seconds: float = 0.1
    min_rms_amplitude: float = 0.001
    expected_sample_rate: int = 24000

@dataclass(slots=True)
class GenerationConfig:
    output_dir: str = "artifacts/phase1"
    audio_subdir: str = "audio"
    max_concurrent_requests: int = 4
    checkpoint_interval: int = 100
    skip_existing: bool = True

@dataclass(slots=True)
class Phase1Config:
    phase0_checkpoint: str = "artifacts/phase0/checkpoints/best_model.pt"
    voicevox: VoicevoxConfig = field(default_factory=VoicevoxConfig)
    grid: GridConfig = field(default_factory=GridConfig)
    text_selection: TextSelectionConfig = field(default_factory=TextSelectionConfig)
    validation: ValidationConfig = field(default_factory=ValidationConfig)
    generation: GenerationConfig = field(default_factory=GenerationConfig)
    # Phase 0 データ設定（テキスト選定用）
    data: DataConfig = field(default_factory=DataConfig)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
```

### 10.2 configs/phase1.yaml の例

```yaml
phase0_checkpoint: artifacts/phase0/checkpoints/best_model.pt

data:
  dataset_name: shunk031/wrime
  dataset_config_name: ver1
  text_field: sentence
  label_source: avg_readers
  filter_max_intensity_lte: 1
  random_seed: 42

voicevox:
  host: 127.0.0.1
  port: 50021
  default_style_id: 3
  timeout_seconds: 30.0
  max_retries: 3

grid:
  strategy: lhs
  lhs_samples_per_text: 128
  random_seed: 42

text_selection:
  num_texts: 200
  texts_per_emotion: 25
  intensity_bins: 3

validation:
  min_file_size_bytes: 1024
  min_duration_seconds: 0.1
  min_rms_amplitude: 0.001

generation:
  output_dir: artifacts/phase1
  max_concurrent_requests: 4
  checkpoint_interval: 100
  skip_existing: true
```

---

## 11. モジュール構成

```
emotionbridge/
  generation/
    __init__.py
    pipeline.py        # GenerationPipeline, GenerationTask, GenerationResult
    grid.py            # GridSampler, GridConfig, SamplingStrategy
    voicevox.py        # VoicevoxClient, VoicevoxConfig, SynthesisResult
    text_selector.py   # TextSelector, TextSelectionConfig, SelectedText
    validator.py       # AudioValidator, ValidationConfig, ValidationResult
    control_params.py  # ControlParamMapping, CONTROL_PARAM_MAPPINGS
    dataset.py         # TripletRecord, dataset I/O (Parquet read/write)
    report.py          # GenerationReport
```

---

## 12. スケール見積もり

### 12.1 総サンプル数

| シナリオ      | テキスト数 | パラメータ/テキスト | 総サンプル数 |
|--------------|-----------|-------------------|-------------|
| 開発・テスト  | 10        | 32 (LHS)          | 320         |
| 標準         | 200       | 128 (LHS)         | 25,600      |
| 大規模       | 500       | 256 (LHS)         | 128,000     |

### 12.2 生成時間の見積もり

VOICEVOX Engine のローカル合成は、テキスト長に依存するが概ね 1リクエストあたり **0.3~1.0秒**（GPU未使用時）。

| シナリオ | 総サンプル数 | 平均合成時間 | 同時リクエスト | 推定総時間       |
|---------|-------------|-------------|--------------|-----------------|
| 開発     | 320         | 0.5秒       | 4            | ~40秒           |
| 標準     | 25,600      | 0.5秒       | 4            | ~53分           |
| 大規模   | 128,000     | 0.5秒       | 4            | ~4.4時間        |

### 12.3 ストレージ容量の見積もり

VOICEVOX の出力は 24kHz 16bit mono WAV。平均音声長を約 2.5秒と仮定すると、1ファイルあたり約 120 KB。

| シナリオ | 総サンプル数 | ファイルサイズ/件 | 総容量       | Parquet サイズ |
|---------|-------------|-----------------|-------------|---------------|
| 開発     | 320         | ~120 KB         | ~38 MB      | ~50 KB        |
| 標準     | 25,600      | ~120 KB         | ~3.0 GB     | ~3 MB         |
| 大規模   | 128,000     | ~120 KB         | ~15.0 GB    | ~15 MB        |

---

## 13. チェックポイント/再開機能

### 13.1 チェックポイント形式

```json
{
  "version": "1.0.0",
  "state": "running",
  "completed_task_ids": ["0000_0000", "0000_0001", "..."],
  "failed_task_ids": ["0005_0032"],
  "total_tasks": 25600,
  "last_updated": "2026-02-16T14:30:00+09:00",
  "config_hash": "sha256:abc123..."
}
```

### 13.2 再開ロジック

1. `generation_progress.json` が存在するか確認
2. 存在する場合、`config_hash` を照合（設定変更を検出）
3. `completed_task_ids` と `failed_task_ids` を読み込み
4. `skip_existing=True` なら完了タスクをスキップ
5. 失敗タスクはリトライ対象に含める

---

## 14. 設計判断の根拠

### Q1: なぜ LHS を採用するか？

全数探索（Full Grid）は 5^5 = 3,125 や 7^5 = 16,807 の組み合わせを全テキストに適用するため、200テキストで最大 336万サンプルになり非現実的。LHS は同じサンプル数で各次元を均等にカバーでき、5D空間の探索効率が高い。128サンプル/テキストで十分な空間カバレッジが得られる。

### Q2: なぜ非同期 (asyncio) を採用するか？

VOICEVOX Engine は HTTP API であり、合成処理の大部分は I/O バウンド（HTTP通信 + エンジン側の処理待ち）。`asyncio` + `aiohttp` により、4並列程度の同時リクエストでスループットを約3~4倍に改善できる。マルチプロセスは VOICEVOX 側がボトルネックになるため不要。

### Q3: なぜ Parquet を選択するか？

- 列指向圧縮によりストレージ効率が高い
- pandas / polars で高速に読み書き可能
- 型情報がスキーマに埋め込まれ、列定義の整合性を保ちやすい
- JSON Lines より圧縮率が高く、数万レコードの一括読み込みが高速

### Q4: なぜ制御パラメータ空間を [-1, +1] に正規化するか？

- TTS エンジン非依存の統一表現。VOICEVOX 以外の TTS（Phase 2以降）に切り替える際、マッピング関数のみ変更すれば良い
- ニューラルネットワークの学習に適した値域
- Proposal Section 4.3 の設計方針に準拠

### Q5: 単一話者で開始する理由は？

Phase 1 の目的は制御パラメータの効果を学習することであり、話者のバリエーションは交絡因子になる。まず単一話者で制御パラメータ→感情の対応を確立し、Phase 2 以降で多話者に拡張する。

### Q6: テキスト200件の根拠は？

8感情 x 25テキスト/感情 = 200件。各感情から十分な数のサンプルを確保しつつ、128パラメータ/テキストとの掛け合わせで 25,600 サンプルという、学習に十分かつ生成が1時間以内で完了する規模を目標とした。

---

## 15. 品質管理チェックリスト

### 生成前

- [ ] Phase 0 チェックポイントが Go 判定を通過している
- [ ] VOICEVOX Engine が起動し、health check が通る
- [ ] 指定した style_id が `/speakers` に存在する
- [ ] 出力ディレクトリに十分なディスク容量がある（見積もり容量の1.5倍以上）
- [ ] 設定ファイルのパラメータ値が妥当な範囲内

### 生成中

- [ ] チェックポイントが `checkpoint_interval` ごとに保存される
- [ ] 進捗ログが定期的に出力される（完了率、推定残り時間）
- [ ] 失敗率が閾値（5%）を超えた場合に警告ログ
- [ ] VOICEVOX Engine のヘルスチェックを定期実行

### 生成後

- [ ] 総サンプル数が期待値と一致（許容: 失敗分を除外して95%以上）
- [ ] is_valid = True のレコード比率が 95% 以上
- [ ] 8感情すべてに対するテキストが存在する（dominant emotion 分布）
- [ ] 音声ファイルがすべて読み込み可能
- [ ] Parquet ファイルが正常に読み込み可能
- [ ] metadata.json が正しく書き出されている
- [ ] 生成レポート（generation_report.json, quality_report.json）が出力されている
