# Phase 1 統合アーキテクチャ設計書

## 1. 概要

本文書は Phase 1「VOICEVOX連携 + データ生成パイプライン」の統合アーキテクチャ設計を定義する。
個別設計文書（`phase1_voicevox_client.md`, `phase1_data_pipeline.md`）の成果を統合し、
モジュール間の不整合を解消した最終的な設計方針を示す。

### 設計文書間の不整合と解決

| 不整合点 | Task #1 (VOICEVOXクライアント) | Task #2 (データ生成) | 統合方針 |
|---|---|---|---|
| パラメータマッピング範囲 | Proposal準拠の保守的範囲 (例: speedScale [0.5, 1.5]) | 広い範囲 (例: speedScale [0.5, 2.0]) | **Task #1を採用**。保守的範囲で開始し、聴取評価後に拡張。ただし範囲を設定で変更可能にする |
| VoicevoxClient定義 | `tts/voicevox_client.py` に型付きクライアント | `generation/voicevox.py` に別クライアント | **Task #1を正とする**。`tts/`パッケージに一元化し、`generation/`からは`tts`をインポート |
| ControlVectorの型 | `ControlVector` frozen dataclass (名前付きフィールド) | `np.ndarray` shape (5,) | **Task #1を採用**。型安全性・可読性を優先。numpy変換メソッドを提供 |
| VoicevoxConfig形式 | `base_url: str` | `host: str` + `port: int` | **Task #2を採用**。`host`/`port`分離の方が設定しやすい。`base_url`プロパティを提供 |
| HTTPクライアント | httpx (型アノテーション完備) | aiohttp (言及のみ) | **httpx採用**。Phase 0との一貫性、型サポートの充実 |
| pause_weight変換先 | 3パラメータ (pre/postPhonemeLength + pauseLengthScale) | 1パラメータ (pauseLengthScale のみ) | **Task #1を採用**。統合的な「間」制御を実現 |

---

## 2. 統合アーキテクチャ図

```
┌─────────────────────────────────────────────────────────────────────┐
│                         Phase 1 全体構成                             │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  [CLI Layer]                                                        │
│    cli.py                                                           │
│      ├── generate-samples   (パラメータグリッドサーチ + 音声生成)    │
│      ├── list-speakers      (VOICEVOX話者一覧)                      │
│      └── synthesize         (単一テキスト感情音声合成)               │
│                                                                     │
│  ┌──────────────────────┐    ┌──────────────────────────────┐       │
│  │  emotionbridge/tts/  │    │  emotionbridge/generation/   │       │
│  │  (TTS抽象層)         │    │  (データ生成パイプライン)     │       │
│  │                      │    │                              │       │
│  │  types.py            │◄───│  pipeline.py                 │       │
│  │  voicevox_client.py  │◄───│  grid.py                     │       │
│  │  adapter.py          │◄───│  text_selector.py            │       │
│  │  exceptions.py       │    │  validator.py                │       │
│  └───────┬──────────────┘    │  dataset.py                  │       │
│          │                   │  report.py                   │       │
│          │                   └──────────────┬───────────────┘       │
│          │                                  │                       │
│  ┌───────▼──────────────────────────────────▼───────────────┐       │
│  │  既存 Phase 0 モジュール                                  │       │
│  │                                                           │       │
│  │  constants.py    (EMOTION_LABELS, CONTROL_PARAM_NAMES)    │       │
│  │  config.py       (Phase0Config + Phase1Config)            │       │
│  │  inference/      (EmotionEncoder)                         │       │
│  │  data/           (build_phase0_splits, PreparedSplit)     │       │
│  └───────────────────────────────────────────────────────────┘       │
│                                                                     │
│  [External]                                                         │
│    VOICEVOX Engine (http://localhost:50021)                          │
│    WRIME Dataset (shunk031/wrime on HF Hub)                         │
│    Phase 0 Checkpoint (artifacts/phase0/checkpoints/best_model.pt)  │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 3. ディレクトリ構成

```
emotionbridge/
  __init__.py                  # 既存（EmotionEncoder, EMOTION_LABELS エクスポート）
  constants.py                 # 既存 + Phase 1 定数追加
  config.py                    # 既存 + Phase1Config 追加
  cli.py                       # 既存 + Phase 1 コマンド追加

  # --- Phase 0 (既存・変更なし) ---
  data/
    __init__.py
    wrime.py
    visualize.py
  model/
    __init__.py
    regressor.py
  training/
    __init__.py
    trainer.py
    metrics.py
  inference/
    __init__.py
    encoder.py

  # --- Phase 1 (新規) ---
  tts/                         # TTS抽象層 + VOICEVOX実装
    __init__.py                # TTSAdapter, VoicevoxAdapter, VoicevoxClient, ControlVector エクスポート
    types.py                   # ControlVector, AudioQuery, Mora, AccentPhrase, SpeakerInfo, SpeakerStyle
    exceptions.py              # VoicevoxError 階層
    voicevox_client.py         # VoicevoxClient (httpx非同期)
    adapter.py                 # TTSAdapter(ABC), VoicevoxAdapter

  generation/                  # データ生成パイプライン
    __init__.py                # GenerationPipeline, GridSampler エクスポート
    pipeline.py                # GenerationPipeline, GenerationTask, GenerationResult
    grid.py                    # GridSampler, SamplingStrategy
    text_selector.py           # TextSelector, SelectedText
    validator.py               # AudioValidator, ValidationResult
    dataset.py                 # TripletRecord, Parquet I/O
    report.py                  # GenerationReport

configs/
  phase0.yaml                  # 既存
  phase1.yaml                  # Phase 1 設定

artifacts/
  phase0/                      # 既存
  phase1/                      # Phase 1 出力
    audio/                     # 生成WAVファイル (text_id別サブディレクトリ)
      0000/
        0000_0000.wav
        ...
    dataset/
      triplet_dataset.parquet  # 三つ組データセット
      metadata.json
    checkpoints/
      generation_progress.json # 生成パイプラインチェックポイント
    reports/
      generation_report.json
      quality_report.json
    effective_config.yaml      # 実行時設定スナップショット
```

---

## 4. モジュール依存関係

```
emotionbridge/tts/types.py
  └── (依存なし: ControlVector, AudioQuery 等の値オブジェクト)

emotionbridge/tts/exceptions.py
  └── (依存なし: VoicevoxError 階層)

emotionbridge/tts/voicevox_client.py
  ├── tts/types.py          (AudioQuery, SpeakerInfo)
  ├── tts/exceptions.py     (VoicevoxError系)
  └── httpx                 (外部: 非同期HTTPクライアント)

emotionbridge/tts/adapter.py
  ├── tts/types.py          (ControlVector, AudioQuery)
  └── abc                   (標準: TTSAdapter ABC)

emotionbridge/generation/grid.py
  ├── tts/types.py          (ControlVector)
  ├── numpy                 (外部: LHS/グリッド生成)
  └── scipy.stats.qmc       (外部: LatinHypercube)

emotionbridge/generation/text_selector.py
  ├── data/wrime.py         (PreparedSplit)
  ├── inference/encoder.py  (EmotionEncoder)
  ├── constants.py          (EMOTION_LABELS)
  └── numpy                 (外部)

emotionbridge/generation/validator.py
  ├── numpy                 (外部: RMS計算)
  └── wave                  (標準: WAVヘッダ検証)

emotionbridge/generation/dataset.py
  ├── tts/types.py          (ControlVector)
  ├── constants.py          (EMOTION_LABELS, CONTROL_PARAM_NAMES)
  └── pyarrow               (外部: Parquet I/O)

emotionbridge/generation/pipeline.py
  ├── tts/voicevox_client.py (VoicevoxClient)
  ├── tts/adapter.py         (VoicevoxAdapter)
  ├── tts/types.py           (ControlVector)
  ├── generation/grid.py     (GridSampler)
  ├── generation/text_selector.py (TextSelector)
  ├── generation/validator.py (AudioValidator)
  ├── generation/dataset.py  (TripletRecord, Parquet出力)
  ├── generation/report.py   (GenerationReport)
  ├── inference/encoder.py   (EmotionEncoder)
  └── asyncio                (標準: 非同期制御)

emotionbridge/cli.py
  ├── config.py              (Phase0Config, Phase1Config, load_config)
  ├── generation/pipeline.py (GenerationPipeline)
  ├── tts/voicevox_client.py (VoicevoxClient — list-speakers用)
  └── inference/encoder.py   (EmotionEncoder — synthesize用)
```

---

## 5. 統合設定システム

### 5.1 Phase1Config (config.py への追加)

```python
from dataclasses import dataclass, field, asdict
from typing import Any

# --- Phase 1 設定 ---

@dataclass(slots=True)
class VoicevoxConfig:
    """VOICEVOX Engine接続設定。"""
    host: str = "127.0.0.1"
    port: int = 50021
    default_speaker_id: int = 0
    timeout: float = 30.0
    max_retries: int = 3
    retry_delay: float = 0.5
    output_sampling_rate: int = 24000
    output_stereo: bool = False

    @property
    def base_url(self) -> str:
        return f"http://{self.host}:{self.port}"

@dataclass(slots=True)
class ControlSpaceConfig:
    """制御空間のマッピング範囲設定。

    デフォルト値はProposal Section 3.2に基づく保守的な範囲。
    聴取評価後に段階的に拡張することを想定。
    """
    speed_range: tuple[float, float] = (0.5, 1.5)
    pitch_shift_range: tuple[float, float] = (-0.15, 0.15)
    pitch_range_range: tuple[float, float] = (0.5, 1.5)
    energy_range: tuple[float, float] = (0.5, 1.5)
    pause_pre_range: tuple[float, float] = (0.0, 0.2)
    pause_post_range: tuple[float, float] = (0.0, 0.2)
    pause_length_scale_range: tuple[float, float] = (0.5, 1.5)

@dataclass(slots=True)
class GridConfig:
    """パラメータグリッド設定。"""
    strategy: str = "lhs"              # "lhs" | "full_grid"
    lhs_samples_per_text: int = 128
    grid_steps: int = 5                # full_grid時の各軸段階数
    random_seed: int = 42

@dataclass(slots=True)
class TextSelectionConfig:
    """テキスト選定設定。"""
    num_texts: int = 200
    texts_per_emotion: int = 25
    intensity_bins: int = 3
    min_text_length: int = 5
    max_text_length: int = 200
    random_seed: int = 42

@dataclass(slots=True)
class ValidationConfig:
    """音声品質検証設定。"""
    min_file_size_bytes: int = 1024
    min_duration_seconds: float = 0.1
    min_rms_amplitude: float = 0.001
    expected_sample_rate: int = 24000

@dataclass(slots=True)
class GenerationConfig:
    """音声生成パイプライン設定。"""
    output_dir: str = "artifacts/phase1"
    audio_subdir: str = "audio"
    max_concurrent_requests: int = 4
    checkpoint_interval: int = 100
    skip_existing: bool = True

@dataclass(slots=True)
class Phase1Config:
    """Phase 1全体の設定。"""
    phase0_checkpoint: str = "artifacts/phase0/checkpoints/best_model.pt"
    device: str = "cuda"
    voicevox: VoicevoxConfig = field(default_factory=VoicevoxConfig)
    control_space: ControlSpaceConfig = field(default_factory=ControlSpaceConfig)
    grid: GridConfig = field(default_factory=GridConfig)
    text_selection: TextSelectionConfig = field(default_factory=TextSelectionConfig)
    validation: ValidationConfig = field(default_factory=ValidationConfig)
    generation: GenerationConfig = field(default_factory=GenerationConfig)
    data: DataConfig = field(default_factory=DataConfig)  # WRIME読み込み用

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
```

### 5.2 設定のロードフロー

```python
def load_config(config_path: str) -> Phase0Config | Phase1Config:
    """YAML設定ファイルを読み込み、適切なConfig型を返す。

    Phase判定: YAMLトップレベルに 'voicevox' キーがあればPhase1Config、
    'model' キーがあればPhase0Configとして解釈する。
    """
    ...
```

### 5.3 configs/phase1.yaml

```yaml
phase0_checkpoint: artifacts/phase0/checkpoints/best_model.pt
device: cuda

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
  default_speaker_id: 0
  timeout: 30.0
  max_retries: 3
  retry_delay: 0.5
  output_sampling_rate: 24000

control_space:
  speed_range: [0.5, 1.5]
  pitch_shift_range: [-0.15, 0.15]
  pitch_range_range: [0.5, 1.5]
  energy_range: [0.5, 1.5]
  pause_pre_range: [0.0, 0.2]
  pause_post_range: [0.0, 0.2]
  pause_length_scale_range: [0.5, 1.5]

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

## 6. 定数の拡張 (constants.py)

```python
# --- 既存 (Phase 0) ---
EMOTION_LABELS: list[str] = [
    "joy", "sadness", "anticipation", "surprise",
    "anger", "fear", "disgust", "trust",
]
NUM_EMOTIONS: int = 8
LABEL_SCALE_MAX: float = 3.0

MAJOR_EMOTION_LABELS: list[str] = [...]
LOW_VARIANCE_EMOTION_LABELS: list[str] = [...]

# --- 新規 (Phase 1) ---
CONTROL_PARAM_NAMES: list[str] = [
    "pitch_shift", "pitch_range", "speed", "energy", "pause_weight",
]
NUM_CONTROL_PARAMS: int = 5
```

---

## 7. 統合データフロー

### 7.1 generate-samples コマンド（データ生成パイプライン）

```
[CLI: generate-samples --config configs/phase1.yaml]
  │
  ├─ 1. load_config("configs/phase1.yaml") → Phase1Config
  │
  ├─ 2. build_phase0_splits(config.data) → dict[str, PreparedSplit]
  │
  ├─ 3. EmotionEncoder(config.phase0_checkpoint) → encoder
  │
  ├─ 4. TextSelector(config.text_selection, encoder)
  │     .select(splits) → list[SelectedText]
  │     (200テキスト: 8感情 x 25件、層化サンプリング)
  │
  ├─ 5. GridSampler(config.grid)
  │     .sample(text_id) → np.ndarray (128, 5)
  │     (各テキストにLHSで128パラメータ組み合わせ)
  │
  ├─ 6. GenerationPipeline 非同期バッチ生成:
  │     for each (text, params_128) pair:
  │       for each control_params in params_128:
  │         ┌───────────────────────────────────────────┐
  │         │ a. ControlVector(*control_params)          │
  │         │ b. VoicevoxClient.audio_query(text, spk)  │
  │         │ c. VoicevoxAdapter.apply(query, ctrl)     │
  │         │ d. VoicevoxClient.synthesis(query, spk)   │
  │         │ e. WAV保存 → audio/XXXX/XXXX_YYYY.wav     │
  │         │ f. AudioValidator.validate(wav_path)       │
  │         └───────────────────────────────────────────┘
  │     (asyncio.Semaphore で同時4リクエスト制御)
  │     (100サンプルごとにチェックポイント保存)
  │
  ├─ 7. TripletDataset → Parquet書き出し
  │     triplet_dataset.parquet + metadata.json
  │
  └─ 8. GenerationReport → reports/ 出力
```

### 7.2 synthesize コマンド（単一テキスト推論）

```
[CLI: synthesize --config configs/phase1.yaml --text "今日は楽しかった！" --output out.wav]
  │
  ├─ 1. EmotionEncoder.encode(text) → emotion_vec[8]
  │
  ├─ 2. (Phase 1時点) ヒューリスティックマッパーで ControlVector 生成
  │     (Phase 3以降: 学習済みパラメータ生成器で推論)
  │
  ├─ 3. VoicevoxClient.audio_query(text, speaker_id) → AudioQuery
  │
  ├─ 4. VoicevoxAdapter.apply(audio_query, control_vector) → AudioQuery (修正済み)
  │
  ├─ 5. VoicevoxClient.synthesis(audio_query, speaker_id) → bytes (WAV)
  │
  └─ 6. ファイル書き出し
```

### 7.3 list-speakers コマンド

```
[CLI: list-speakers --config configs/phase1.yaml]
  │
  ├─ 1. VoicevoxClient.health_check() → 接続確認
  │
  └─ 2. VoicevoxClient.speakers() → list[SpeakerInfo]
        → style_type == "talk" のみフィルタして表示
```

---

## 8. 統合型定義

### 8.1 ControlVector (正規化表現)

Task #1の`ControlVector` frozen dataclassを正とする。numpy変換を提供。

```python
@dataclass(frozen=True, slots=True)
class ControlVector:
    """TTS制御空間の5次元ベクトル。各値は [-1.0, +1.0]。"""
    pitch_shift: float = 0.0
    pitch_range: float = 0.0
    speed: float = 0.0
    energy: float = 0.0
    pause_weight: float = 0.0

    def __post_init__(self) -> None:
        for name in CONTROL_PARAM_NAMES:
            val = getattr(self, name)
            if not (-1.0 <= val <= 1.0):
                raise ValueError(f"{name} must be in [-1.0, 1.0], got {val}")

    def to_numpy(self) -> np.ndarray:
        """shape (5,) のnumpy配列に変換。順序はCONTROL_PARAM_NAMES準拠。"""
        return np.array([
            self.pitch_shift, self.pitch_range,
            self.speed, self.energy, self.pause_weight,
        ], dtype=np.float32)

    @classmethod
    def from_numpy(cls, arr: np.ndarray) -> "ControlVector":
        """shape (5,) のnumpy配列からControlVectorを生成。"""
        if arr.shape != (5,):
            raise ValueError(f"Expected shape (5,), got {arr.shape}")
        return cls(
            pitch_shift=float(arr[0]),
            pitch_range=float(arr[1]),
            speed=float(arr[2]),
            energy=float(arr[3]),
            pause_weight=float(arr[4]),
        )
```

### 8.2 VoicevoxAdapter (マッピング実装)

マッピング範囲を`ControlSpaceConfig`から取得し、設定で変更可能にする。

```python
class VoicevoxAdapter(TTSAdapter):
    """VOICEVOX Engine用TTSアダプタ。制御空間→VOICEVOXパラメータ線形マッピング。"""

    def __init__(self, config: ControlSpaceConfig | None = None) -> None:
        self._config = config or ControlSpaceConfig()

    def apply(self, audio_query: AudioQuery, control: ControlVector) -> AudioQuery:
        """制御ベクトルをAudioQueryに適用する。元のAudioQueryは変更しない。

        マッピング:
            speedScale       = lerp(speed_range, control.speed)
            pitchScale       = lerp(pitch_shift_range, control.pitch_shift)
            intonationScale  = lerp(pitch_range_range, control.pitch_range)
            volumeScale      = lerp(energy_range, control.energy)
            prePhonemeLength  = lerp(pause_pre_range, control.pause_weight)
            postPhonemeLength = lerp(pause_post_range, control.pause_weight)
            pauseLengthScale  = lerp(pause_length_scale_range, control.pause_weight)

        ここで lerp(range, v) = (range[0] + range[1]) / 2 + v * (range[1] - range[0]) / 2
        つまり v=0 で範囲の中央値、v=-1 で最小値、v=+1 で最大値。
        """
        cfg = self._config
        return replace(
            audio_query,
            speedScale=_lerp(cfg.speed_range, control.speed),
            pitchScale=_lerp(cfg.pitch_shift_range, control.pitch_shift),
            intonationScale=_lerp(cfg.pitch_range_range, control.pitch_range),
            volumeScale=_lerp(cfg.energy_range, control.energy),
            prePhonemeLength=_lerp(cfg.pause_pre_range, control.pause_weight),
            postPhonemeLength=_lerp(cfg.pause_post_range, control.pause_weight),
            pauseLengthScale=_lerp(cfg.pause_length_scale_range, control.pause_weight),
        )


def _lerp(range_: tuple[float, float], normalized: float) -> float:
    """[-1, +1] の正規化値を [min, max] にマッピング。"""
    lo, hi = range_
    return (lo + hi) / 2.0 + normalized * (hi - lo) / 2.0
```

---

## 9. CLI拡張設計

### 9.1 コマンド一覧

```python
# cli.py に追加

@app.command()
def generate_samples(
    config: str = typer.Option(..., "--config", help="Phase 1設定ファイルパス"),
) -> None:
    """パラメータグリッドサーチによる音声サンプルの一括生成。

    WRIMEデータセットからテキストを選定し、5D制御空間のパラメータ組み合わせに対して
    VOICEVOX音声を生成する。出力は三つ組データセット (Parquet) として保存される。

    中断した場合は同じコマンドで再開可能 (skip_existing=true)。
    """
    ...

@app.command()
def list_speakers(
    config: str = typer.Option("configs/phase1.yaml", "--config"),
) -> None:
    """VOICEVOX Engineの利用可能なキャラクター・スタイル一覧を表示。"""
    ...

@app.command()
def synthesize(
    config: str = typer.Option("configs/phase1.yaml", "--config"),
    text: str = typer.Option(..., "--text", help="合成するテキスト"),
    output: str = typer.Option("output.wav", "--output", help="出力WAVファイルパス"),
    speaker_id: int | None = typer.Option(None, "--speaker-id"),
) -> None:
    """単一テキストの感情音声合成。

    Phase 0の感情エンコーダでテキストを分析し、ヒューリスティックなパラメータ
    マッピングでVOICEVOX音声を生成する。
    (Phase 3以降: 学習済みパラメータ生成器に置き換え)
    """
    ...
```

### 9.2 synthesize コマンドの暫定マッパー

Phase 1時点では学習済みパラメータ生成器が存在しないため、ヒューリスティックなマッパーを提供する。
感情ベクトルの支配的な感情に応じて事前定義された ControlVector を返す。

```python
# emotionbridge/tts/heuristic_mapper.py

EMOTION_TO_CONTROL: dict[str, ControlVector] = {
    "joy":          ControlVector(pitch_shift=0.4,  pitch_range=0.5,  speed=0.2,  energy=0.4,  pause_weight=-0.2),
    "sadness":      ControlVector(pitch_shift=-0.3, pitch_range=-0.3, speed=-0.3, energy=-0.3, pause_weight=0.3),
    "anticipation": ControlVector(pitch_shift=0.1,  pitch_range=0.2,  speed=0.1,  energy=0.1,  pause_weight=-0.1),
    "surprise":     ControlVector(pitch_shift=0.5,  pitch_range=0.6,  speed=0.3,  energy=0.5,  pause_weight=-0.3),
    "anger":        ControlVector(pitch_shift=0.2,  pitch_range=0.3,  speed=0.2,  energy=0.6,  pause_weight=-0.2),
    "fear":         ControlVector(pitch_shift=0.2,  pitch_range=-0.2, speed=0.2,  energy=-0.2, pause_weight=0.2),
    "disgust":      ControlVector(pitch_shift=-0.2, pitch_range=-0.1, speed=-0.1, energy=0.1,  pause_weight=0.1),
    "trust":        ControlVector(pitch_shift=0.0,  pitch_range=0.1,  speed=-0.1, energy=0.0,  pause_weight=0.0),
}

def heuristic_map(emotion_vec: np.ndarray) -> ControlVector:
    """8D感情ベクトルから支配的感情に基づくControlVectorを返す。

    感情ベクトルの各要素を重みとして、事前定義テンプレートの加重平均を計算。
    全感情が弱い場合（max < 0.1）はニュートラル（全0）を返す。
    """
    if emotion_vec.max() < 0.1:
        return ControlVector()  # ニュートラル

    weighted = np.zeros(NUM_CONTROL_PARAMS, dtype=np.float32)
    for i, label in enumerate(EMOTION_LABELS):
        template = EMOTION_TO_CONTROL[label].to_numpy()
        weighted += emotion_vec[i] * template

    # 正規化して [-1, +1] にクリップ
    if emotion_vec.sum() > 0:
        weighted /= emotion_vec.sum()
    weighted = np.clip(weighted, -1.0, 1.0)

    return ControlVector.from_numpy(weighted)
```

---

## 10. 依存パッケージの追加

```toml
# pyproject.toml [project.dependencies] に追加

dependencies = [
    # ... 既存 ...
    "httpx>=0.27.0",              # 非同期HTTPクライアント (VOICEVOX API)
    "pyarrow>=15.0.0",            # Parquet I/O
    "scipy>=1.12.0",              # Latin Hypercube Sampling (scipy.stats.qmc)
    "soundfile>=0.12.0",          # WAV読み書き（品質検証用）
]
```

**scipy について**: LHSに`scipy.stats.qmc.LatinHypercube`を使用。既にscikit-learnが依存関係にあるため、
scipyのバージョン制約は互換性の問題を起こしにくい。

---

## 11. Parquetスキーマ (統合版)

Task #2のスキーマを基に、ControlVectorの列名を`constants.py`のCONTROL_PARAM_NAMESと整合させる。

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
audio_path:            string (output_dir相対)
audio_duration_sec:    float32
audio_file_size_bytes: int32
sample_rate:           int32
style_id:              int32
vv_speedScale:         float32
vv_pitchScale:         float32
vv_intonationScale:    float32
vv_volumeScale:        float32
vv_prePhonemeLength:   float32
vv_postPhonemeLength:  float32
vv_pauseLengthScale:   float32
is_valid:              bool
generation_timestamp:  string (ISO 8601)
```

**列名規則:**
- `emotion_*`: Phase 0感情エンコーダ出力 (8列)
- `ctrl_*`: 制御空間パラメータ (5列, [-1, +1])
- `vv_*`: VOICEVOX AudioQueryに適用された実値 (7列)

---

## 12. エラーハンドリング統合ポリシー

### 12.1 レイヤー別責務

| レイヤー | 例外処理の責務 |
|---|---|
| `tts/voicevox_client.py` | HTTP通信エラーをVoicevoxError系に変換。リトライ。4xxは即座にraise |
| `tts/adapter.py` | ControlVectorのバリデーション (frozen dataclass __post_init__) |
| `generation/pipeline.py` | 個別タスクの失敗をキャッチしてGenerationResultに記録。パイプライン全体は停止しない |
| `generation/validator.py` | WAVファイルの品質チェック。is_validフラグで判定、例外は送出しない |
| `cli.py` | ユーザー向けエラーメッセージ表示。VoicevoxConnectionErrorはEngine未起動の旨を表示 |

### 12.2 致命的エラー (パイプライン停止)

- VOICEVOX Engine接続不可（リトライ全失敗後）
- Phase 0チェックポイント読み込み失敗
- 出力ディレクトリ書き込み不可
- 設定ファイルの解析失敗

### 12.3 非致命的エラー (スキップして継続)

- 個別テキストの合成失敗 → GenerationResult.error に記録
- 個別WAVの品質チェック失敗 → is_valid=False
- 失敗率が閾値(5%)を超えた場合 → 警告ログ出力、パイプラインは継続

---

## 13. スケール見積もり (統合版)

| シナリオ | テキスト | パラメータ/テキスト | 総サンプル | 推定時間 (4並列) | WAVストレージ | Parquet |
|---------|---------|-------------------|-----------|-----------------|-------------|---------|
| 開発・テスト | 10 | 32 (LHS) | 320 | ~40秒 | ~38 MB | ~50 KB |
| **標準** | **200** | **128 (LHS)** | **25,600** | **~53分** | **~3.0 GB** | **~3 MB** |
| 大規模 | 500 | 256 (LHS) | 128,000 | ~4.4時間 | ~15 GB | ~15 MB |

---

## 14. テスト戦略

### 14.1 ユニットテスト

| モジュール | テスト対象 |
|---|---|
| `tts/types.py` | ControlVectorバリデーション、numpy変換の往復、AudioQuery.to_dict() |
| `tts/adapter.py` | _lerp計算、VoicevoxAdapter.apply()の出力範囲検証 |
| `generation/grid.py` | LHSの次元・範囲・再現性、full_gridの組み合わせ数 |
| `generation/text_selector.py` | 感情分布の均衡、テキスト数の正確性 |
| `generation/validator.py` | 有効/無効WAVの判定 |
| `generation/dataset.py` | Parquet書き出し・読み込みの往復 |

### 14.2 統合テスト (VOICEVOX Engine必須)

| テストケース | 内容 |
|---|---|
| health_check | VOICEVOX起動確認 |
| speakers | 話者一覧取得 + style_type フィルタ |
| audio_query + synthesis | テキストからWAV生成 end-to-end |
| adapter integration | ControlVector → AudioQuery変更 → synthesis → WAV検証 |
| pipeline mini-run | 2テキスト x 4パラメータの小規模パイプライン実行 |

### 14.3 テストフィクスチャ

```python
# tests/conftest.py
import pytest

@pytest.fixture
def voicevox_available():
    """VOICEVOX Engine稼働チェック。未起動ならスキップ。"""
    import httpx
    try:
        resp = httpx.get("http://127.0.0.1:50021/version", timeout=3.0)
        resp.raise_for_status()
    except Exception:
        pytest.skip("VOICEVOX Engine not available")

@pytest.fixture
def neutral_control():
    return ControlVector()  # 全て0.0

@pytest.fixture
def sample_audio_query():
    """最小限のAudioQueryフィクスチャ。"""
    ...
```
