# Phase 1: VOICEVOX APIクライアント & TTSアダプタ層 設計文書

## 1. 概要

Phase 1では、Phase 0で訓練されたテキスト感情エンコーダの出力（8D感情ベクトル）を音声合成パラメータに変換し、VOICEVOX Engine経由で感情的な音声を生成する。本文書は、VOICEVOX REST APIクライアントとTTSアダプタ層の詳細設計を定義する。

### 対象バージョン

- VOICEVOX Engine: v0.25.1
- API: REST (http://localhost:50021)

---

## 2. クラス図

```
                         +------------------------+
                         |    <<abstract>>         |
                         |    TTSAdapter           |
                         +------------------------+
                         | + apply(query, ctrl)    |
                         | + parameter_ranges()    |
                         +------------------------+
                                    ^
                                    |
                         +------------------------+
                         |  VoicevoxAdapter        |
                         +------------------------+
                         | - _mappings: dict       |
                         | + apply(query, ctrl)    |
                         | + parameter_ranges()    |
                         +------------------------+
                                    |
                                    | uses
                                    v
+-------------------+    +------------------------+    +-------------------+
| ControlVector     |    |  VoicevoxClient        |    | SpeakerInfo       |
+-------------------+    +------------------------+    +-------------------+
| pitch_shift: f32  |    | - _client: AsyncClient |    | speaker_id: int   |
| pitch_range: f32  |    | - _base_url: str       |    | name: str         |
| speed: f32        |    | - _timeout: Timeout    |    | style_name: str   |
| energy: f32       |    +------------------------+    | style_type: str   |
| pause_weight: f32 |    | + health_check()       |    +-------------------+
+-------------------+    | + speakers()           |
                         | + audio_query(text,     |
                         |     speaker_id)         |
                         | + synthesis(query,      |
                         |     speaker_id) -> bytes|
                         | + initialize_speaker(   |
                         |     speaker_id)         |
                         | + close()               |
                         +------------------------+
                                    |
                                    | owns
                                    v
                         +------------------------+
                         | AudioQuery (dataclass)  |
                         +------------------------+
                         | accent_phrases: list    |
                         | speedScale: float       |
                         | pitchScale: float       |
                         | intonationScale: float  |
                         | volumeScale: float      |
                         | prePhonemeLength: float  |
                         | postPhonemeLength: float |
                         | pauseLength: float|None |
                         | pauseLengthScale: float |
                         | outputSamplingRate: int |
                         | outputStereo: bool      |
                         +------------------------+
```

---

## 3. データフロー

```
テキスト入力
    |
    v
EmotionEncoder.encode(text)          # Phase 0 (既存)
    |
    v
8D感情ベクトル [joy, sadness, anticipation, surprise, anger, fear, disgust, trust]
    |
    v
EmotionToControlMapper.map(emotion_vector) -> ControlVector   # Phase 1 (別タスク: 学習済みマッパー)
    |
    v
ControlVector(pitch_shift, pitch_range, speed, energy, pause_weight)  # 各値: [-1.0, +1.0]
    |
    +---> VoicevoxClient.audio_query(text, speaker_id)
    |         |
    |         v
    |     AudioQuery (VOICEVOXデフォルト値)
    |         |
    +---> VoicevoxAdapter.apply(audio_query, control_vector)
              |
              v
          AudioQuery (感情パラメータ適用済み)
              |
              v
          VoicevoxClient.synthesis(audio_query, speaker_id)
              |
              v
          bytes (WAV音声データ)
```

---

## 4. インターフェース定義

### 4.1 ControlVector

制御空間の5次元ベクトル。全パラメータの正規化範囲は `[-1.0, +1.0]`。

```python
from dataclasses import dataclass

@dataclass(frozen=True, slots=True)
class ControlVector:
    """TTS制御空間の5次元ベクトル。

    各値は [-1.0, +1.0] に正規化されている。
    0.0 がニュートラル（デフォルト）を表す。
    """
    pitch_shift: float = 0.0
    pitch_range: float = 0.0
    speed: float = 0.0
    energy: float = 0.0
    pause_weight: float = 0.0

    def __post_init__(self) -> None:
        for field_name in ("pitch_shift", "pitch_range", "speed", "energy", "pause_weight"):
            value = getattr(self, field_name)
            if not (-1.0 <= value <= 1.0):
                msg = f"{field_name} must be in [-1.0, 1.0], got {value}"
                raise ValueError(msg)
```

### 4.2 AudioQuery / Mora / AccentPhrase (型定義)

VOICEVOX APIレスポンスに対応するデータクラス。

```python
from dataclasses import dataclass, field

@dataclass(slots=True)
class Mora:
    text: str
    vowel: str
    vowel_length: float
    pitch: float
    consonant: str | None = None
    consonant_length: float | None = None

@dataclass(slots=True)
class AccentPhrase:
    moras: list[Mora]
    accent: int
    pause_mora: Mora | None = None
    is_interrogative: bool = False

@dataclass(slots=True)
class AudioQuery:
    accent_phrases: list[AccentPhrase]
    speedScale: float
    pitchScale: float
    intonationScale: float
    volumeScale: float
    prePhonemeLength: float
    postPhonemeLength: float
    outputSamplingRate: int
    outputStereo: bool
    pauseLength: float | None = None
    pauseLengthScale: float = 1.0
    kana: str | None = None

    def to_dict(self) -> dict:
        """VOICEVOX API送信用の辞書に変換する。"""
        ...
```

### 4.3 SpeakerInfo

```python
@dataclass(frozen=True, slots=True)
class SpeakerStyle:
    name: str
    id: int
    style_type: str = "talk"

@dataclass(frozen=True, slots=True)
class SpeakerInfo:
    name: str
    speaker_uuid: str
    styles: list[SpeakerStyle]
    version: str
```

### 4.4 VoicevoxClient

```python
import httpx

class VoicevoxClient:
    """VOICEVOX Engine REST APIの非同期クライアント。"""

    def __init__(
        self,
        base_url: str = "http://localhost:50021",
        timeout: float = 30.0,
        max_retries: int = 3,
        retry_delay: float = 0.5,
    ) -> None: ...

    async def health_check(self) -> bool:
        """GET /version でエンジンの疎通を確認する。

        Returns:
            True: エンジンが応答した
            False: 接続失敗またはタイムアウト
        """
        ...

    async def speakers(self) -> list[SpeakerInfo]:
        """GET /speakers で利用可能なキャラクター一覧を取得する。

        Returns:
            SpeakerInfoのリスト
        Raises:
            VoicevoxAPIError: API呼び出し失敗時
        """
        ...

    async def audio_query(
        self,
        text: str,
        speaker_id: int,
    ) -> AudioQuery:
        """POST /audio_query でテキストから音声合成クエリを生成する。

        Args:
            text: 合成対象のテキスト
            speaker_id: スピーカースタイルID
        Returns:
            AudioQuery
        Raises:
            VoicevoxAPIError: API呼び出し失敗時
        """
        ...

    async def synthesis(
        self,
        audio_query: AudioQuery,
        speaker_id: int,
    ) -> bytes:
        """POST /synthesis でAudioQueryから音声を合成する。

        Args:
            audio_query: 音声合成クエリ
            speaker_id: スピーカースタイルID
        Returns:
            WAV形式の音声バイナリ
        Raises:
            VoicevoxAPIError: API呼び出し失敗時
        """
        ...

    async def initialize_speaker(
        self,
        speaker_id: int,
        skip_reinit: bool = True,
    ) -> None:
        """POST /initialize_speaker でスタイルを事前初期化する。

        Args:
            speaker_id: スピーカースタイルID
            skip_reinit: 初期化済みスタイルの再初期化をスキップ
        Raises:
            VoicevoxAPIError: API呼び出し失敗時
        """
        ...

    async def close(self) -> None:
        """HTTPクライアントを閉じる。"""
        ...

    async def __aenter__(self) -> "VoicevoxClient": ...
    async def __aexit__(self, *args) -> None: ...
```

### 4.5 TTSAdapter (抽象基底クラス)

```python
from abc import ABC, abstractmethod

class TTSAdapter(ABC):
    """制御空間ベクトルをTTSエンジン固有のパラメータに変換する抽象アダプタ。

    将来のTTSエンジン差し替え（COEIROINK, Style-Bert-VITS2等）に対応するための
    共通インターフェースを定義する。
    """

    @abstractmethod
    def apply(self, audio_query: AudioQuery, control: ControlVector) -> AudioQuery:
        """制御ベクトルをAudioQueryに適用し、新しいAudioQueryを返す。

        元のAudioQueryは変更しない（immutableパターン）。

        Args:
            audio_query: VOICEVOX APIから取得したベースクエリ
            control: 制御空間の5Dベクトル
        Returns:
            パラメータ適用済みのAudioQuery
        """
        ...

    @abstractmethod
    def parameter_ranges(self) -> dict[str, tuple[float, float]]:
        """このアダプタが出力する各TTSパラメータの有効範囲を返す。

        Returns:
            パラメータ名 -> (min, max) の辞書
        """
        ...
```

### 4.6 VoicevoxAdapter

```python
from dataclasses import replace

class VoicevoxAdapter(TTSAdapter):
    """VOICEVOX Engine用のTTSアダプタ。

    制御空間(5D) -> VOICEVOXパラメータへの線形マッピングを行う。
    """

    def apply(self, audio_query: AudioQuery, control: ControlVector) -> AudioQuery:
        """制御ベクトルをVOICEVOXパラメータに変換し適用する。

        マッピング:
            speedScale        = 1.0 + control.speed * 0.5        -> [0.5, 1.5]
            pitchScale        = control.pitch_shift * 0.15       -> [-0.15, +0.15]
            intonationScale   = 1.0 + control.pitch_range * 0.5  -> [0.5, 1.5]
            volumeScale       = 1.0 + control.energy * 0.5       -> [0.5, 1.5]
            prePhonemeLength  = 0.1 + control.pause_weight * 0.1 -> [0.0, 0.2]
            postPhonemeLength = 0.1 + control.pause_weight * 0.1 -> [0.0, 0.2]
            pauseLengthScale  = 1.0 + control.pause_weight * 0.5 -> [0.5, 1.5]
        """
        ...

    def parameter_ranges(self) -> dict[str, tuple[float, float]]:
        return {
            "speedScale": (0.5, 1.5),
            "pitchScale": (-0.15, 0.15),
            "intonationScale": (0.5, 1.5),
            "volumeScale": (0.5, 1.5),
            "prePhonemeLength": (0.0, 0.2),
            "postPhonemeLength": (0.0, 0.2),
            "pauseLengthScale": (0.5, 1.5),
        }
```

### 4.7 例外クラス

```python
class VoicevoxError(Exception):
    """VOICEVOX関連エラーの基底クラス。"""

class VoicevoxConnectionError(VoicevoxError):
    """VOICEVOX Engineへの接続失敗。"""

class VoicevoxAPIError(VoicevoxError):
    """VOICEVOX APIがエラーレスポンスを返した。"""
    def __init__(self, status_code: int, detail: str) -> None:
        self.status_code = status_code
        self.detail = detail
        super().__init__(f"VOICEVOX API error {status_code}: {detail}")

class VoicevoxTimeoutError(VoicevoxError):
    """VOICEVOX APIリクエストがタイムアウトした。"""
```

---

## 5. 制御空間 -> VOICEVOXパラメータ マッピング詳細

### 5.1 マッピングテーブル

| 制御パラメータ | VOICEVOXプロパティ | 変換式 | 出力範囲 | 根拠 |
|---|---|---|---|---|
| speed | speedScale | `1.0 + speed * 0.5` | [0.5, 1.5] | 1.0がニュートラル、最大50%の速度変動 |
| pitch_shift | pitchScale | `pitch_shift * 0.15` | [-0.15, +0.15] | pitchScaleは微調整が主用途のため範囲を抑制 |
| pitch_range | intonationScale | `1.0 + pitch_range * 0.5` | [0.5, 1.5] | 抑揚の倍率。1.0でデフォルト |
| energy | volumeScale | `1.0 + energy * 0.5` | [0.5, 1.5] | 音量の倍率。1.0でデフォルト |
| pause_weight | prePhonemeLength | `0.1 + pause_weight * 0.1` | [0.0, 0.2] | 音声前の無音時間（秒） |
| pause_weight | postPhonemeLength | `0.1 + pause_weight * 0.1` | [0.0, 0.2] | 音声後の無音時間（秒） |
| pause_weight | pauseLengthScale | `1.0 + pause_weight * 0.5` | [0.5, 1.5] | 句読点の無音時間の倍率 |

### 5.2 pause_weightの変換先に関する設計判断

VOICEVOX APIには句読点の無音時間を制御するプロパティが複数存在する:

- `pauseLength` (number|null): 句読点の無音時間を秒単位で直接指定。nullの場合はデフォルト挙動。
- `pauseLengthScale` (number, default=1): 句読点の無音時間の倍率。

**判断**: `pauseLength` はnullがデフォルトであり、直接指定すると元のプロソディが失われる。`pauseLengthScale` は倍率として相対的に調整できるため、`pauseLength` はnullのまま（APIデフォルト任せ）とし、`pauseLengthScale` を `pause_weight` から制御する。`prePhonemeLength` / `postPhonemeLength` も併せて調整することで、文全体のペーシング（間の取り方）を統合的に制御する。

### 5.3 マッピング範囲の設計根拠

全マッピングで控えめな範囲を採用している理由:

1. **音質の安全性**: 極端なパラメータはVOICEVOXの音声品質を著しく劣化させる。特にpitchScaleは大きな値でロボット的な音声になる。
2. **段階的拡張**: Phase 1初期は保守的な範囲でスタートし、聴取評価を経て必要に応じて拡大する。マッピング式は `VoicevoxAdapter` に閉じているため、範囲調整は容易。
3. **線形マッピング**: 非線形マッピング（シグモイド、区分線形等）は聴取データが蓄積されてから検討する。現段階で不必要な複雑性を避ける。

---

## 6. VoicevoxClient 実装詳細

### 6.1 HTTPクライアント構成

```python
import httpx

# クライアント初期化
self._client = httpx.AsyncClient(
    base_url=base_url,
    timeout=httpx.Timeout(timeout, connect=5.0),
    limits=httpx.Limits(max_connections=10, max_keepalive_connections=5),
)
```

**httpx選定理由**:
- async/await ネイティブ対応
- タイムアウト・リトライの粒度の高い制御
- 型アノテーション完備
- EmotionBridgeの他の依存関係（transformers等）との衝突が少ない

### 6.2 リトライ戦略

```python
async def _request_with_retry(
    self,
    method: str,
    url: str,
    **kwargs,
) -> httpx.Response:
    last_exception: Exception | None = None
    for attempt in range(self._max_retries):
        try:
            response = await self._client.request(method, url, **kwargs)
            response.raise_for_status()
            return response
        except httpx.TimeoutException as e:
            last_exception = VoicevoxTimeoutError(str(e))
        except httpx.ConnectError as e:
            last_exception = VoicevoxConnectionError(str(e))
        except httpx.HTTPStatusError as e:
            # 4xx はリトライしない（クライアントエラー）
            if 400 <= e.response.status_code < 500:
                raise VoicevoxAPIError(e.response.status_code, e.response.text) from e
            last_exception = VoicevoxAPIError(e.response.status_code, e.response.text)

        if attempt < self._max_retries - 1:
            await asyncio.sleep(self._retry_delay * (2 ** attempt))

    raise last_exception  # type: ignore[misc]
```

**リトライポリシー**:
- 対象: タイムアウト、接続エラー、5xxサーバーエラー
- 非対象: 4xxクライアントエラー（即座に例外送出）
- バックオフ: 指数バックオフ (`delay * 2^attempt`)
- デフォルト: 最大3回、初期遅延0.5秒

### 6.3 コンテキストマネージャ

```python
async def __aenter__(self) -> "VoicevoxClient":
    return self

async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
    await self.close()
```

使用例:
```python
async with VoicevoxClient() as client:
    query = await client.audio_query("こんにちは", speaker_id=0)
    wav = await client.synthesis(query, speaker_id=0)
```

---

## 7. スピーカー管理

### 7.1 スピーカー一覧取得

```python
async def speakers(self) -> list[SpeakerInfo]:
    response = await self._request_with_retry("GET", "/speakers")
    return [
        SpeakerInfo(
            name=s["name"],
            speaker_uuid=s["speaker_uuid"],
            styles=[
                SpeakerStyle(
                    name=st["name"],
                    id=st["id"],
                    style_type=st.get("type", "talk"),
                )
                for st in s["styles"]
            ],
            version=s["version"],
        )
        for s in response.json()
    ]
```

### 7.2 スピーカー選定戦略

- `speaker_id` はVOICEVOXの「スタイルID」に対応する（キャラクターIDではない）
- デフォルト値: 設定ファイルで指定。未指定時はID `0`（通常は「四国めたん - ノーマル」）
- `style_type == "talk"` のスタイルのみが音声合成の対象。`singing_teacher`, `frame_decode`, `sing` は除外する
- 初回使用時に `initialize_speaker` を呼び出し、モデルのロードを済ませておくことを推奨

---

## 8. 設定システム拡張

既存の `config.py` に Phase 1 用の設定を追加する。

```python
@dataclass(slots=True)
class VoicevoxConfig:
    """VOICEVOX Engine接続設定。"""
    base_url: str = "http://localhost:50021"
    timeout: float = 30.0
    max_retries: int = 3
    retry_delay: float = 0.5
    default_speaker_id: int = 0
    output_sampling_rate: int = 24000
    output_stereo: bool = False

@dataclass(slots=True)
class TTSConfig:
    """TTS全般の設定。"""
    adapter: str = "voicevox"  # 将来のTTSエンジン差し替え用
    voicevox: VoicevoxConfig = field(default_factory=VoicevoxConfig)
    output_dir: str = "artifacts/phase1/audio"

@dataclass(slots=True)
class Phase1Config:
    """Phase 1全体の設定。"""
    tts: TTSConfig = field(default_factory=TTSConfig)
```

---

## 9. ファイル配置

```
emotionbridge/
  tts/
    __init__.py
    types.py              # ControlVector, AudioQuery, Mora, AccentPhrase, SpeakerInfo, SpeakerStyle
    exceptions.py         # VoicevoxError, VoicevoxConnectionError, VoicevoxAPIError, VoicevoxTimeoutError
    voicevox_client.py    # VoicevoxClient
    adapter.py            # TTSAdapter (ABC), VoicevoxAdapter
```

---

## 10. エラーハンドリングポリシー

### 10.1 例外階層

```
VoicevoxError (基底)
  ├── VoicevoxConnectionError    # Engineへの接続失敗
  ├── VoicevoxAPIError           # APIエラーレスポンス (status_code, detail)
  └── VoicevoxTimeoutError       # リクエストタイムアウト
```

### 10.2 ポリシー

| 状況 | 対応 |
|---|---|
| Engine未起動 / 接続不可 | `VoicevoxConnectionError`。リトライ後も失敗すれば例外送出。呼び出し元でフォールバック判断 |
| 不正なspeaker_id | `VoicevoxAPIError(422, ...)`。リトライなしで即座に送出 |
| 不正なテキスト入力 | `VoicevoxAPIError(422, ...)`。リトライなしで即座に送出 |
| synthesis タイムアウト | `VoicevoxTimeoutError`。長文テキストの場合は分割を呼び出し元に委ねる |
| 5xxサーバーエラー | リトライ対象。指数バックオフ後にも失敗すれば `VoicevoxAPIError` |
| ControlVector 範囲外 | `ControlVector.__post_init__` で `ValueError`。API呼び出し前にバリデーション |

### 10.3 ログ出力

- リトライ時: `logger.warning(f"VOICEVOX API retry {attempt}/{max_retries}: {error}")`
- 接続失敗: `logger.error(f"VOICEVOX Engine unreachable: {base_url}")`
- 成功: `logger.debug(f"VOICEVOX synthesis completed: {len(wav_bytes)} bytes")`

---

## 11. 設計判断の根拠まとめ

| 判断 | 根拠 |
|---|---|
| httpx採用（aiohttpではなく） | 型アノテーション完備、タイムアウトのきめ細かい制御、asyncio対応。aiohttp は低レベルAPI寄りで設定が煩雑 |
| dataclass（Pydanticではなく） | 既存Phase 0コードとの一貫性。Phase 0は全て dataclass で構成されている。シリアライゼーション要件がVOICEVOX APIの dict 変換のみのため Pydantic は過剰 |
| 抽象TTSAdapter | Two-space設計の核心: 共有感情潜在空間は固定、TTSアダプタのみ差し替え可能。COEIROINK, Style-Bert-VITS2等への将来対応を保証するため |
| AudioQueryの浅いコピー(replace)パターン | 元クエリの accent_phrases（VOICEVOXが生成したプロソディ情報）を保持しつつ、スカラーパラメータのみ上書き。元データの破壊を防ぐ |
| pauseLengthをnullのまま保持 | pauseLength直接指定はVOICEVOXの内部プロソディ推定を上書きしてしまう。pauseLengthScaleによる倍率調整の方が自然な結果を得られる |
| pause_weight -> 3パラメータ同時制御 | prePhonemeLength, postPhonemeLength, pauseLengthScale を単一の pause_weight で統合制御。制御空間の次元数を抑えつつ、「間の取り方」を直感的に操作可能にする |
| 控えめなマッピング範囲 | 音声品質の劣化を防ぐ安全策。聴取評価データの蓄積後に段階的に拡張する方針 |
| speaker_id（style IDベース）の管理 | VOICEVOXではキャラクター内に複数スタイル（ノーマル、あまあま等）があり、style_id で直接指定する方がAPI呼び出しがシンプル |
| 非同期API設計 | VOICEVOX synthesis はネットワークI/O + 推論処理で数百ミリ秒かかる。バッチ合成時の並行処理に備えてasync/awaitで設計 |
