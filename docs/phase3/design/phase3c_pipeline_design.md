# Phase 3c/3d 詳細設計: スタイル選択・推論パイプライン・評価

> **対象要求**: EB3-E01 (スタイル選択), EB3-F01 (推論パイプライン), EB3-G01 (評価)
> **作成日**: 2026-02-18
> **ステータス**: 初版
> **前提**: Phase 3a/3b 詳細設計完了、V-01/V-02/V-03 検証完了

---

## A. スタイル選択 (EB3-E01)

### A.1 スタイル別韻律プロファイルの構築手順

#### 概要

各 style_id のグリッドサーチ音声から eGeMAPS 韻律プロファイル（重心）を算出し、JVNV 感情プロファイルとの距離比較の基盤とする。

#### 算出手順

1. **入力データ**: `artifacts/phase3/v02/voicevox_egemaps_normalized.parquet`（style_id 内 z-score 正規化済み 88D 特徴量）
2. **スタイルごとの重心計算**:
   ```
   for style_id in target_style_ids:
       subset = normalized_df[normalized_df["style_id"] == style_id]
       centroid = subset[feature_cols].mean(axis=0)  # 88D ベクトル
       std = subset[feature_cols].std(axis=0)         # 分布情報の保持
   ```
3. **出力**: スタイル別韻律プロファイル辞書
   ```python
   style_profiles: dict[int, StyleProsodyProfile]
   # StyleProsodyProfile = {centroid: np.ndarray(88,), std: np.ndarray(88,), n_samples: int}
   ```

#### 妥当性に関する注記

グリッドサーチ音声の重心は、当該スタイルの「平均的」韻律であり、必ずしもそのスタイルの最良の代表ではない。グリッドサーチは 5D 制御パラメータ空間を均一にサンプリングしているため、重心は制御パラメータが (0,0,0,0,0) 付近の中性的な韻律に引っ張られる。

この制限は認識した上で、以下の理由から初期実装としては許容する:
- JVNV 感情プロファイルとの **相対的な距離比較** が目的であり、絶対的な韻律代表性は不要
- 全スタイルを同一手法で算出することで、スタイル間の比較が公平
- 将来的には、制御パラメータを固定（例: ゼロベクトル）した音声のみで重心を算出する改良が可能

#### 現在のデータ可用性（2026-02-18）

現行成果物では `artifacts/phase1/dataset/triplet_dataset.parquet` および
`artifacts/phase3/v02/voicevox_egemaps_normalized.parquet` の `style_id` が実質 `0` のみであり、
複数スタイル間の距離比較は未実施である。したがって、A.2 の感情→スタイルは本時点では仮説表として扱う。

#### 対象スタイル

| キャラクター | スタイル名 | style_id |
|-------------|-----------|----------|
| ずんだもん   | ノーマル   | 3        |
| ずんだもん   | あまあま   | 1        |
| ずんだもん   | ツンツン   | 7        |
| ずんだもん   | セクシー   | 5        |
| ずんだもん   | ささやき   | 22       |
| ずんだもん   | ヒソヒソ   | 38       |
| ずんだもん   | ヘロヘロ   | 75       |
| ずんだもん   | なみだめ   | 76       |
| 四国めたん   | ノーマル   | 2        |
| 四国めたん   | あまあま   | 0        |
| 四国めたん   | ツンツン   | 6        |
| 四国めたん   | セクシー   | 4        |
| 四国めたん   | ささやき   | 36       |
| 四国めたん   | ヒソヒソ   | 37       |

### A.2 感情 → スタイルマッピング表の設計

#### マッピング決定手順

1. 各 style_id の韻律プロファイル重心 `style_centroid[s]` を算出（A.1 の手順）
2. 各 JVNV 感情の韻律プロファイル重心 `jvnv_centroid[e]` を取得（V-01 で算出済み、`artifacts/phase3/v01/jvnv_egemaps_normalized.parquet` から再計算可能）
3. 各感情 e について、全スタイルとのユークリッド距離を計算:
   ```
   for emotion in JVNV_EMOTION_LABELS:
       for style_id in character_style_ids:
           distance[emotion][style_id] = euclidean(jvnv_centroid[emotion], style_centroid[style_id])
       best_style[emotion] = argmin(distance[emotion])
   ```
4. 距離が最小のスタイルをその感情のデフォルトスタイルとする

#### 注意: 正規化空間の整合性

JVNV 韻律プロファイルは話者内 z-score 正規化、VOICEVOX は style_id 内 z-score 正規化で構築されている。正規化基準が異なるため、直接のユークリッド距離比較には制限がある。

V-03 のマッチングでは、この差異にもかかわらず文献整合的なパラメータパターンを得ている（正規化後 overlap ratio 0.72）。したがって、スタイル選択においてもこの距離比較は実用上有効と判断する。

#### ずんだもん (8 スタイル) 初期仮説マッピング（未確定）

以下は提案書セクション 3.4 の直感的マッピングに基づく初期案。実データでの距離計算結果で修正する。

| 感情     | 第一候補スタイル | style_id | 根拠                                          |
|---------|----------------|----------|-----------------------------------------------|
| anger   | ツンツン        | 7        | 怒り・不満の表現に特化したスタイル                    |
| happy   | あまあま        | 1        | 喜び・親しみの表現に特化                            |
| sad     | なみだめ        | 76       | 悲しみ（強）の表現に特化                            |
| disgust | ツンツン        | 7        | 嫌悪は怒りに近い韻律パターン（V-01: anger-disgust 分離あり）|
| fear    | ヒソヒソ        | 38       | 恐怖・秘密の低強度発話                              |
| surprise| ノーマル        | 3        | 驚きは韻律的にニュートラルに近い（V-01: fear-surprise 分離弱） |

**確定手順**: 上記は仮説。複数style_idを含むv02データを再生成した後、A.1の手順で距離行列を算出しデータドリブンで確定する。距離行列の結果と上記仮説が乖離する場合は距離行列を優先し、理由を記録する。

#### 四国めたん (6 スタイル) 初期仮説マッピング（未確定）

| 感情     | 第一候補スタイル | style_id | 根拠                     |
|---------|----------------|----------|--------------------------|
| anger   | ツンツン        | 6        | 怒り表現                  |
| happy   | あまあま        | 0        | 喜び表現                  |
| sad     | ささやき        | 36       | なみだめ不在のため弱発話で代替 |
| disgust | ツンツン        | 6        | anger に準じる             |
| fear    | ヒソヒソ        | 37       | 恐怖の低強度発話           |
| surprise| ノーマル        | 2        | ニュートラルベース          |

### A.3 デフォルトスタイルとフォールバック設計

#### デフォルトスタイル

感情分類の確信度が低い場合（EB3-F01-004）のフォールバック先:

| キャラクター | デフォルトスタイル | style_id |
|-------------|------------------|----------|
| ずんだもん   | ノーマル          | 3        |
| 四国めたん   | ノーマル          | 2        |

#### フォールバック条件

1. **低確信度フォールバック**: `max(emotion_probs) < threshold` の場合、デフォルトスタイル + 5D ゼロベクトルで合成
2. **スタイル未対応フォールバック**: 指定キャラクターが対象スタイルを持たない場合、ノーマルスタイルを使用

運用値:
- `threshold` の初期値は **0.30** で固定して開始する
- 最終調整はPhase 3d主観評価で行う

### A.4 将来のスタイル混合への拡張ポイント

Phase 3 初期はルールベース（1感情 = 1スタイル）で実装する。将来の拡張として以下を想定:

1. **スタイル混合比率の学習**: 感情確率ベクトル → スタイル選択確率（Softmax）を学習し、確率的にスタイルを選択
2. **VOICEVOX のモーフィング API**: VOICEVOX が将来スタイル間のモーフィングをサポートした場合、混合比率をモーフィングパラメータに変換
3. **混合感情対応**: 例えば「怒り 0.6 + 悲しみ 0.4」の場合、支配的感情のスタイルを採用しつつ、5D パラメータで副次感情の韻律特性を反映

**現時点での実装**: 拡張ポイントとして `StyleSelector` クラスのインターフェースを設計し、ルールベース実装を `RuleBasedStyleSelector` として提供。将来の `LearnedStyleSelector` への差し替えを容易にする。

---

## B. 推論パイプライン (EB3-F01)

### B.1 EmotionBridgePipeline クラス設計

#### クラス構成図

```
EmotionBridgePipeline
  ├── EmotionClassifier          # Phase 0 再設計: テキスト → 6D 感情確率
    ├── DeterministicMixer         # Phase 3b: 6D 感情確率 → 5D 制御パラメータ
  ├── StyleSelector (ABC)        # EB3-E01: 感情カテゴリ → style_id
  │    └── RuleBasedStyleSelector
  ├── VoicevoxAdapter            # 既存: 5D → VOICEVOX パラメータ変換
  └── VoicevoxClient             # 既存: VOICEVOX Engine API クライアント
```

#### クラスインターフェース

```python
@dataclass(frozen=True, slots=True)
class SynthesisResult:
    """推論パイプラインの出力。"""
    audio_bytes: bytes                    # WAV 音声バイナリ
    audio_path: Path | None               # 保存先パス（保存時のみ）
    emotion_probs: dict[str, float]       # 6D 感情確率 {emotion: prob}
    dominant_emotion: str                 # argmax 感情ラベル
    control_params: dict[str, float]      # 5D 制御パラメータ
    style_id: int                         # 選択された VOICEVOX style_id
    style_name: str                       # スタイル名（表示用）
    confidence: float                     # max(emotion_probs)
    is_fallback: bool                     # 低確信度フォールバックが発動したか
    metadata: dict[str, Any]              # 追加メタデータ


class StyleSelector(ABC):
    """感情カテゴリ → VOICEVOX style_id の抽象インターフェース。"""

    @abstractmethod
    def select(
        self,
        emotion_probs: dict[str, float],
        character: str,
    ) -> tuple[int, str]:
        """感情確率から最適な style_id とスタイル名を返す。"""
        ...

    @abstractmethod
    def default_style(self, character: str) -> tuple[int, str]:
        """フォールバック用のデフォルトスタイルを返す。"""
        ...


class RuleBasedStyleSelector(StyleSelector):
    """ルールベースのスタイル選択。

    A.2 で構築した感情→スタイルマッピング表を使用。
    """

    def __init__(self, mapping_path: str | Path):
        """マッピングテーブル（JSON）をロードする。"""
        ...

    def select(
        self,
        emotion_probs: dict[str, float],
        character: str,
    ) -> tuple[int, str]:
        """argmax 感情に対応するスタイルを返す。"""
        dominant = max(emotion_probs, key=emotion_probs.get)
        return self._mapping[character][dominant]

    def default_style(self, character: str) -> tuple[int, str]:
        return self._defaults[character]


class EmotionBridgePipeline:
    """テキスト → 感情音声合成の統合パイプライン。"""

    def __init__(
        self,
        classifier: EmotionClassifier,
        generator: DeterministicMixer,
        style_selector: StyleSelector,
        voicevox_client: VoicevoxClient,
        adapter: VoicevoxAdapter,
        *,
        character: str = "zundamon",
        fallback_threshold: float = 0.3,
    ) -> None: ...

    async def synthesize(self, text: str) -> SynthesisResult:
        """単一テキストの感情音声合成。"""
        ...

    async def synthesize_batch(
        self,
        texts: list[str],
        *,
        max_concurrent: int = 4,
    ) -> list[SynthesisResult]:
        """複数テキストの並行合成。"""
        ...

    async def synthesize_to_file(
        self,
        text: str,
        output_path: str | Path,
    ) -> SynthesisResult:
        """合成結果をファイルに保存。"""
        ...
```

### B.2 SynthesisResult データクラス設計

```python
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

@dataclass(frozen=True, slots=True)
class SynthesisResult:
    # === 主要出力 ===
    audio_bytes: bytes                    # WAV 音声バイナリ
    audio_path: Path | None = None        # 保存先（synthesize_to_file 使用時）

    # === 感情分類結果 ===
    emotion_probs: dict[str, float] = field(default_factory=dict)
    # {anger: 0.05, disgust: 0.02, fear: 0.01, happy: 0.80, sad: 0.10, surprise: 0.02}
    dominant_emotion: str = ""            # argmax で決定される感情ラベル
    confidence: float = 0.0              # max(emotion_probs)

    # === 制御パラメータ ===
    control_params: dict[str, float] = field(default_factory=dict)
    # {pitch_shift: 0.22, pitch_range: 0.19, speed: 0.26, energy: -0.32, pause_weight: 0.16}

    # === スタイル選択結果 ===
    style_id: int = 0
    style_name: str = ""
    character: str = ""

    # === フォールバック情報 ===
    is_fallback: bool = False             # 低確信度フォールバック発動の有無

    # === メタデータ ===
    metadata: dict[str, Any] = field(default_factory=dict)
    # {input_text, processing_time_ms, voicevox_query_params, ...}
```

### B.3 パイプライン処理フロー

```
synthesize(text: str) の処理フロー:

1. テキスト感情分類
   emotion_probs = classifier.classify(text)  # dict[str, float], 合計 1.0
   dominant_emotion = argmax(emotion_probs)
   confidence = max(emotion_probs.values())

2. 低確信度フォールバック判定
   if confidence < fallback_threshold:
       style_id, style_name = style_selector.default_style(character)
       control = ControlVector()  # 全ゼロ (0,0,0,0,0)
       is_fallback = True
   else:
       # 通常処理
       goto step 3

3. パラメータ生成
   control = generator.predict(emotion_probs)  # 6D → 5D

4. スタイル選択
   style_id, style_name = style_selector.select(emotion_probs, character)

5. VOICEVOX 音声合成
   audio_query = await voicevox_client.audio_query(text, style_id)
   modified_query = adapter.apply(audio_query, control)
   audio_bytes = await voicevox_client.synthesis(modified_query, style_id)

6. 結果構築
   return SynthesisResult(
       audio_bytes=audio_bytes,
       emotion_probs=emotion_probs,
       dominant_emotion=dominant_emotion,
       control_params=control.to_dict(),
       style_id=style_id,
       style_name=style_name,
       confidence=confidence,
       is_fallback=is_fallback,
       metadata={...},
   )
```

### B.4 初期化時の設定・依存注入

#### ファクトリ関数

コンポーネントの初期化を一括で行うファクトリ関数を提供する:

```python
async def create_pipeline(
    *,
    classifier_checkpoint: str | Path = "artifacts/phase3/checkpoints/classifier_best.pt",
    generator_model_dir: str | Path = "artifacts/phase3/checkpoints/generator_best",
    style_mapping: str | Path = "artifacts/phase3/style_mapping.json",
    voicevox_url: str = "http://localhost:50021",
    character: str = "zundamon",
    fallback_threshold: float = 0.3,
    device: str = "cuda",
) -> EmotionBridgePipeline:
    """パイプラインの全コンポーネントを初期化して返す。

    Raises:
        FileNotFoundError: チェックポイントまたはマッピングファイルが見つからない場合
        VoicevoxConnectionError: VOICEVOX Engine に接続できない場合
    """
    # 1. 感情分類器のロード
    classifier = EmotionClassifier(str(classifier_checkpoint), device=device)

    # 2. パラメータ生成器のロード
    generator = load_generator_model_dir(str(generator_model_dir), device=device)

    # 3. スタイル選択器のロード
    style_selector = RuleBasedStyleSelector(style_mapping)

    # 4. VOICEVOX クライアントの初期化と疎通確認
    client = VoicevoxClient(base_url=voicevox_url)
    if not await client.health_check():
        await client.close()
        raise VoicevoxConnectionError(
            f"VOICEVOX Engine に接続できません: {voicevox_url}"
        )

    # 5. アダプタ（既存の ControlSpaceConfig デフォルト使用）
    adapter = VoicevoxAdapter()

    return EmotionBridgePipeline(
        classifier=classifier,
        generator=generator,
        style_selector=style_selector,
        voicevox_client=client,
        adapter=adapter,
        character=character,
        fallback_threshold=fallback_threshold,
    )
```

#### 必要なアーティファクト一覧

| アーティファクト | パス | 生成元 |
|---------------|------|--------|
| 感情分類器チェックポイント | `artifacts/phase3/checkpoints/classifier_best.pt` | Phase 3a (EB3-C01) |
| パラメータ生成器モデルディレクトリ | `artifacts/phase3/checkpoints/generator_best/` | Phase 3b (EB3-D01) |
| スタイルマッピング JSON | `artifacts/phase3/style_mapping.json` | Phase 3c (EB3-E01) |
| VOICEVOX Engine | `http://localhost:50021` | 外部依存（ユーザーが起動） |

### B.5 CLI 統合

#### 新コマンド: `bridge`

```python
# cli.py への追加

bridge_parser = subparsers.add_parser(
    "bridge",
    help="Synthesize emotional speech using the full EmotionBridge pipeline",
)
bridge_parser.add_argument("--text", required=True, help="Input text")
bridge_parser.add_argument(
    "--output", default="output.wav", help="Output WAV file path"
)
bridge_parser.add_argument(
    "--character",
    choices=["zundamon", "shikoku_metan"],
    default="zundamon",
    help="VOICEVOX character",
)
bridge_parser.add_argument(
    "--classifier-checkpoint",
    default="artifacts/phase3/checkpoints/classifier_best.pt",
    help="Path to emotion classifier checkpoint",
)
bridge_parser.add_argument(
    "--generator-model-dir",
    default="artifacts/phase3/checkpoints/generator_best",
    help="Path to parameter generator model directory",
)
bridge_parser.add_argument(
    "--style-mapping",
    default="artifacts/phase3/style_mapping.json",
    help="Path to style mapping JSON",
)
bridge_parser.add_argument(
    "--voicevox-url",
    default="http://localhost:50021",
    help="VOICEVOX Engine URL",
)
bridge_parser.add_argument(
    "--fallback-threshold",
    type=float,
    default=0.3,
    help="Confidence threshold for fallback to neutral",
)
bridge_parser.add_argument("--device", default="cuda", help="cuda or cpu")
```

#### 使用例

```bash
# 基本的な使用
python main.py bridge --text "今日は楽しかった！" --output happy.wav

# キャラクター指定
python main.py bridge --text "許さない" --character zundamon --output angry.wav

# CPU で実行
python main.py bridge --text "怖い..." --device cpu --output fear.wav
```

#### 出力フォーマット

```
感情分析結果:
  anger: 0.050  disgust: 0.020  fear: 0.010
  happy: 0.800  sad: 0.100     surprise: 0.020
  支配感情: happy (確信度: 0.800)

スタイル選択: あまあま (ID: 1)
制御パラメータ:
  pitch_shift: +0.220  pitch_range: +0.190  speed: +0.260
  energy: -0.320       pause_weight: +0.160

音声を保存しました: happy.wav (42,368 bytes)
```

### B.6 低確信度フォールバック

#### 判定基準

```python
confidence = max(emotion_probs.values())
is_fallback = confidence < fallback_threshold
```

#### 閾値の選定

| 閾値候補 | 特性 |
|---------|------|
| 0.25 | 6 感情均等分布 (1/6 = 0.167) より少し上。ほぼフォールバックしない |
| 0.30 | 推奨初期値。弱い感情でもスタイル選択を試みる |
| 0.40 | 保守的。明確な感情がない限りニュートラルを維持 |

**決定**: 初期運用値は `0.30` を採用する。Phase 3d の主観評価で最終チューニングする。

#### フォールバック時の動作

```
if is_fallback:
    style_id = default_style_id  # ノーマル (ずんだもん: 3, 四国めたん: 2)
    control = ControlVector()     # (0, 0, 0, 0, 0) = VOICEVOX デフォルト韻律
```

### B.7 エラーハンドリング

#### VOICEVOX 未起動時

```python
# create_pipeline() 内で疎通確認
if not await client.health_check():
    raise VoicevoxConnectionError(
        f"VOICEVOX Engine に接続できません: {voicevox_url}\n"
        "VOICEVOX Engine を起動してから再実行してください。\n"
        "起動方法: https://voicevox.hiroshiba.jp/"
    )
```

CLI レベルでは `SystemExit` に変換し、ユーザーフレンドリーなメッセージを表示:

```python
try:
    pipeline = await create_pipeline(...)
except VoicevoxConnectionError as e:
    raise SystemExit(f"エラー: {e}") from e
```

#### モデルファイル欠損時

```python
# EmotionClassifier.__init__() / DeterministicMixer.load() で検出
if not checkpoint_path.exists():
    raise FileNotFoundError(
        f"チェックポイントが見つかりません: {checkpoint_path}\n"
        "Phase 3a/3b の学習を先に実行してください。"
    )
```

#### スタイルマッピング欠損時

```python
# RuleBasedStyleSelector.__init__() で検出
if not mapping_path.exists():
    raise FileNotFoundError(
        f"スタイルマッピングが見つかりません: {mapping_path}\n"
        "Phase 3c のスタイルプロファイル構築を先に実行してください。"
    )
```

#### VOICEVOX API エラー（合成時）

既存の `VoicevoxClient` のリトライ機構を活用。リトライ上限超過時は `VoicevoxAPIError` が送出され、パイプライン呼び出し元で処理する。バッチ合成時は個別の失敗を記録し、成功分のみ返す:

```python
async def synthesize_batch(self, texts, *, max_concurrent=4):
    results = []
    errors = []
    semaphore = asyncio.Semaphore(max_concurrent)
    for text in texts:
        async with semaphore:
            try:
                result = await self.synthesize(text)
                results.append(result)
            except VoicevoxError as e:
                errors.append((text, e))
                logger.warning("合成失敗: %s: %s", text[:30], e)
    if errors:
        logger.warning("バッチ合成: %d/%d 件失敗", len(errors), len(texts))
    return results
```

### B.8 バッチ処理設計

`synthesize_batch` は `asyncio.Semaphore` で同時リクエスト数を制限する。これは既存の Phase 1 `GenerationPipeline` と同じパターン。

```python
async def synthesize_batch(
    self,
    texts: list[str],
    *,
    max_concurrent: int = 4,
) -> list[SynthesisResult]:
    # 感情分類はバッチで実行（GPU 効率化）
    all_probs = self.classifier.classify_batch(texts)

    # パラメータ生成もバッチで実行
    all_controls = self.generator.predict_batch(all_probs)

    # VOICEVOX 合成は非同期並行
    semaphore = asyncio.Semaphore(max_concurrent)
    tasks = [
        self._synthesize_single(text, probs, control, semaphore)
        for text, probs, control in zip(texts, all_probs, all_controls)
    ]
    return await asyncio.gather(*tasks)
```

---

## C. 評価フレームワーク (EB3-G01)

### C.1 定量評価 (EB3-G01-01)

#### C.1.1 ラウンドトリップ評価パイプライン

**目的**: パイプライン全体の韻律制御精度を、V-03 のベースライン（mean 3.84）と比較する。

**パイプライン**:

```
テスト入力テキスト
  → EmotionClassifier (6D 感情確率)
    → DeterministicMixer (5D 制御パラメータ)
  → StyleSelector (style_id)
  → VoicevoxClient (音声合成)
  → openSMILE eGeMAPSv02 (88D 韻律特徴量)
  → style_id 内 z-score 正規化
  → JVNV 感情プロファイル重心との距離計算
```

**テストセット構成**:

| 感情     | テキスト数 | 選定基準                                      |
|---------|-----------|-----------------------------------------------|
| anger   | 20        | Phase 0 で anger が支配的に分類されるテキスト      |
| happy   | 20        | 同 happy                                       |
| sad     | 20        | 同 sad                                         |
| disgust | 15        | 同 disgust                                     |
| fear    | 15        | 同 fear                                        |
| surprise| 15        | 同 surprise                                    |
| **計**  | **105**   |                                                |

anger/happy/sad は韻律分離が強いため多め、disgust/fear/surprise は分離が弱いため少なめ。

テキストは WRIME テストスプリットから、Phase 0 分類器の確信度上位を選定する。Phase 1 のグリッドサーチに使用したテキストは除外し、汎化性能を測定する。

**ベースライン比較**:

| 指標 | V-03 ベースライン | 目標 |
|-----|------------------|------|
| 韻律距離 mean | 3.84 | < 3.84 |

V-03 の距離はオラクル条件（全グリッドサーチ音声から最近傍 k=25 件を探索）。パイプラインの予測パラメータがこれを下回ることは容易ではないが、同等レベルであれば学習の成功を示す。

**評価スクリプト構成**:

```python
# emotionbridge/scripts/evaluate_roundtrip.py

def evaluate_roundtrip(
    *,
    pipeline: EmotionBridgePipeline,
    test_texts: list[tuple[str, str]],  # (text, expected_emotion)
    jvnv_centroids: dict[str, np.ndarray],
    output_dir: Path,
) -> dict[str, Any]:
    """ラウンドトリップ評価を実行し、結果を返す。"""
    ...
```

#### C.1.2 感情分類一致率

**目的**: 生成音声の韻律が意図した感情として認識可能かを測定する。

**手順**:

1. テストテキストから EmotionBridgePipeline で音声を生成
2. 生成音声の eGeMAPS 特徴量を抽出
3. 各 JVNV 感情プロファイル重心との距離を計算
4. 距離最小の感情ラベルを「音声の感情」とする
5. 元テキストの感情ラベルと比較し、一致率を算出

```
classification_accuracy[emotion] = (
    一致件数[emotion] / テスト件数[emotion]
)
```

**目標値**:

| 感情 | 目標一致率 | 根拠 |
|-----|----------|------|
| anger | >= 60% | V-01 で韻律的に明確に分離 (silhouette 0.284) |
| happy | >= 60% | 同上 (silhouette 0.195) |
| sad   | >= 60% | 同上、energy が極めてタイト (std=0.13) |
| disgust | 参考値 | 中程度の分離 |
| fear | 参考値 | 分離が弱い (fear-surprise: 0.026) |
| surprise | 参考値 | 分離が弱い |

fear/surprise/disgust は韻律分離が弱いため目標値を設けない。

> **注記**: EB3-G01-01-002（感情分類一致率）は廃止済み。Phase 0はテキスト感情分類器であり生成音声を処理できないため。評価はEB3-G01-01-001（韻律距離）とEB3-G01-01-003（MAE）で実施する。

#### C.1.3 パラメータ予測 MAE

**目的**: パラメータ生成器の予測精度を教師データと比較する。

**手順**:

1. テストセットの各テキストについて、Phase 0 分類器の出力（6D 確率）を入力
2. パラメータ生成器の出力（5D）と、当該感情の V-03 推奨パラメータ（教師値）との MAE を算出

```
MAE[axis] = mean(|predicted[axis] - recommended[axis]|)
```

教師値は V-03 の `emotion_param_matches.json` の各感情 `control_summary` の mean 値:

| 感情 | pitch_shift | pitch_range | speed | energy | pause_weight |
|------|------------|------------|-------|--------|-------------|
| anger | -0.081 | 0.524 | 0.015 | 0.540 | 0.049 |
| happy | 0.242 | 0.046 | 0.257 | -0.318 | 0.133 |
| sad | -0.322 | -0.017 | -0.222 | -0.795 | 0.182 |
| disgust | -0.305 | 0.088 | -0.321 | -0.478 | -0.222 |
| fear | 0.198 | -0.045 | 0.185 | -0.437 | 0.347 |
| surprise | 0.114 | 0.004 | 0.141 | -0.378 | 0.293 |

**目標**: 各軸 MAE <= 0.2 (USDM EB3-G01-01-003)

### C.2 主観評価 (EB3-G01-02)

#### C.2.1 刺激セットの設計

**設計原則**:
- 各感情について、EmotionBridge 条件とデフォルト条件の対を生成
- テキストは自然で短い日本語文（10-30 文字程度）
- 評価者の疲労を考慮し、1セッション 30 分以内

**刺激構成**:

| 感情 | テキスト数 | 条件数 | 刺激数/感情 |
|-----|----------|--------|-----------|
| anger | 5 | 2 (デフォルト, EB) | 10 |
| happy | 5 | 2 | 10 |
| sad | 5 | 2 | 10 |
| disgust | 3 | 2 | 6 |
| fear | 3 | 2 | 6 |
| surprise | 3 | 2 | 6 |
| **計** | **24** | | **48** |

- **デフォルト条件**: ノーマルスタイル + 5D ゼロベクトル（VOICEVOX デフォルト韻律）
- **EmotionBridge 条件**: パイプライン出力（スタイル選択 + 5D パラメータ適用）

テキスト例:
- anger: 「何度言ったらわかるんだ」「ふざけるな」
- happy: 「やったー、合格した！」「今日は最高の一日だった」
- sad: 「もう会えないんだね」「一人で帰る夜道は寂しい」
- fear: 「誰かいるの...？」「暗闇の中で物音がした」
- disgust: 「あの食べ物は二度と食べたくない」
- surprise: 「えっ、本当に！？」「まさかこんなことになるとは」

#### C.2.2 A/B テスト設計

**目的**: EmotionBridge がデフォルトより感情表現として適切かを判定する。

**手順**:
1. 各テキストについて、デフォルト音声と EmotionBridge 音声をランダム順で提示
2. 質問: 「テキスト『[テキスト内容]』の感情をより適切に表現しているのはどちらですか？」
3. 選択肢: A / B / どちらも同じ

**分析**:
- EmotionBridge 選好率 = (EB 選択数) / (全回答数 - 「同じ」回答数)
- 目標: 選好率 > 50% (片側二項検定で有意)

**提示件数**: 24 テキスト x 1 ペア = 24 問/評価者

#### C.2.3 5 段階 MOS 設計

**目的**: 感情表現の自然さを絶対評価する。

**スケール**:
| 点数 | 説明 |
|-----|------|
| 5 | 完全に自然 — テキストの感情を非常によく表現している |
| 4 | ほぼ自然 — 感情表現が適切で、違和感が少ない |
| 3 | 普通 — 感情が伝わるが、やや不自然 |
| 2 | やや不自然 — 感情表現に違和感がある |
| 1 | 全く不自然 — テキストの感情が全く伝わらない |

**提示方法**: テキストと音声を同時に提示。1 音声につき 1 評価。
**提示件数**: 48 刺激 (24 テキスト x 2 条件) を全評価者に提示。

#### C.2.4 感情識別テスト設計

**目的**: 音声から感情を正しく識別できるかを測定する。

**手順**:
1. EmotionBridge 条件の音声のみを提示（テキストは非表示）
2. 質問: 「この音声が表現している感情は何ですか？」
3. 選択肢: anger / happy / sad / disgust / fear / surprise / わからない

**分析**:
- 感情別正解率 = 正解数 / 提示数
- ベンチマーク: JVNV 94% 認識率（ただし JVNV は俳優の演技音声であり、VOICEVOX 合成音声とは直接比較不可）
- 混同行列の分析により、特定の感情ペアの混同パターンを把握

**提示件数**: 24 音声 (EmotionBridge 条件のみ)/評価者

#### C.2.5 回答収集方法

**推奨**: Gradio UI

| 方法 | 利点 | 欠点 |
|-----|------|------|
| Google フォーム | 導入コストゼロ、回答集約容易 | 音声の埋め込みが不便（リンクで対応） |
| Gradio UI | 音声再生が容易、プロジェクト技術スタックと一致 | 公開サーバーが必要（HuggingFace Spaces 利用可） |

Gradio UI を推奨する理由:
1. 音声のインライン再生が可能（Google フォームはリンク経由になる）
2. A/B テストの音声ペアをインタラクティブに提示可能
3. HuggingFace Spaces にデプロイすれば外部公開も容易
4. 回答データを JSON/CSV で自動保存可能

**Gradio UI の基本構成**:
```
[評価画面]
  テキスト表示エリア
  音声プレイヤー A / 音声プレイヤー B
  A/B 選択ボタン
  MOS スライダー (1-5)
  感情識別ドロップダウン
  [次へ] ボタン
```

#### C.2.6 評価者

- パイロット評価: 5-10 名（EB3-G01-02-004 準拠）
- 評価者要件: 日本語ネイティブスピーカー、聴覚に問題がないこと
- 評価環境: ヘッドホン推奨
- 大規模評価の実施判断: パイロット評価の結果に基づく

---

## D. 重要な関心事・未決定事項

### D.1 スタイル韻律プロファイルの妥当性

**懸念**: グリッドサーチ音声の重心は、5D パラメータ空間の均一サンプリングによるものであり、そのスタイルの「ベストな代表」ではない。パラメータが極端な値の音声も含まれるため、重心が中性的に引っ張られる。

**対策案**:
1. 制御パラメータを (0,0,0,0,0) に固定した音声のみで重心を算出する（「素の」スタイル韻律）
2. パラメータ範囲を中央付近 ([-0.3, +0.3]) に限定した音声で重心を算出する

**決定方針**: 両方を算出し、JVNV 感情プロファイルとの距離で比較。より意味のあるマッピングが得られる方を採用。

**実施前提**: まず複数style_idを含むPhase 1/Phase 3 v02データを再生成する（現状はstyle_id=0のみのため比較不能）。

### D.2 低確信度フォールバック閾値

**現状**: 初期運用値として `0.30` を採用済み（EB3-F01-004）。

**決定実験**:
1. WRIME テストセットの全テキストに対して Phase 0 分類器の出力を計算
2. `max(emotion_probs)` の分布を可視化
3. 閾値 {0.25, 0.30, 0.35, 0.40} でフォールバック率を算出
4. Phase 3d の主観評価で、フォールバック音声 vs 低確信度非フォールバック音声の品質を比較

**判定ルール**:
- 0.30でフォールバック率と主観品質のバランスが取れていれば維持
- 低確信度入力の誤スタイル選択が多い場合のみ 0.35/0.40 へ引き上げを検討

### D.3 主観評価の評価者確保

**必要**: 5-10 名のパイロット評価者（日本語ネイティブ）
**確保手段**: プロジェクト関係者、大学研究室、SNS での募集
**リスク**: 十分な評価者が集まらない場合、パイロット規模を 3 名まで縮小して実施

**運用決定**:
- 目標: 5名以上
- 最低実施ライン: 3名（探索的評価として扱い、統計的主張は抑制）

### D.4 推論パイプラインの実行時間

**要件**: リアルタイム性は Phase 3 では求めない。バッチ処理（オフライン評価用）を主眼とする。

**見積もり**:
| ステップ | 推定時間/テキスト |
|---------|----------------|
| Phase 0 感情分類 (BERT 推論) | ~50ms (GPU) / ~200ms (CPU) |
| パラメータ生成 (軽量 MLP) | < 1ms |
| スタイル選択 (テーブル参照) | < 1ms |
| VOICEVOX audio_query | ~100ms |
| VOICEVOX synthesis | ~200-500ms |
| **合計** | ~350-650ms/テキスト (GPU) |

リアルタイム性が将来必要になった場合は、BERT の量子化/蒸留、VOICEVOX のバッチ合成 API 利用等で対応可能。

### D.5 すぐに決められない事項の一覧

| 事項 | 理由 | 影響 | 決定時期 |
|-----|------|------|---------|
| フォールバック閾値の最終値 | 初期値0.30で運用開始済み。最終値は主観評価結果で微調整 | パイプラインの挙動に直接影響 | Phase 3d パイロット完了時 |
| 感情→スタイル距離行列の具体値 | v02 データからの算出が必要 | マッピング表の確定 | Phase 3c スタイルプロファイル構築時 |
| スタイル混合の実施判断 (EB3-E01-02-002) | Phase 3c 実験結果に依存 | 将来の拡張方針 | Phase 3c 完了時 |
| 主観評価の大規模実施判断 | パイロット評価結果に依存 | 評価の信頼性 | Phase 3d パイロット完了時 |
| 感情識別テストの chance level 設定 | 6 選択肢 + 「わからない」の chance level 理論値の算出 | 統計的有意性の判定 | 主観評価設計最終化時 |

---

## E. ファイル配置計画

Phase 3c/3d で追加・変更するファイル:

```
emotionbridge/
  pipeline/
    __init__.py
    bridge.py            # EmotionBridgePipeline クラス
    types.py             # SynthesisResult データクラス
    style_selector.py    # StyleSelector ABC + RuleBasedStyleSelector
  scripts/
    build_style_profiles.py    # スタイル別韻律プロファイル構築
    build_style_mapping.py     # 感情→スタイルマッピング生成
    evaluate_roundtrip.py      # ラウンドトリップ定量評価
    evaluate_classification.py # 感情分類一致率評価
    evaluate_parameter_mae.py  # パラメータ予測 MAE 評価
  evaluation/
    __init__.py
    gradio_app.py        # 主観評価用 Gradio UI
    stimuli.py           # 刺激セット生成
cli.py                   # bridge コマンド追加

configs/
  bridge_config.yaml     # パイプライン設定（チェックポイントパス等）

artifacts/phase3/
  style_profiles/        # スタイル別韻律プロファイル
  style_mapping.json     # 感情→スタイルマッピング
  evaluation/            # 評価結果
    roundtrip/
    classification/
    parameter_mae/
    subjective/
```
