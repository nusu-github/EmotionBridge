# EmotionBridge DSP後処理層

## 1. 背景と目的

### 1.1 EmotionBridgeの現状

EmotionBridgeは、テキストから感情を分類し、VOICEVOX TTSエンジンの制御パラメータ（pitch_shift, pitch_range, speed, energy, pause_weight）を自動調整することで、感情的な音声合成を実現するパイプラインである。

現在のパイプラインは以下の構成で動作している：

```
テキスト → 感情分類（BERT/WRIME） → DeterministicMixer → VOICEVOX API → 音声出力
```

### 1.2 特定された限界

VOICEVOXのAPI制御パラメータは、感情韻律の3層のうち1層しかカバーしていない：

| 層                   | 内容                               | VOICEVOX APIで制御可能か |
| :------------------- | :--------------------------------- | :----------------------- |
| 超分節的韻律         | F0パターン、発話速度、音量、ポーズ | 部分的に可能             |
| 声質 (voice quality) | jitter, shimmer, HNR, ブレス       | **不可能**               |
| 調音的特徴           | フォルマント遷移、子音強度         | **不可能**               |

特に怒り・悲しみは声質変化に大きく依存しており、現行パイプラインでは表現力に原理的な上限がある。

現行の韻律分析（`docs/prosody_analysis_findings.md`）と整合させるため、以下を前提とする：

- 感情ラベルは common6（anger, disgust, fear, happy, sad, surprise）を使用する（neutral は使わない）
- fear は「高ピッチ平均」だが「F0変動幅の増大」はJVNV重心では確認されない
- style 差は `style_id` 別Z正規化空間では縮退しやすいため、スタイル評価は raw/global z-score 空間で補助確認する

### 1.3 VOICEVOXの内部介入が不可能である根拠

VOICEVOX Coreの検証により、音響モデル中間表現やVocoder入力への介入は不可能であることが確認済み：

- 公式VVMはTalkDomainのみを持ち、モノリシックdecodeで中間表現が外部に露出しない
- モデルフォーマットが`vv_bin`（VOICEVOX独自バイナリ）であり、ONNXツールによる分割も不可
- ExperimentalTalkDomain経路に入れるVVMが存在しない

### 1.4 本計画の目的

VOICEVOX出力波形に対するDSP後処理を追加し、声質層の感情表現を補完する。具体的には以下の制御を実現する：

- **jitter（F0微細揺らぎ）**：声の震え・不安定さ
- **spectral tilt（スペクトル傾斜）**：息混じり感・緊張感
- **shimmer（振幅微細揺らぎ）**：声のかすれ
- **HNR関連操作**：ブレシネス（非周期性成分の増減）

### 1.5 設計制約

- VOICEVOX Coreをブラックボックスとして扱う（モデル改変なし）
- キャラクター音声のアイデンティティを保持する
- TTS エンジン非依存（VOICEVOX以外にも適用可能な設計）
- 利用規約に抵触しない（リバースエンジニアリング禁止条項を回避）

---

## 2. 全体アーキテクチャ

```
テキスト
  → 感情分類（既存）
  → DeterministicMixer（既存）
  → VOICEVOX API（既存：韻律層制御）
  → DSP後処理（新規：声質層制御）← 本計画の対象
  → 最終音声出力
```

DSP後処理層は、感情分類器の出力する感情確率ベクトル（DeterministicMixerと同じ入力）を受け取り、声質操作パラメータを決定する。VOICEVOX出力WAVに対してWORLD解析→パラメータ操作→再合成を行う。

---

## 3. フェーズ構成

本計画は4フェーズで構成される。各フェーズにGo/No-Go判定基準を設け、順次進行する。

| フェーズ | 内容                         | 推定工数 | 依存         |
| :------- | :--------------------------- | :------- | :----------- |
| Phase 0  | WORLD再合成の品質検証        | 0.5〜1日 | なし         |
| Phase 1  | 感情別声質ターゲット値の定義 | 1〜2日   | Phase 0 Pass |
| Phase 2  | DSP操作エンジンの実装        | 2〜3日   | Phase 1      |
| Phase 3  | 統合・評価                   | 2〜3日   | Phase 2      |

---

## 4. Phase 0：WORLD再合成の品質検証

### 4.1 目的

VOICEVOX出力WAVをWORLDで解析→**何も変更せず**再合成し、音質劣化がキャラクター性を損なわないレベルであることを確認する。ここで品質が不十分なら、WORLD以外の手法（rubberband + Parselmouth等）に切り替える。

### 4.2 手順

#### 4.2.1 環境構築

```bash
pip install pyworld numpy scipy soundfile
```

#### 4.2.2 テスト音声の準備

VOICEVOXで以下の条件の音声を生成する（最低3キャラクター × 3文 = 9ファイル）：

- **キャラクター**: ずんだもん（ID:3）、四国めたん（ID:2）、春日部つむぎ（ID:8）
  - 異なる声質特性を持つキャラクターを選ぶことで汎用性を検証
- **テキスト**: 短文（〜10文字）、中文（〜30文字）、長文（〜60文字）
  - 例: 「こんにちは」「今日はとてもいい天気ですね」「昨日の会議で決まったことを皆さんにお伝えしたいと思います」
- **VOICEVOX設定**: デフォルトパラメータ（speed=1.0, pitch=0.0, intonation=1.0）

#### 4.2.3 WORLD解析→再合成スクリプト

```python
import pyworld as pw
import soundfile as sf
import numpy as np

def world_roundtrip(input_path, output_path):
    """WORLD解析→無変更再合成"""
    wav, sr = sf.read(input_path)
    wav = wav.astype(np.float64)

    # 解析
    f0, time_axis = pw.dio(wav, sr)        # F0抽出
    f0 = pw.stonemask(wav, f0, time_axis, sr)  # F0精密化
    sp = pw.cheaptrick(wav, f0, time_axis, sr)  # スペクトル包絡
    ap = pw.d4c(wav, f0, time_axis, sr)     # 非周期性指標

    # 無変更再合成
    wav_out = pw.synthesize(f0, sp, ap, sr)

    sf.write(output_path, wav_out.astype(np.float32), sr)
    return f0, sp, ap
```

#### 4.2.4 定量評価

以下の3指標を算出する：

| 指標                           | 算出方法                                 | 意味                                       |
| :----------------------------- | :--------------------------------------- | :----------------------------------------- |
| **PESQ**                       | `pesq` ライブラリ（ITU-T P.862）         | 知覚的音声品質。電話帯域の品質評価標準     |
| **メルケプストラム歪み (MCD)** | メルケプストラム係数間のユークリッド距離 | スペクトル包絡の変形量。値が小さいほど良い |
| **F0 RMSE**                    | 原音と再合成音のF0コンター間のRMSE       | ピッチ再現精度                             |

```python
# pip install pesq pypesq
from pesq import pesq

def evaluate_roundtrip(original_path, resynthesized_path):
    orig, sr_orig = sf.read(original_path)
    resynth, sr_resynth = sf.read(resynthesized_path)

    # 長さを揃える
    min_len = min(len(orig), len(resynth))
    orig = orig[:min_len]
    resynth = resynth[:min_len]

    # PESQ（16kHzにリサンプル必要な場合あり）
    # PESQは8kHz or 16kHzのみ対応。VOICEVOX出力が24kHz等の場合はリサンプルする
    from scipy.signal import resample
    if sr_orig != 16000:
        orig_16k = resample(orig, int(len(orig) * 16000 / sr_orig))
        resynth_16k = resample(resynth, int(len(resynth) * 16000 / sr_orig))
        pesq_score = pesq(16000, orig_16k, resynth_16k, 'wb')
    else:
        pesq_score = pesq(sr_orig, orig, resynth, 'wb')

    return pesq_score
```

#### 4.2.5 主観評価（簡易）

実施者自身が以下をABテスト形式で判定する（ブラインドで実施すること）：

1. 原音と再合成音をランダム順で再生
2. 以下の3項目を5段階で評価：
   - **キャラクター同一性**: 同じキャラクターに聞こえるか（1=全く別人 〜 5=区別不能）
   - **音質**: 不自然なノイズやアーティファクトがないか（1=著しい劣化 〜 5=劣化なし）
   - **自然さ**: 発話として自然か（1=機械的 〜 5=自然）

### 4.3 Go/No-Go判定基準

| 指標               | Go条件                      | 根拠                                                            |
| :----------------- | :-------------------------- | :-------------------------------------------------------------- |
| PESQ               | ≥ 3.5（全ファイル平均）     | 3.5はnarrowbandで"Good"相当。これ以下では後段操作前に品質が不足 |
| MCD                | ≤ 6.0 dB                    | 音声変換研究における一般的な許容範囲                            |
| F0 RMSE            | ≤ 5.0 Hz                    | EmotionBridgeの韻律制御が無意味化しないために必要               |
| キャラクター同一性 | ≥ 4.0（全キャラクター平均） | 3以下ではキャラクター性の保持という前提が崩れる                 |

#### Go の場合

→ Phase 1に進む。

#### No-Go の場合

→ 以下の代替手段を検討する：

1. **WORLDのパラメータ調整**: DIOの代わりにHarvestを使用（F0精度向上、計算時間増加）
2. **ParselmouthによるPraat操作に切り替え**: WORLD再合成を経由せず、Praatのフォルマント操作・ピッチ操作を直接適用
3. **rubberband + Sox の組み合わせ**: タイムストレッチとフィルタリングのみで声質操作を近似
4. **DSP後処理アプローチ自体を断念**: VC学習ルートの検討に移行

代替手段1→2→3の順で検証し、いずれも基準を満たさない場合のみ4を選択する。

---

## 5. Phase 1：感情別声質ターゲット値の定義

### 5.1 目的

JVNVコーパスの感情別音声から、DSP操作のターゲットとなる声質パラメータの統計量を抽出する。EmotionBridgeの既存eGeMAPS分析結果を活用する。

### 5.2 前提

EmotionBridgeの既存パイプラインで以下が利用可能であること：

- JVNVコーパスの話者正規化済みeGeMAPS特徴量（`artifacts/prosody/v01/jvnv_egemaps_normalized.parquet`）
- 感情重心プロファイル（`artifacts/prosody/v01/jvnv_emotion_profiles.json`）
- 話者正規化済みの特徴量（z-score正規化後）

### 5.3 抽出対象パラメータ

eGeMAPS特徴量からDSP操作に直接対応するものを抽出する：

| eGeMAPS特徴量                          | DSP操作              | 操作方法                                       |
| :------------------------------------- | :------------------- | :--------------------------------------------- |
| `egemaps__jitterLocal_sma3nz_amean`    | F0微細揺らぎ付加     | F0コンターにフレーム単位のランダム揺らぎを加算 |
| `egemaps__shimmerLocaldB_sma3nz_amean` | 振幅微細揺らぎ付加   | 波形振幅にフレーム単位のランダム変動を乗算     |
| `egemaps__HNRdBACF_sma3nz_amean`       | 非周期性成分の増減   | WORLDのaperiodicity行列を操作                  |
| `egemaps__spectralFluxV_sma3nz_amean`  | スペクトル変動の増減 | スペクトル包絡のフレーム間変化量を操作         |
| `egemaps__slopeV0-500_sma3nz_amean`    | スペクトル傾斜操作   | スペクトル包絡の傾斜を調整（ブレシネス）       |
| `egemaps__slopeV500-1500_sma3nz_amean` | 同上（中域）         | 同上                                           |
| `egemaps__F1frequency_sma3nz_amean`    | フォルマントシフト   | **操作しない**（キャラクター性に直結）         |
| `egemaps__F2frequency_sma3nz_amean`    | 同上                 | **操作しない**                                 |
| `egemaps__F3frequency_sma3nz_amean`    | 同上                 | **操作しない**                                 |

**重要: フォルマント周波数（F1/F2/F3）は操作対象から除外する。** これらはキャラクターの声のアイデンティティの根幹であり、操作するとキャラクター性が崩壊する。EmotionBridgeの知見（話者正規化でフォルマント差が桁違い）がこの判断を裏付ける。

### 5.4 手順

#### 5.4.1 感情別統計量の算出

JVNVの話者正規化済みeGeMAPS特徴量から、感情ごとに上記パラメータの統計量を算出する：

```python
import pandas as pd
import numpy as np

# EmotionBridgeの既存データを読み込み
# （実際のプロジェクト成果物を直接使う）
features_df = pd.read_parquet("artifacts/prosody/v01/jvnv_egemaps_normalized.parquet")

# common6ラベルに正規化
label_map = {
    "anger": "anger",
    "disgust": "disgust",
    "fear": "fear",
    "happy": "happy",
    "joy": "happy",
    "sad": "sad",
    "sadness": "sad",
    "surprise": "surprise",
}
features_df["emotion_common6"] = (
    features_df["emotion"].astype(str).str.lower().map(label_map)
)
features_df = features_df[features_df["emotion_common6"].notna()].copy()

TARGET_FEATURES = [
    "egemaps__jitterLocal_sma3nz_amean",
    "egemaps__shimmerLocaldB_sma3nz_amean",
    "egemaps__HNRdBACF_sma3nz_amean",
    "egemaps__spectralFluxV_sma3nz_amean",
    "egemaps__slopeV0-500_sma3nz_amean",
    "egemaps__slopeV500-1500_sma3nz_amean",
]

EMOTIONS = ["anger", "disgust", "fear", "happy", "sad", "surprise"]

# 感情別統計量
emotion_stats = {}
for emo in EMOTIONS:
    emo_data = features_df[features_df["emotion_common6"] == emo][TARGET_FEATURES]
    emotion_stats[emo] = {
        "mean": emo_data.mean().to_dict(),
        "std": emo_data.std().to_dict(),
        "median": emo_data.median().to_dict(),
    }

# 全感情平均との差分（操作量の基準）
global_mean = features_df[TARGET_FEATURES].mean().to_dict()
for emo in EMOTIONS:
    emotion_stats[emo]["delta_from_global"] = {
        feat: emotion_stats[emo]["mean"][feat] - global_mean[feat]
        for feat in TARGET_FEATURES
    }
```

#### 5.4.2 DSP操作マッピングテーブルの構築

上記のdelta値を、DSP操作の具体的なパラメータに変換する。変換関数は以下の構造を取る：

```python
# eGeMAPS delta → DSP操作量の変換
DSP_MAPPING = {
    "jitter_amount": {
        # jitterのdelta_from_globalに比例（増加方向のみ加算）
        "source": "egemaps__jitterLocal_sma3nz_amean",
        "transform": lambda delta: np.clip(delta * JITTER_SCALE, 0.0, MAX_JITTER),
    },
    "shimmer_amount": {
        "source": "egemaps__shimmerLocaldB_sma3nz_amean",
        "transform": lambda delta: np.clip(delta * SHIMMER_SCALE, 0.0, MAX_SHIMMER),
    },
    "aperiodicity_shift": {
        # HNRが低い = 非周期性が高い → aperiodicityを増加
        "source": "egemaps__HNRdBACF_sma3nz_amean",
        "transform": lambda delta: np.clip(-delta * AP_SCALE, -MAX_AP, MAX_AP),
    },
    "spectral_tilt_shift": {
        "source": "egemaps__slopeV0-500_sma3nz_amean",
        "transform": lambda delta: np.clip(delta * TILT_SCALE, -MAX_TILT, MAX_TILT),
    },
}
```

スケーリング係数（`JITTER_SCALE`等）の初期値はPhase 2で手動チューニングにより決定する。

#### 5.4.3 感情ごとの操作方向の検証

各感情に対する操作方向は、一般論ではなく JVNV 実測値（現行 `jvnv_emotion_profiles.json`）に合わせる。初期仮説として以下を採用する：

| 感情     | jitter                   | shimmer               | HNR(→AP)     | spectral tilt |
| :------- | :----------------------- | :-------------------- | :----------- | :------------ |
| anger    | ↑（明確に増加）          | ↑（明確に増加）       | ↓（AP↑）     | 要実測調整    |
| disgust  | →〜微増                  | →                     | ↓（AP↑、小） | 要実測調整    |
| fear     | ↓                        | ↓                     | ↑（AP↓）     | 要実測調整    |
| happy    | ↓                        | →                     | ↑（AP↓）     | 要実測調整    |
| sad      | ↓〜→（増加は仮定しない） | →（増加は仮定しない） | ↑（AP↓）     | 要実測調整    |
| surprise | ↓                        | ↓                     | ↑（AP↓）     | 要実測調整    |

特に `sad` と `fear` は「jitter/shimmer を増やす」という固定仮説を置かず、AB評価で有効な方向のみ採用する。

### 5.5 Go/No-Go判定基準

| 条件             | Go基準                                                                    | 根拠                                 |
| :--------------- | :------------------------------------------------------------------------ | :----------------------------------- |
| データ利用可能性 | 上記6特徴量のうち4つ以上がeGeMAPS抽出済みデータに含まれている             | 最低限の操作軸数の確保               |
| 感情間分離性     | 上記特徴量のうち2つ以上で、少なくとも2感情ペアがCohen's d ≥ 0.5の差を持つ | 操作しても知覚可能な差が生じる見込み |
| 方向整合性       | 上表の「実測ベース方向」と、算出delta方向が4感情以上で一致                | データ駆動の方向を維持するため       |

#### No-Go の場合

- 特徴量不足 → openSMILEで対象特徴量のみ再抽出（eGeMAPS全88次元は不要）
- 感情間分離性不足 → 声質操作の効果が限定的であることを受け入れ、操作量を控えめに設定してPhase 2に進む（ただし期待値を下方修正）
- 方向不整合 → 固定仮説を捨て、感情ごとにABテストで最適方向を探索する（JVNV特性を優先）

---

## 6. Phase 2：DSP操作エンジンの実装

### 6.1 目的

VOICEVOX出力WAVに対して、感情パラメータに基づく声質操作を行うDSPモジュールを実装する。

### 6.2 モジュール構成

```
emotion_dsp/
├── __init__.py
├── analyzer.py        # WORLD解析ラッパー
├── manipulator.py     # 声質操作関数群
├── synthesizer.py     # WORLD再合成ラッパー
├── pipeline.py        # 解析→操作→再合成のパイプライン
├── params.py          # 感情→DSP操作量のマッピング定義
└── config.py          # スケーリング係数・上下限値の設定
```

### 6.3 各操作の実装仕様

#### 6.3.1 Jitter（F0微細揺らぎ）付加

```python
def add_jitter(f0, amount, seed=None):
    """
    F0コンターにフレーム単位のランダム揺らぎを付加する。

    Parameters:
        f0: np.ndarray - F0コンター（Hz）。0はunvoiced。
        amount: float - 揺らぎ量（0.0〜1.0）。0.0=操作なし、1.0=最大揺らぎ。
        seed: int - 再現性のための乱数シード

    Returns:
        np.ndarray - 揺らぎ付加後のF0コンター

    操作方法:
        voiced区間のF0に対して、ガウスノイズを加算する。
        ノイズの標準偏差 = amount * LOCAL_F0_STD * JITTER_MAX_RATIO
        LOCAL_F0_STDは当該voiced区間のF0標準偏差。
        JITTER_MAX_RATIO = 0.05（F0の5%を上限とする）

    安全制約:
        - unvoiced区間（f0 == 0）は操作しない
        - 操作後のF0が50Hz以下または800Hz以上にならないようクリップ
        - voiced/unvoiced境界の3フレームはフェードイン/アウト
    """
```

#### 6.3.2 Shimmer（振幅揺らぎ）付加

```python
def add_shimmer(wav, f0, amount, sr, seed=None):
    """
    波形振幅にフレーム単位のランダム変動を付加する。

    Parameters:
        wav: np.ndarray - 波形データ
        f0: np.ndarray - F0コンター（voiced/unvoiced判定に使用）
        amount: float - 揺らぎ量（0.0〜1.0）
        sr: int - サンプリングレート
        seed: int - 乱数シード

    操作方法:
        voiced区間をピッチ周期ごとに分割し、各周期の振幅に
        ランダムなゲイン変動を乗算する。
        ゲイン変動 = 1.0 + amount * SHIMMER_MAX_DB * noise
        SHIMMER_MAX_DB = 0.1（約1dBの変動を上限）

    安全制約:
        - unvoiced区間は操作しない
        - ゲインが0.5以下または1.5以上にならないようクリップ
        - 周期境界はクロスフェード（5ms）
    """
```

#### 6.3.3 Aperiodicity（非周期性）操作

```python
def modify_aperiodicity(ap, shift, f0):
    """
    WORLDの非周期性指標を操作し、ブレシネスを増減する。

    Parameters:
        ap: np.ndarray - 非周期性指標行列（WORLD D4Cの出力）
        shift: float - シフト量（-1.0〜1.0）。正=ブレシネス増加、負=減少
        f0: np.ndarray - voiced/unvoiced判定に使用

    操作方法:
        voiced区間のaperiodicityに対して一様にshiftを加算する。
        aperiodicityはdBスケールで0.0（完全周期的）〜1.0（完全非周期的）。
        shift > 0 → 非周期性増加 → 息混じりの声
        shift < 0 → 非周期性減少 → クリアな声

    安全制約:
        - aperiodicityの値域を[0.0, 0.95]にクリップ
          （1.0にすると完全ノイズになりキャラクター性が崩壊する）
        - unvoiced区間は操作しない
        - 低域（0-500Hz）は操作量を50%に抑制（基本波の保持）
    """
```

#### 6.3.4 Spectral Tilt（スペクトル傾斜）操作

```python
def modify_spectral_tilt(sp, tilt_shift, sr):
    """
    スペクトル包絡の傾斜を調整し、声の明暗を制御する。

    Parameters:
        sp: np.ndarray - スペクトル包絡行列（WORLD CheapTrickの出力）
        tilt_shift: float - 傾斜調整量（-1.0〜1.0）。
                    正=高域強調（緊張・怒り）、負=高域減衰（ブレシネス・悲しみ）
        sr: int - サンプリングレート

    操作方法:
        周波数軸に沿って線形のゲインカーブを適用する。
        gain(f) = 1.0 + tilt_shift * TILT_SCALE * (f / (sr/2) - 0.5)
        TILT_SCALE = 0.3（最大±3dB程度の変化に制限）

        具体的には：
        - tilt_shift > 0: 高域を持ち上げ、低域を下げる → 緊張した声
        - tilt_shift < 0: 高域を下げ、低域を持ち上げる → 柔らかい声

    安全制約:
        - ゲインの範囲を[0.5, 2.0]にクリップ（±6dBが限度）
        - 0-200Hz帯域はゲイン固定（基本波保護）
        - 5000Hz以上はゲイン変化を50%に抑制（高域ノイズ抑制）
    """
```

### 6.4 パイプライン統合

```python
def process_emotion_dsp(wav_path, dsp_params, output_path):
    """
    メインパイプライン。

    Parameters:
        wav_path: str - VOICEVOX出力WAVのパス
        dsp_params: dict - DSP操作パラメータ
            例: {"jitter_amount": 0.3, "shimmer_amount": 0.2, ...}
        output_path: str - 出力WAVパス
    """
    # 1. WAV読み込み
    wav, sr = sf.read(wav_path)

    # 2. WORLD解析
    f0, sp, ap = analyze(wav, sr)

    # 3. DSP操作の適用（順序固定）
    f0 = add_jitter(f0, dsp_params["jitter_amount"])
    ap = modify_aperiodicity(ap, dsp_params["aperiodicity_shift"], f0)
    sp = modify_spectral_tilt(sp, dsp_params["spectral_tilt_shift"], sr)

    # 4. WORLD再合成
    wav_out = pw.synthesize(f0, sp, ap, sr)
    wav_out = add_shimmer(wav_out, f0, dsp_params["shimmer_amount"], sr)

    # 5. 出力
    sf.write(output_path, wav_out.astype(np.float32), sr)
```

### 6.5 スケーリング係数のチューニング方法

Phase 2の段階では、まず1感情（怒り）だけを対象にチューニングする。

1. Phase 1で算出したdelta値を確認
2. スケーリング係数の初期値を設定（控えめに0.3倍程度から開始）
3. 怒り（100%確率）でテスト音声を生成
4. 聴取して「怒りの声質変化が知覚できるか」を確認
5. 知覚できない → スケーリング係数を1.5倍に増加して再試行
6. 知覚できるがキャラクター性が損なわれる → スケーリング係数を0.7倍に減少
7. 知覚でき、かつキャラクター性が維持されている → その係数を採用

**このループは最大10回とし、10回で収束しない場合は当該パラメータの操作を断念する。**

### 6.6 Go/No-Go判定基準

| 条件               | Go基準                                            | 根拠                                               |
| :----------------- | :------------------------------------------------ | :------------------------------------------------- |
| 怒り音声の主観評価 | 「怒りらしさ」が無操作時より向上（5段階で+1以上） | DSP操作の効果が知覚可能であることの最低条件        |
| キャラクター同一性 | Phase 0と同等（スコア差が-0.5以内）               | 声質操作がキャラクター性を破壊していないことの確認 |
| アーティファクト   | 不自然なノイズ・クリック音が聴取されない          | 基本的な品質の担保                                 |

---

## 7. Phase 3：統合・評価

### 7.1 目的

全6感情に対してDSP操作を適用し、EmotionBridgeの既存パイプラインと統合した上で、総合的な品質評価を行う。

### 7.2 統合作業

#### 7.2.1 DSP Mapperの追加（DeterministicMixerは非改変）

既存の DeterministicMixer（6×5）は変更しない。代わりに、同じ感情確率ベクトルを入力とする `EmotionDSPMapper`（6×4）を追加し、DSP操作パラメータ（jitter, shimmer, aperiodicity, spectral_tilt）を独立生成する。

#### 7.2.2 パイプラインの結合

```python
# 統合後のメインフロー
def generate_emotional_speech(text, speaker_id):
    # 1. 感情分類（既存）
    emotion_probs = classify_emotion(text)

    # 2. 韻律制御（既存）とDSP制御（新規）を並列生成
    tts_params = deterministic_mixer.generate(emotion_probs)
    dsp_params = emotion_dsp_mapper.generate(emotion_probs)

    # 3. VOICEVOX音声合成（既存）
    wav_path = voicevox_synthesize(text, speaker_id, tts_params)

    # 4. DSP後処理（新規）
    output_path = emotion_dsp.process(
        wav_path=wav_path,
        dsp_params=dsp_params,
        output_path=build_output_path(text, speaker_id),
    )

    return output_path
```

### 7.3 総合評価

#### 7.3.1 比較条件

| 条件                | 説明                                                           |
| :------------------ | :------------------------------------------------------------- |
| **A. ベースライン** | VOICEVOX デフォルト（感情制御なし）                            |
| **B. 韻律のみ**     | EmotionBridge既存パイプライン（DeterministicMixer → VOICEVOX） |
| **C. 韻律＋声質**   | 既存パイプライン + DSP後処理（本計画の成果物）                 |

#### 7.3.2 評価項目

| 項目               | 評価方法                                                                           | 指標                           |
| :----------------- | :--------------------------------------------------------------------------------- | :----------------------------- |
| 感情認識精度       | 外部の音声感情認識モデル（SER）に入力（任意）。SER未導入時は主観評価を主指標とする | 意図した感情の分類精度（%）    |
| 感情らしさ         | 主観評価（5段階MOS）                                                               | 各感情の「らしさ」スコア       |
| キャラクター同一性 | 主観評価（5段階）                                                                  | キャラクターとして認識できるか |
| 音質               | 主観評価（5段階MOS）                                                               | 全体的な音声品質               |
| eGeMAPS距離        | B vs Cの出力音声をeGeMAPSで分析し、JVNVの感情クラスター重心との距離を比較          | 距離の減少率（%）              |
| ギャップ重点評価   | controllability重み付き距離（V-02のfeature_weights）で怒り/悲しみの残差変化を比較  | 重み付き距離の減少率（%）      |

#### 7.3.3 テスト音声セット

- キャラクター: 3名（Phase 0と同じ）
- 感情: 6感情 × 各3文 = 18文
- 条件: A/B/C × 18文 × 3キャラクター = 162ファイル

#### 7.3.4 成功基準

| 指標               | 基準                                                | 根拠                                   |
| :----------------- | :-------------------------------------------------- | :------------------------------------- |
| 感情らしさ         | 条件Cが条件Bを2感情以上で有意に上回る               | DSP後処理の追加に意味があることの確認  |
| キャラクター同一性 | 条件Cの平均スコアが4.0以上                          | キャラクター性の保持が大前提           |
| 音質               | 条件Cの平均スコアが3.5以上                          | 実用に耐える品質                       |
| eGeMAPS距離        | 条件Cが条件Bより、怒りまたは悲しみで距離10%以上短縮 | 声質依存感情のギャップ縮小の定量的確認 |

---

## 8. リスクと対策

| リスク                                     | 影響度                 | 対策                                                               |
| :----------------------------------------- | :--------------------- | :----------------------------------------------------------------- |
| WORLD再合成の品質劣化が想定以上            | 高（プロジェクト停止） | Phase 0で早期に検出。代替手法（Parselmouth, rubberband）へ切り替え |
| DSP操作がキャラクター性を損なう            | 高                     | 各操作に安全制約を設定。操作量の上限を保守的に設定                 |
| 操作量のチューニングが収束しない           | 中                     | 10回ルールで打ち切り。当該パラメータは操作対象から除外             |
| 感情間で操作が干渉する                     | 中                     | 操作の独立性をeGeMAPSの偏相関分析で事前検証                        |
| style評価の前処理縮退（style_id z-score）  | 中                     | raw/global z-score空間での再評価を併用し、縮退時は結論を保留       |
| VOICEVOXのバージョンアップで出力特性が変化 | 低                     | DSP層はWAV入力のみに依存するため、影響は限定的                     |

---

## 9. 依存ライブラリ

| ライブラリ | 用途                               | インストール            |     |
| :--------- | :--------------------------------- | :---------------------- | --- |
| pyworld    | WORLD解析・再合成                  | `pip install pyworld`   |     |
| soundfile  | WAV読み書き                        | `pip install soundfile` |     |
| numpy      | 数値計算                           | `pip install numpy`     |     |
| scipy      | リサンプリング等                   | `pip install scipy`     |     |
| pesq       | 音声品質評価                       | `pip install pesq`      |     |
| opensmile  | eGeMAPS特徴量抽出（Phase 3評価用） | `pip install opensmile` |     |

---

## 10. 用語集

| 用語               | 説明                                                                                                            |
| :----------------- | :-------------------------------------------------------------------------------------------------------------- |
| WORLD              | 音声分析合成システム。F0、スペクトル包絡、非周期性指標の3要素に音声を分解し、各要素を独立に操作して再合成できる |
| F0                 | 基本周波数。声の高さに対応する                                                                                  |
| Jitter             | F0の周期ごとの微細なゆらぎ。感情的な声の不安定さに関与                                                          |
| Shimmer            | 振幅の周期ごとの微細なゆらぎ。声のかすれに関与                                                                  |
| HNR                | Harmonics-to-Noise Ratio。調波成分と雑音成分の比。高い=クリアな声、低い=かすれた声                              |
| Aperiodicity       | 非周期性指標。WORLDにおけるHNRの逆数的な概念                                                                    |
| Spectral Tilt      | スペクトル包絡の傾き。低域優勢=柔らかい声、高域優勢=鋭い声                                                      |
| eGeMAPS            | extended Geneva Minimalistic Acoustic Parameter Set。88次元の音響特徴量セット                                   |
| common6            | EmotionBridgeで使用する6感情ラベル集合（anger, disgust, fear, happy, sad, surprise）                            |
| JVNV               | 日本語感情音声コーパス。EmotionBridgeの韻律分析の基盤データ                                                     |
| DeterministicMixer | EmotionBridgeの既存モジュール。感情確率ベクトルから教師行列の線形混合でTTS制御パラメータを生成                  |
| EmotionDSPMapper   | 本計画で追加するモジュール。感情確率ベクトルからDSP後処理パラメータ（4軸）を生成                                |
| MCD                | Mel Cepstral Distortion。メルケプストラム歪み。音声変換の品質評価に標準的に使用される指標                       |
| PESQ               | Perceptual Evaluation of Speech Quality。ITU-T P.862に基づく知覚的音声品質評価指標                              |
