# Phase 3b: パラメータ生成器 詳細設計書

> **対象要求**: EB3-D01, EB3-B01-02, EB3-G01-01
> **作成日**: 2026-02-18
> **ステータス**: 初版
> **前提**: V-01/V-02/V-03 検証完了、emotion_param_matches.json 生成済み

---

## A. 教師データの構造設計

### A.1 emotion_param_matches.json の実データ構造

V-03マッチングスクリプト（`match_emotion_params.py`）の出力を確認した結果、以下の構造が得られている。

```
{
  "nearest_k": 25,
  "feature_count": 88,
  "summary": {
    "<emotion>": {
      "num_jvnv_samples": int,      // JVNVの当該感情サンプル数
      "nearest_k": 25,
      "distance_mean": float,        // 近傍25件の平均距離
      "distance_std": float,
      "control_summary": {
        "ctrl_<param>": {
          "mean": float,             // 近傍25件のパラメータ平均
          "std": float,
          "min": float,
          "max": float
        }
      }
    }
  }
}
```

**重要な観察**: `match_emotion_params.py` の要約は mean 中心だが、`direct_matching_profiles.json` では各感情の `recommended_control` が median で保存されている。USDM仕様 EB3-B01-02-003 に合わせ、Phase 3b の教師値は median を正式採用する（後述 A.3）。

### A.2 6感情 x 5Dパラメータ 教師テーブル（採用値: median）

V-03成果物（`direct_matching_profiles.json`）に基づく推奨パラメータ（k=25の中央値）:

| 感情 | pitch_shift | pitch_range | speed | energy | pause_weight |
|------|-------------|-------------|-------|--------|--------------|
| anger | -0.0920 | **+0.6498** | +0.0662 | **+0.5284** | +0.0329 |
| disgust | -0.3255 | +0.1160 | -0.2902 | -0.5168 | -0.2150 |
| fear | +0.1897 | -0.0508 | +0.2254 | -0.4127 | **+0.4492** |
| happy | **+0.2248** | +0.1874 | **+0.2599** | -0.3250 | +0.1606 |
| sad | **-0.3143** | -0.0187 | **-0.2226** | **-0.8372** | +0.3043 |
| surprise | +0.0891 | -0.0120 | +0.2598 | -0.3831 | **+0.4157** |

**文献整合性**: anger（energy最大・pitch_range最大）、happy（pitch_shift最大・speed正）、sad（energy最小で極めてタイト std=0.13）は感情韻律研究の知見と整合。

### A.3 mean vs median の実装状況と対応方針

**現状**:
- `match_emotion_params.py` の `summary.control_summary` は mean/std/min/max を出力
- `prepare_direct_matching.py` は median を用いた `recommended_control` を `direct_matching_profiles.json` に保存

**USDM規定**: EB3-B01-02-003「推奨パラメータ候補k件の中央値を教師値とする（外れ値に頑健にするため、平均ではなく中央値）」

**決定方針**:
1. Phase 3bの教師テーブルは `direct_matching_profiles.json` の `recommended_control`（median）を正とする
2. `emotion_param_matches.parquet` は近傍個票として保持し、必要に応じてmean/median差分を再計算する
3. `match_emotion_params.py` に median を併記する改善は「可読性向上のための後続タスク」として扱う（必須ではない）

**実測メモ（k=25）**: 30軸中7軸で `|mean-median| >= 0.05`、最大差は0.141（happy/pitch_range）。median採用の影響は実質的に有意。

### A.4 教師データのバリエーション生成

k=25件の近傍データは `emotion_param_matches.parquet` に個別レコードとして保存済み。バリエーション生成方法:

1. **中央値 1点**: 感情あたり1つの代表教師値（基本方式）
2. **k件全展開**: 各近傍サンプルの制御パラメータを個別の教師レコードとして使用（6感情 x 25件 = 150レコード）
3. **ブートストラップ**: k件からリサンプリングして複数の中央値推定を生成（信頼区間の確認用）

---

## B. 学習データ生成戦略

### B.1 方式1: ルックアップテーブル方式（推奨: 初期ベースライン）

**概要**: 6感情の推奨パラメータを固定教師とし、one-hot入力で学習する最もシンプルな方式。

- **入力**: one-hot(6D) — 例: anger = [1, 0, 0, 0, 0, 0]
- **出力**: recommended_params(5D) — 例: [-0.08, +0.52, +0.01, +0.54, +0.05]
- **データ量**: 実質6点（拡張なしの場合）
- **拡張方法**:
  - k件全展開: 150レコード（6感情 x 25近傍）
  - Dropout augmentation: 学習時のDropout(0.3)が暗黙的な正則化として機能
  - ソフトラベル混合: one-hotに微小ノイズを加えて混合感情を模擬（例: anger=0.85, surprise=0.15）
    - ノイズ注入: Dirichlet分布（alpha=10 で元クラス支配的、alpha=1 で均等に近い）
    - 生成数: 感情あたり100-500サンプル

**利点**: Phase 0再学習への依存なし。即座に開始可能。
**欠点**: 教師データが少なく、混合感情領域の汎化が弱い可能性。

### B.2 方式2: WRIME合成教師方式

**概要**: WRIMEの感情強度分布から合成soft labelを生成し、各soft labelに対して重み付き推奨パラメータを教師値として計算する。

- **入力**: WRIMEの6感情強度を正規化したsoft label（6D、合計1.0に正規化）
  - WRIME raw: joy=2.1, sadness=0.3, anger=0.1, fear=0.0, disgust=0.0, surprise=0.5
  - 正規化後: [0.70, 0.10, 0.03, 0.00, 0.00, 0.17]
- **出力（教師値）**: 重み付き推奨パラメータ = Σ_e p(e) × recommended_params_e
- **データ量**: WRIMEのフィルタリング後サンプル数（max_intensity > 1のフィルタ適用後、数千件規模）
  - anticipation, trust は除外。6感情の強度のみ使用

**利点**: 実際のテキスト感情分布を反映した自然な混合パターンが得られる。データ量が豊富。
**欠点**: WRIMEの感情強度が「テキストの感情ラベル」であり、Phase 0分類器の出力確率分布とは性質が異なる可能性がある。

**実装上の注意**:
- WRIMEの6感情強度合計が0のサンプル（neutral）は除外する
- 正規化はsoftmax（温度付き）ではなく単純なL1正規化を使用
- 各サンプルの教師パラメータは `params_target = Σ_e (intensity_e / Σ intensity) × recommended_e`

### B.3 方式3: Phase 0出力直接利用方式

**概要**: 再学習後のPhase 0分類器でWRIMEテキストを推論し、出力確率を入力として使用。

- **入力**: Phase 0 Softmax出力（6D確率ベクトル、合計1.0）
- **出力（教師値）**: 重み付き推奨パラメータ = Σ_e p(e|text) × recommended_params_e
- **データ量**: WRIME全体（Phase 0が推論可能な全テキスト）

**利点**: 推論時の入力分布と学習時の入力分布が一致（train-test分布の整合性が最高）。
**欠点**: Phase 0再学習（EB3-C01）完了が前提。循環依存のリスクあり（後述 G.2）。

### B.4 推奨戦略

**段階的アプローチ**:

1. **Phase 3b初期**: 方式1（ルックアップテーブル + Dirichlet augmentation）でベースラインを確立
2. **Phase 3b中期**: 方式2（WRIME合成教師）で混合感情の汎化を検証
3. **Phase 0再学習完了後**: 方式3（Phase 0出力直接利用）で最終学習

方式1で各軸MAE 0.2以内（EB3-G01-01-003）を達成できればPhase 3bのGo基準を満たすため、方式2/3は改善オプションとして位置づける。

---

## C. モデルアーキテクチャ詳細

### C.1 ParameterGenerator クラス設計

```python
class ParameterGenerator(nn.Module):
    """6D感情確率ベクトル → 5D制御パラメータの変換器。

    EB3-D01-003: Linear(6, 64) -> ReLU -> Dropout(0.3) -> Linear(64, 5) -> tanh
    """

    def __init__(
        self,
        num_emotions: int = 6,        # JVNV_EMOTION_LABELS の数
        hidden_dim: int = 64,
        num_params: int = 5,           # NUM_CONTROL_PARAMS
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        self.fc1 = nn.Linear(num_emotions, hidden_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_dim, num_params)
        self.tanh = nn.Tanh()

    def forward(self, emotion_probs: torch.Tensor) -> torch.Tensor:
        """
        Args:
            emotion_probs: (batch_size, 6) - 感情確率ベクトル
        Returns:
            (batch_size, 5) - 制御パラメータ [-1, +1]
        """
        x = self.fc1(emotion_probs)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return self.tanh(x)
```

### C.2 パラメータ数の詳細

| 層 | パラメータ | 計算 |
|---|---|---|
| fc1 (weight) | 6 x 64 = 384 | |
| fc1 (bias) | 64 | |
| fc2 (weight) | 64 x 5 = 320 | |
| fc2 (bias) | 5 | |
| **合計** | **773** | |

Phase 0 の TextEmotionRegressor（約110Mパラメータ）と比較して5桁小さい。

### C.3 初期化方法

- **fc1, fc2**: PyTorchデフォルト（Kaiming Uniform）を使用。773パラメータでは初期化方法の影響は限定的
- **代替案（検討）**: fc2のbiasを推奨パラメータの全感情平均で初期化する。学習初期から合理的な出力範囲に近づける効果が期待できる
  - bias_init = mean(recommended_params, axis=0) → atanh変換してbias初期値とする
  - ただしtanh後の値がrequiredなので、atanh(target)がbias初期値となる

### C.4 配置場所

`emotionbridge/model/generator.py` として新規作成。`emotionbridge/model/__init__.py` から公開する。

---

## D. 損失関数の詳細設計

### D.1 重み付きMSE損失

USDM EB3-D01-004 に基づく損失関数:

```
loss = Σ_e p(e|text) × MSE(predicted_params, recommended_params_e)
     = Σ_e p(e|text) × (1/5) Σ_d (predicted_d - recommended_e_d)²
```

ここで:
- `p(e|text)`: 入力の感情確率ベクトル（6D、合計1.0）
- `predicted_params`: モデル出力（5D）
- `recommended_params_e`: 感情eの推奨パラメータ（5D、定数テンソル）

### D.2 推奨パラメータテンソル

教師テーブルを `(6, 5)` のテンソルとして事前計算し、`register_buffer` でモデルに保持する。

```python
# recommended_params: shape (6, 5) - 各行が1感情の推奨パラメータ
# 行の順序: JVNV_EMOTION_LABELS = [anger, disgust, fear, happy, sad, surprise]
# 列の順序: CONTROL_PARAM_NAMES = [pitch_shift, pitch_range, speed, energy, pause_weight]

recommended_params = torch.tensor([
  [-0.0920,  0.6498,  0.0662,  0.5284,  0.0329],  # anger
  [-0.3255,  0.1160, -0.2902, -0.5168, -0.2150],  # disgust
  [ 0.1897, -0.0508,  0.2254, -0.4127,  0.4492],  # fear
  [ 0.2248,  0.1874,  0.2599, -0.3250,  0.1606],  # happy
  [-0.3143, -0.0187, -0.2226, -0.8372,  0.3043],  # sad
  [ 0.0891, -0.0120,  0.2598, -0.3831,  0.4157],  # surprise
], dtype=torch.float32)
```

> **注**: 上記は median 値（k=25）。mean値は比較用ログとして保持する。

### D.3 テンソル演算での損失計算

```python
def weighted_mse_loss(
    predicted: torch.Tensor,      # (B, 5)
    emotion_probs: torch.Tensor,  # (B, 6)
    recommended: torch.Tensor,    # (6, 5) - register_buffer
) -> torch.Tensor:
    # predicted を 6感情分に展開して各推奨パラメータとの差を計算
    # predicted: (B, 1, 5) vs recommended: (1, 6, 5) → (B, 6, 5)
    diff = predicted.unsqueeze(1) - recommended.unsqueeze(0)  # (B, 6, 5)
    mse_per_emotion = (diff ** 2).mean(dim=2)                 # (B, 6)

    # 感情確率で重み付け
    weighted = (emotion_probs * mse_per_emotion).sum(dim=1)   # (B,)
    return weighted.mean()                                     # scalar
```

### D.4 勾配の流れ: p(e|text) の detach 判断

**設計判断: p(e|text) は detach する（定数として扱う）。**

**理由**:
1. パラメータ生成器への入力である `emotion_probs` は、Phase 0分類器の出力またはWRIMEラベルから生成された固定値である
2. パラメータ生成器の学習中にPhase 0の勾配を逆伝播させる必要はない（Phase 0は別途学習済み）
3. 方式1/2では入力自体が定数（one-hotまたはWRIMEラベル）であり、detachの有無は実質無関係
4. 方式3でPhase 0と同時にfine-tuneする場合のみ detach の是非が問題になるが、**Phase 0とパラメータ生成器の同時学習は本設計の範囲外**とする

**実装上の影響**: 損失関数内で `emotion_probs` を重みとして使う際、`emotion_probs.detach()` を明示的に呼ぶ。コードの意図を明確にするため。

---

## E. 学習設定

### E.1 Optimizer

- **Optimizer**: Adam（AdamWは不要。weight decayは773パラメータに対して過剰正則化のリスク）
- **Learning rate**: 1e-3（初期値。小規模モデルのため比較的大きなlrが適切）
- **代替**: lr=5e-4 をグリッドサーチ候補に含める

### E.2 Scheduler

- **方式**: ReduceLROnPlateau（patience=10, factor=0.5）
- **理由**: 学習データが少ないためwarmupの必要性が低い。validation lossの停滞時にlrを下げる方が適切
- **代替**: 学習が十分高速（<100エポック）であれば、scheduler なしも妥当

### E.3 Epochs / Batch Size

- **Max epochs**: 500（early stoppingで早期終了を期待）
- **Batch size**:
  - 方式1（6点 + augmentation）: batch_size=32（augmentation後のデータ量に依存）
  - 方式2（WRIME合成）: batch_size=64
- **理由**: データが小さいため、1エポックの計算コストは無視できる。十分な学習回数を確保

### E.4 Early Stopping

- **Validation metric**: weighted MSE loss（学習損失と同一指標）
- **Patience**: 30エポック
- **理由**: 小規模データでは損失の変動が大きい。patience を大きめに設定して早期打ち切りを回避
- **補助指標**: パラメータ予測MAE（EB3-G01-01-003の目標値 0.2 を監視）

### E.5 チェックポイント保存戦略

- **保存トリガ**: validation loss が best を更新したタイミング
- **保存内容**:
  ```python
  {
      "model_state_dict": model.state_dict(),
      "recommended_params": recommended_params,  # 教師テーブル（再現性のため）
      "emotion_labels": JVNV_EMOTION_LABELS,
      "control_param_names": CONTROL_PARAM_NAMES,
      "training_strategy": "lookup_table_v1",     # 方式の識別子
      "nearest_k": 25,
      "config": training_config_dict,
  }
  ```
- **保存先**: `artifacts/phase3b/checkpoints/best_generator.pt`
- **Phase 0との構造的対称性**: Phase 0のチェックポイント（`best_model.pt`）と同様の形式で、モデル設定を同梱して自己完結的にする

### E.6 学習データの分割

- **方式1**: 6点しかないためhold-outは不適切。Leave-One-Out Cross Validation（LOOCV）で汎化性能を推定。最終モデルは全6点で学習
- **方式1 + augmentation**: augmented data を 80:20 に分割。augmentation はtrain splitにのみ適用
- **方式2**: WRIME の既存split（train/val/test）をそのまま使用

---

## F. ラウンドトリップ評価設計

### F.1 評価パイプライン

```
テストテキスト（感情ラベル付き）
  → Phase 0 感情分類器 → 6D確率ベクトル p(e|text)
  → パラメータ生成器 → 5D予測パラメータ
  → VOICEVOX音声合成（style_id + 5Dパラメータ）
  → openSMILE eGeMAPSv02 抽出（88D）
  → style_id内 z-score正規化
  → JVNV感情プロファイル重心との距離計算
  → ラウンドトリップ韻律距離
```

### F.2 ベースラインとの比較

**V-03ベースライン（EB3-G01-01-001）**: 感情別距離の平均

| 感情 | V-03 distance mean |
|------|-------------------|
| anger | 4.6354 |
| disgust | 3.3547 |
| fear | 3.8535 |
| happy | 3.7605 |
| sad | 3.7846 |
| surprise | 3.6488 |
| **全体平均** | **3.8396** |

**目標（2段階）**:
- **必達**: V-03と同等水準（全体平均で +10% 以内）
- **ストレッチ**: V-03 の各感情距離を下回る

**注意**: V-03の距離は「最近傍k件の集合統計」であり、推論パイプラインが常にこれを下回るとは限らない。まずは同等水準を満たすことを成功条件とする。

### F.3 評価テキストセット

- **Phase 0テストセット**: WRIME test splitのテキスト（未知テキストへの汎化評価）
- **感情別代表テキスト**: 高確信度テキスト（Phase 0が90%以上の確率で特定感情と判定したもの）を感情別に10件ずつ選定

### F.4 自動化パイプライン

`emotionbridge/scripts/evaluate_roundtrip.py` として実装。

```
入力:
  --generator-checkpoint: パラメータ生成器のチェックポイント
  --phase0-checkpoint: Phase 0分類器のチェックポイント
  --config: experiment_config.yaml（正規化パラメータ等の参照）
  --test-texts: 評価用テキストファイル（1行1テキスト + 感情ラベル）
  --voicevox-style-id: 音声合成に使用するstyle_id

処理:
  1. テキスト → Phase 0 → 6D確率
  2. 6D確率 → パラメータ生成器 → 5Dパラメータ
  3. VOICEVOX音声合成（要: VOICEVOXエンジン起動）
  4. eGeMAPS抽出
  5. 正規化（事前計算済みのμ, σを使用）
  6. JVNV重心との距離計算

出力:
  - artifacts/phase3b/reports/roundtrip_evaluation.json
  - 感情別距離テーブル（V-03ベースラインとの比較）
  - パラメータ予測MAE（推奨パラメータとの比較）
```

### F.5 追加指標: パラメータ予測MAE（EB3-G01-01-003）

ラウンドトリップ評価とは独立に、推奨パラメータとの直接比較を行う:

```
MAE_axis = (1/N) Σ_i |predicted_d_i - target_d_i|
```

- `target_d`: テキストの最支配感情に対応する推奨パラメータ
- **目標**: 各軸 MAE 0.2 以内
- **注意**: 混合感情テキストの場合、targetの定義が曖昧になる。最支配感情のパラメータを暫定targetとし、重み付きtarget（Σ_e p(e) × recommended_e）との比較も併記する

---

## G. 重要な関心事

### G.1 教師データが実質6点の場合のオーバーフィット問題

**論点**: 773パラメータのモデルに対して教師点が少ないため、通常の回帰タスク観点では過学習リスクが懸念される。

**方針**:
1. **解析ベースラインを先に導入**: `target = Σ_e p(e|text) * recommended_e` を基準出力として採用（学習不要・解釈容易）
2. **MLPは改善枠として比較**: 解析ベースラインに対する改善がある場合のみ採用
3. **Dirichlet augmentation + LOOCV** で補間の安定性を確認

**結論**: 本タスクの本質は「6頂点間の補間」であり、まずは解析ベースラインで堅実に成立性を担保し、MLPは二段階目で評価する。

### G.2 Phase 0再学習との依存関係

**依存構造**:
```
EB3-C01 (Phase 0再学習: 8D→6クラス分類)
  ↓ 出力: 6D確率ベクトル
EB3-D01 (パラメータ生成器)
  ↓ 入力: 6D確率ベクトル
```

**循環依存のリスク**: パラメータ生成器の学習にPhase 0の出力が必要（方式3）だが、Phase 0の評価にパラメータ生成器が必要（EB3-G01-01-002: 感情分類一致率）。

**回避策**:
- **方式1/2 はPhase 0に依存しない**: one-hotまたはWRIMEラベルから入力を生成。Phase 0の完成を待たずに着手可能
- **Phase 0とパラメータ生成器は逐次学習**: Phase 0完了後に方式3で再学習（fine-tune）する。同時学習は行わない
- **推論パイプラインのみPhase 0が必要**: ラウンドトリップ評価にはPhase 0が必要だが、パラメータ生成器の学習自体にはPhase 0は必須でない

### G.3 k値の最適化タイミング

**現状**: k=25（`experiment_config.yaml` のデフォルト値）

**k値の影響**:
- k が小さい: 推奨パラメータが最近傍に強く依存し、外れ値の影響大。教師値の分散が大きい
- k が大きい: 感情間の差が平滑化され、教師値が似通ってくる

**最適化方針**: EB3-B01-02-002に基づき、`k=25` を暫定維持し、`k={15, 25, 50}` で最終決定する。

**実施手順**:
1. k = {15, 25, 50} でそれぞれ教師テーブルを生成
2. 各kでパラメータ生成器を学習
3. ラウンドトリップ評価で最適なkを選定
4. 選定基準: ラウンドトリップ韻律距離が最小のk

補足:
- k=5 は教師値の揺れが大きく（一部軸で符号反転を観測）、初期探索から除外
- k=50 は平滑化が強く、感情差を弱めるリスクがあるため比較枠として扱う

**タイミング**: Phase 3b中期。方式1のベースラインが確立された後にk値のグリッドサーチを実施。

### G.4 すぐに決められない事項

| 事項 | 理由 | 影響 | 決定タイミング |
|------|------|------|----------------|
| 最適なk値 | ラウンドトリップ評価の結果に依存 | k=5とk=50では教師値が大幅に異なる可能性 | Phase 3b中期（ベースライン確立後） |
| 方式2のWRIMEラベル正規化温度 | 感情強度の分布によってsoft labelの尖り具合が変わる | 混合感情の学習バランスに影響 | Phase 3b中期（方式2着手時） |
| ニュートラルフォールバック閾値（EB3-F01-004） | パラメータ生成器の出力分布の確認が必要 | 推論時の「確信度が低い入力」の定義 | Phase 3b完了時 |
| Phase 0再学習のGo判定 | Phase 0再学習（EB3-C01）の結果に依存 | 方式3の実施可否、ラウンドトリップ評価の信頼性 | Phase 3a完了時 |

---

## 付録: ファイル配置計画

```
emotionbridge/
├── model/
│   ├── regressor.py          # 既存: TextEmotionRegressor (Phase 0)
│   ├── generator.py          # 新規: ParameterGenerator (Phase 3b)
│   └── __init__.py           # generator を公開に追加
├── training/
│   ├── trainer.py            # 既存: Phase 0 学習ループ
│   ├── generator_trainer.py  # 新規: Phase 3b 学習ループ
│   └── metrics.py            # 既存 + Phase 3b 用メトリクス追加
├── scripts/
│   ├── match_emotion_params.py     # 既存（median追加予定）
│   ├── train_generator.py          # 新規: パラメータ生成器学習スクリプト
│   └── evaluate_roundtrip.py       # 新規: ラウンドトリップ評価スクリプト
├── constants.py              # JVNV_EMOTION_LABELS（既存）, 新規定数追加
└── config.py                 # Phase3bConfig（新規 dataclass）

configs/
└── phase3b.yaml              # パラメータ生成器学習設定

artifacts/
└── phase3b/                  # 新規ディレクトリ
    ├── checkpoints/
    │   └── best_generator.pt
    ├── teacher_table/
    │   └── recommended_params.json  # 教師テーブル（k値・median/mean の記録）
    └── reports/
        ├── training_history.json
        ├── roundtrip_evaluation.json
        └── parameter_mae.json
```
