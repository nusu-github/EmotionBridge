"""TTSアダプタ層。制御空間ベクトルをTTSエンジン固有パラメータに変換する。"""

from abc import ABC, abstractmethod
from dataclasses import replace

from emotionbridge.config import ControlSpaceConfig
from emotionbridge.tts.types import AudioQuery, ControlVector


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
            audio_query: VOICEVOX APIから取得したベースクエリ。
            control: 制御空間の5Dベクトル。
        Returns:
            パラメータ適用済みのAudioQuery。

        """
        ...

    @abstractmethod
    def parameter_ranges(self) -> dict[str, tuple[float, float]]:
        """このアダプタが出力する各TTSパラメータの有効範囲を返す。

        Returns:
            パラメータ名 -> (min, max) の辞書。

        """
        ...


def _lerp(range_: tuple[float, float], normalized: float) -> float:
    """[-1, +1] の正規化値を [min, max] にマッピングする。

    v=0 で範囲の中央値、v=-1 で最小値、v=+1 で最大値を返す。
    """
    lo, hi = range_
    return (lo + hi) / 2.0 + normalized * (hi - lo) / 2.0


class VoicevoxAdapter(TTSAdapter):
    """VOICEVOX Engine用TTSアダプタ。制御空間→VOICEVOXパラメータ線形マッピング。

    ``ControlSpaceConfig`` のマッピング範囲設定に基づき、ControlVectorの
    各値 [-1, +1] をVOICEVOXの各スカラーパラメータに線形変換する。
    ``dataclasses.replace`` によるAudioQueryの浅いコピーを作成し、
    accent_phrases（VOICEVOXが生成したプロソディ情報）は保持する。

    マッピング:
        - speedScale       <- lerp(speed_range, control.speed)
        - pitchScale       <- lerp(pitch_shift_range, control.pitch_shift)
        - intonationScale  <- lerp(pitch_range_range, control.pitch_range)
        - volumeScale      <- lerp(energy_range, control.energy)
        - prePhonemeLength  <- lerp(pause_pre_range, control.pause_weight)
        - postPhonemeLength <- lerp(pause_post_range, control.pause_weight)
        - pauseLengthScale  <- lerp(pause_length_scale_range, control.pause_weight)
    """

    def __init__(self, config: ControlSpaceConfig | None = None) -> None:
        self._config = config or ControlSpaceConfig()

    def apply(self, audio_query: AudioQuery, control: ControlVector) -> AudioQuery:
        """制御ベクトルをVOICEVOXパラメータに変換し適用する。

        元のAudioQueryは変更せず、新しいAudioQueryを返す。
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

    def parameter_ranges(self) -> dict[str, tuple[float, float]]:
        """VOICEVOXパラメータの有効範囲を返す。"""
        cfg = self._config
        return {
            "speedScale": cfg.speed_range,
            "pitchScale": cfg.pitch_shift_range,
            "intonationScale": cfg.pitch_range_range,
            "volumeScale": cfg.energy_range,
            "prePhonemeLength": cfg.pause_pre_range,
            "postPhonemeLength": cfg.pause_post_range,
            "pauseLengthScale": cfg.pause_length_scale_range,
        }
