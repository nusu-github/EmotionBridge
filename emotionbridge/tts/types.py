"""TTS層の型定義。

ControlVector, AudioQuery, Mora, AccentPhrase, SpeakerInfo, SpeakerStyle を提供する。
"""

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from emotionbridge.constants import CONTROL_PARAM_NAMES

# ---------------------------------------------------------------------------
# ControlVector
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class ControlVector:
    """TTS制御空間の5次元ベクトル。各値は [-1.0, +1.0]。

    フィールド順序は ``constants.CONTROL_PARAM_NAMES`` に一致する:
    pitch_shift, pitch_range, speed, energy, pause_weight
    """

    pitch_shift: float = 0.0
    pitch_range: float = 0.0
    speed: float = 0.0
    energy: float = 0.0
    pause_weight: float = 0.0

    def __post_init__(self) -> None:
        for name in CONTROL_PARAM_NAMES:
            val = getattr(self, name)
            if not isinstance(val, (int, float)):
                msg = f"{name} must be a number, got {type(val).__name__}"
                raise TypeError(msg)
            if not (-1.0 <= val <= 1.0):
                msg = f"{name} must be in [-1.0, 1.0], got {val}"
                raise ValueError(msg)

    def to_numpy(self) -> np.ndarray:
        """Shape (5,) のnumpy配列に変換。順序はCONTROL_PARAM_NAMES準拠。"""
        return np.array(
            [
                self.pitch_shift,
                self.pitch_range,
                self.speed,
                self.energy,
                self.pause_weight,
            ],
            dtype=np.float32,
        )

    @classmethod
    def from_numpy(cls, arr: np.ndarray) -> "ControlVector":
        """Shape (5,) のnumpy配列からControlVectorを生成。"""
        if arr.shape != (5,):
            msg = f"Expected shape (5,), got {arr.shape}"
            raise ValueError(msg)
        return cls(
            pitch_shift=float(arr[0]),
            pitch_range=float(arr[1]),
            speed=float(arr[2]),
            energy=float(arr[3]),
            pause_weight=float(arr[4]),
        )


# ---------------------------------------------------------------------------
# VOICEVOX AudioQuery 関連
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class Mora:
    """VOICEVOX APIのモーラ情報。"""

    text: str
    vowel: str
    vowel_length: float
    pitch: float
    consonant: str | None = None
    consonant_length: float | None = None

    def to_dict(self) -> dict[str, Any]:
        """VOICEVOX API送信用dictに変換。"""
        d: dict[str, Any] = {
            "text": self.text,
            "vowel": self.vowel,
            "vowel_length": self.vowel_length,
            "pitch": self.pitch,
            "consonant": self.consonant,
            "consonant_length": self.consonant_length,
        }
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Mora":
        """APIレスポンスからMoraを生成。"""
        return cls(
            text=d["text"],
            vowel=d["vowel"],
            vowel_length=d["vowel_length"],
            pitch=d["pitch"],
            consonant=d.get("consonant"),
            consonant_length=d.get("consonant_length"),
        )


@dataclass(slots=True)
class AccentPhrase:
    """VOICEVOX APIのアクセント句情報。"""

    moras: list[Mora]
    accent: int
    pause_mora: Mora | None = None
    is_interrogative: bool = False

    def to_dict(self) -> dict[str, Any]:
        """VOICEVOX API送信用dictに変換。"""
        d: dict[str, Any] = {
            "moras": [m.to_dict() for m in self.moras],
            "accent": self.accent,
            "pause_mora": self.pause_mora.to_dict() if self.pause_mora else None,
            "is_interrogative": self.is_interrogative,
        }
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "AccentPhrase":
        """APIレスポンスからAccentPhraseを生成。"""
        pause_mora_raw = d.get("pause_mora")
        return cls(
            moras=[Mora.from_dict(m) for m in d["moras"]],
            accent=d["accent"],
            pause_mora=Mora.from_dict(pause_mora_raw) if pause_mora_raw else None,
            is_interrogative=d.get("is_interrogative", False),
        )


@dataclass(slots=True)
class AudioQuery:
    """VOICEVOX APIの音声合成クエリ。"""

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

    def to_dict(self) -> dict[str, Any]:
        """VOICEVOX API送信用dictに変換。

        ネストしたMora/AccentPhraseも再帰的にdictに変換する。
        Noneフィールドもnullとして含める（VOICEVOXのAPIがnullを受け付けるため）。
        """
        return {
            "accent_phrases": [ap.to_dict() for ap in self.accent_phrases],
            "speedScale": self.speedScale,
            "pitchScale": self.pitchScale,
            "intonationScale": self.intonationScale,
            "volumeScale": self.volumeScale,
            "prePhonemeLength": self.prePhonemeLength,
            "postPhonemeLength": self.postPhonemeLength,
            "outputSamplingRate": self.outputSamplingRate,
            "outputStereo": self.outputStereo,
            "pauseLength": self.pauseLength,
            "pauseLengthScale": self.pauseLengthScale,
            "kana": self.kana,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "AudioQuery":
        """APIレスポンスからAudioQueryを生成。"""
        return cls(
            accent_phrases=[AccentPhrase.from_dict(ap) for ap in d["accent_phrases"]],
            speedScale=d["speedScale"],
            pitchScale=d["pitchScale"],
            intonationScale=d["intonationScale"],
            volumeScale=d["volumeScale"],
            prePhonemeLength=d["prePhonemeLength"],
            postPhonemeLength=d["postPhonemeLength"],
            outputSamplingRate=d["outputSamplingRate"],
            outputStereo=d["outputStereo"],
            pauseLength=d.get("pauseLength"),
            pauseLengthScale=d.get("pauseLengthScale", 1.0),
            kana=d.get("kana"),
        )


# ---------------------------------------------------------------------------
# SpeakerInfo
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class SpeakerStyle:
    """VOICEVOXのスピーカースタイル情報。"""

    name: str
    id: int
    style_type: str = "talk"


@dataclass(frozen=True, slots=True)
class SpeakerInfo:
    """VOICEVOXのスピーカー情報。"""

    name: str
    speaker_uuid: str
    styles: list[SpeakerStyle] = field(default_factory=list)
    version: str = ""
