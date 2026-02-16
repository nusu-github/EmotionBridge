"""TTS抽象層 + VOICEVOX実装。"""

from emotionbridge.tts.adapter import TTSAdapter, VoicevoxAdapter
from emotionbridge.tts.types import (
    AudioQuery,
    ControlVector,
    SpeakerInfo,
)
from emotionbridge.tts.voicevox_client import VoicevoxClient

__all__ = [
    "AudioQuery",
    "ControlVector",
    "SpeakerInfo",
    "TTSAdapter",
    "VoicevoxAdapter",
    "VoicevoxClient",
]
