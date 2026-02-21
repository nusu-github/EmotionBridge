COMMON6_CIRCUMPLEX_COORDS: dict[str, tuple[float, float]] = {
    "anger": (0.90, -0.60),
    "disgust": (0.35, -0.85),
    "fear": (0.80, -0.75),
    "happy": (0.75, 0.80),
    "sad": (-0.75, -0.80),
    "surprise": (0.70, 0.05),
}

# TTS制御空間パラメータ
CONTROL_PARAM_NAMES: list[str] = [
    "pitch_shift",
    "pitch_range",
    "speed",
    "energy",
    "pause_weight",
]
NUM_CONTROL_PARAMS = len(CONTROL_PARAM_NAMES)

# DSP後処理制御空間パラメータ
DSP_PARAM_NAMES: list[str] = [
    "jitter_amount",
    "shimmer_amount",
    "aperiodicity_shift",
    "spectral_tilt_shift",
]
NUM_DSP_PARAMS = len(DSP_PARAM_NAMES)

# JVNV感情ラベル
JVNV_EMOTION_LABELS = ["anger", "disgust", "fear", "happy", "sad", "surprise"]
NUM_JVNV_EMOTIONS = len(JVNV_EMOTION_LABELS)

# 分類Go/No-Goで重視する感情
KEY_EMOTION_LABELS = ["anger", "happy", "sad"]
