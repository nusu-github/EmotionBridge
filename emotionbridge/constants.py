EMOTION_LABELS = [
    "joy",
    "sadness",
    "anticipation",
    "surprise",
    "anger",
    "fear",
    "disgust",
    "trust",
]

NUM_EMOTIONS = len(EMOTION_LABELS)
LABEL_SCALE_MAX = 3.0

EMOTION_CIRCUMPLEX_COORDS: dict[str, tuple[float, float]] = {
    "joy": (0.75, 0.80),
    "sadness": (-0.75, -0.80),
    "anticipation": (0.55, 0.35),
    "surprise": (0.70, 0.05),
    "anger": (0.90, -0.60),
    "fear": (0.80, -0.75),
    "disgust": (0.35, -0.85),
    "trust": (0.20, 0.65),
}

# 共通6感情ラベル向け（JVNV）
COMMON6_CIRCUMPLEX_COORDS: dict[str, tuple[float, float]] = {
    "anger": EMOTION_CIRCUMPLEX_COORDS["anger"],
    "disgust": EMOTION_CIRCUMPLEX_COORDS["disgust"],
    "fear": EMOTION_CIRCUMPLEX_COORDS["fear"],
    "happy": EMOTION_CIRCUMPLEX_COORDS["joy"],
    "sad": EMOTION_CIRCUMPLEX_COORDS["sadness"],
    "surprise": EMOTION_CIRCUMPLEX_COORDS["surprise"],
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

# JVNV感情ラベル
JVNV_EMOTION_LABELS = ["anger", "disgust", "fear", "happy", "sad", "surprise"]
NUM_JVNV_EMOTIONS = len(JVNV_EMOTION_LABELS)

# WRIME 8感情 -> JVNV 6感情への対応
WRIME_TO_JVNV_MAPPING: dict[str, str] = {
    "joy": "happy",
    "sadness": "sad",
    "anger": "anger",
    "fear": "fear",
    "disgust": "disgust",
    "surprise": "surprise",
}

# WRIME 8Dのうち、JVNV 6感情をJVNV順で取り出すためのインデックス
# EMOTION_LABELS = [joy, sadness, anticipation, surprise, anger, fear, disgust, trust]
# JVNV_EMOTION_LABELS = [anger, disgust, fear, happy, sad, surprise]
WRIME_TO_JVNV_INDICES = [4, 6, 5, 0, 1, 3]

# 分類Go/No-Goで重視する感情
KEY_EMOTION_LABELS = ["anger", "happy", "sad"]
