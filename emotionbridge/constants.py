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

LOW_VARIANCE_EMOTION_LABELS = ["anger", "trust"]
MAJOR_EMOTION_LABELS = [
    emotion for emotion in EMOTION_LABELS if emotion not in LOW_VARIANCE_EMOTION_LABELS
]

NUM_EMOTIONS = len(EMOTION_LABELS)
LABEL_SCALE_MAX = 3.0

# Phase 0→Phase 1.5: 8D感情ベクトルを連続軸へ写像するための座標
# 値域は [-1, +1] を想定（Arousal, Valence）
CIRCUMPLEX_AXIS_NAMES = ["arousal", "valence"]

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

# Phase 2/3 で使う共通6感情ラベル向け（JVNV / emotion2vec）
COMMON6_CIRCUMPLEX_COORDS: dict[str, tuple[float, float]] = {
    "anger": EMOTION_CIRCUMPLEX_COORDS["anger"],
    "disgust": EMOTION_CIRCUMPLEX_COORDS["disgust"],
    "fear": EMOTION_CIRCUMPLEX_COORDS["fear"],
    "happy": EMOTION_CIRCUMPLEX_COORDS["joy"],
    "sad": EMOTION_CIRCUMPLEX_COORDS["sadness"],
    "surprise": EMOTION_CIRCUMPLEX_COORDS["surprise"],
}

# Phase 1: TTS制御空間パラメータ
CONTROL_PARAM_NAMES: list[str] = [
    "pitch_shift",
    "pitch_range",
    "speed",
    "energy",
    "pause_weight",
]
NUM_CONTROL_PARAMS = len(CONTROL_PARAM_NAMES)

# Phase 2: JVNV感情ラベル
JVNV_EMOTION_LABELS = ["anger", "disgust", "fear", "happy", "sad", "surprise"]
