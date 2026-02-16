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
