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
    emotion
    for emotion in EMOTION_LABELS
    if emotion not in LOW_VARIANCE_EMOTION_LABELS
]

NUM_EMOTIONS = len(EMOTION_LABELS)
LABEL_SCALE_MAX = 3.0
