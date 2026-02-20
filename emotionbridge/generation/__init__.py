"""音声サンプル生成パイプラインパッケージ。

テキスト感情ベクトルとTTS制御パラメータの対応関係を学習するための
三つ組データセット（テキスト x 制御パラメータ x 音声）を自動生成する。
"""

from emotionbridge.generation.grid import GridSampler
from emotionbridge.generation.pipeline import GenerationPipeline
from emotionbridge.generation.validator import AudioValidator

__all__ = [
    "AudioValidator",
    "GenerationPipeline",
    "GridSampler",
]
