"""音声品質検証モジュール。

生成されたWAVファイルの品質を検証し、ファイルサイズ・音声長・
RMS振幅・サンプルレートのチェックを行う。
"""

import logging
import wave
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from emotionbridge.config import ValidationConfig

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class ValidationResult:
    """音声品質検証結果。"""

    is_valid: bool
    file_size_bytes: int
    duration_seconds: float
    sample_rate: int
    rms_amplitude: float
    errors: list[str] = field(default_factory=list)


class AudioValidator:
    """生成音声の品質検証。

    wave標準ライブラリでWAVファイルを読み込み、以下のチェックを行う:
    - ファイルサイズ
    - 音声長
    - RMS振幅（無音検出）
    - サンプルレート
    """

    def __init__(self, config: ValidationConfig | None = None) -> None:
        self._config = config or ValidationConfig()

    def validate(self, audio_path: Path) -> ValidationResult:
        """WAVファイルの品質を検証する。

        Args:
            audio_path: WAVファイルのパス。

        Returns:
            ValidationResult。

        """
        errors: list[str] = []

        # ファイル存在チェック
        if not audio_path.exists():
            return ValidationResult(
                is_valid=False,
                file_size_bytes=0,
                duration_seconds=0.0,
                sample_rate=0,
                rms_amplitude=0.0,
                errors=["ファイルが存在しない"],
            )

        # ファイルサイズチェック
        file_size = audio_path.stat().st_size
        if file_size < self._config.min_file_size_bytes:
            errors.append(
                f"ファイルサイズ不足: {file_size} bytes "
                f"(最小: {self._config.min_file_size_bytes} bytes)",
            )

        # WAVヘッダ読み込み
        try:
            with wave.open(str(audio_path), "rb") as wf:
                sample_rate = wf.getframerate()
                n_channels = wf.getnchannels()
                sample_width = wf.getsampwidth()
                n_frames = wf.getnframes()
                raw_data = wf.readframes(n_frames)
        except Exception as e:
            return ValidationResult(
                is_valid=False,
                file_size_bytes=file_size,
                duration_seconds=0.0,
                sample_rate=0,
                rms_amplitude=0.0,
                errors=[f"WAVヘッダ読み込み失敗: {e}"],
            )

        # 音声長チェック
        duration = n_frames / sample_rate if sample_rate > 0 else 0.0
        if duration < self._config.min_duration_seconds:
            errors.append(
                f"音声長不足: {duration:.3f}秒 "
                f"(最小: {self._config.min_duration_seconds}秒)",
            )

        # サンプルレートチェック
        if sample_rate != self._config.expected_sample_rate:
            errors.append(
                f"サンプルレート不一致: {sample_rate} Hz "
                f"(期待値: {self._config.expected_sample_rate} Hz)",
            )

        # RMS振幅計算
        rms = self._compute_rms(raw_data, sample_width, n_channels)
        if rms < self._config.min_rms_amplitude:
            errors.append(
                f"RMS振幅不足（無音検出）: {rms:.6f} "
                f"(最小: {self._config.min_rms_amplitude})",
            )

        is_valid = len(errors) == 0

        if not is_valid:
            logger.debug(
                "検証失敗: %s - %s",
                audio_path.name,
                "; ".join(errors),
            )

        return ValidationResult(
            is_valid=is_valid,
            file_size_bytes=file_size,
            duration_seconds=duration,
            sample_rate=sample_rate,
            rms_amplitude=rms,
            errors=errors,
        )

    def validate_batch(self, audio_paths: list[Path]) -> list[ValidationResult]:
        """複数ファイルの一括検証。

        Args:
            audio_paths: WAVファイルのパスリスト。

        Returns:
            ValidationResultのリスト。

        """
        return [self.validate(path) for path in audio_paths]

    @staticmethod
    def _compute_rms(
        raw_data: bytes,
        sample_width: int,
        n_channels: int,
    ) -> float:
        """生バイトデータからRMS振幅を計算する。

        Args:
            raw_data: WAVの生バイトデータ。
            sample_width: サンプル幅（バイト数）。
            n_channels: チャンネル数。

        Returns:
            正規化されたRMS振幅 [0.0, 1.0]。

        """
        if not raw_data:
            return 0.0

        # sample_widthに応じたdtype
        if sample_width == 1:
            dtype = np.uint8
            max_val = 128.0
        elif sample_width == 2:
            dtype = np.int16
            max_val = 32768.0
        elif sample_width == 4:
            dtype = np.int32
            max_val = 2147483648.0
        else:
            return 0.0

        samples = np.frombuffer(raw_data, dtype=dtype)

        # uint8の場合は中央値を0に補正
        if sample_width == 1:
            samples = samples.astype(np.float64) - 128.0
        else:
            samples = samples.astype(np.float64)

        # ステレオの場合はモノラルに変換（左チャンネルのみ使用）
        if n_channels > 1:
            samples = samples[::n_channels]

        if len(samples) == 0:
            return 0.0

        # 正規化してRMS計算
        normalized = samples / max_val
        return float(np.sqrt(np.mean(normalized**2)))
