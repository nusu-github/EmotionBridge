"""生成パイプラインのレポートモジュール。

パイプライン実行結果の統計情報をまとめ、JSONとして保存可能な形式で提供する。
"""

from dataclasses import dataclass, field


@dataclass(slots=True)
class GenerationReport:
    """生成パイプラインの実行レポート。"""

    total_tasks: int
    completed: int
    failed: int
    skipped: int
    invalid: int
    elapsed_seconds: float
    avg_synthesis_time_seconds: float
    total_audio_duration_seconds: float
    total_audio_size_bytes: int
    dominant_emotion_distribution: dict[str, int] = field(default_factory=dict)
    failure_reasons: dict[str, int] = field(default_factory=dict)
    config_snapshot: dict[str, object] = field(default_factory=dict)

    def to_dict(self) -> dict[str, object]:
        """JSON シリアライズ用の辞書を返す。

        Returns:
            レポート内容の辞書表現。

        """
        return {
            "total_tasks": self.total_tasks,
            "completed": self.completed,
            "failed": self.failed,
            "skipped": self.skipped,
            "invalid": self.invalid,
            "elapsed_seconds": self.elapsed_seconds,
            "avg_synthesis_time_seconds": self.avg_synthesis_time_seconds,
            "total_audio_duration_seconds": self.total_audio_duration_seconds,
            "total_audio_size_bytes": self.total_audio_size_bytes,
            "dominant_emotion_distribution": dict(self.dominant_emotion_distribution),
            "failure_reasons": dict(self.failure_reasons),
            "config_snapshot": dict(self.config_snapshot),
        }
