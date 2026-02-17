"""音声生成パイプライン。

テキスト選定 → パラメータグリッド生成 → 非同期バッチ音声合成 →
品質検証 → データセット書き出し → レポート生成 の一連のフローを実行する。
"""

import asyncio
import json
import logging
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from emotionbridge.config import Phase1Config
from emotionbridge.data.wrime import PreparedSplit
from emotionbridge.generation.dataset import (
    TripletRecord,
    save_dataset,
    save_metadata,
)
from emotionbridge.generation.grid import GridSampler
from emotionbridge.generation.report import GenerationReport
from emotionbridge.generation.text_selector import TextSelector
from emotionbridge.generation.validator import AudioValidator
from emotionbridge.inference.encoder import EmotionEncoder
from emotionbridge.tts.adapter import VoicevoxAdapter
from emotionbridge.tts.types import AudioQuery, ControlVector
from emotionbridge.tts.voicevox_client import VoicevoxClient

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class GenerationTask:
    """音声生成の1単位。"""

    task_id: str  # "{text_id:04d}_{param_idx:04d}_s{style_id:04d}"
    text_id: int
    text: str
    emotion_vec: np.ndarray
    control_vector: ControlVector
    style_id: int
    output_path: Path


@dataclass(slots=True)
class GenerationResult:
    """音声生成結果。"""

    task: GenerationTask
    success: bool
    audio_duration_seconds: float = 0.0
    audio_file_size_bytes: int = 0
    voicevox_params: dict[str, float] | None = None
    error: str | None = None


class GenerationPipeline:
    """テキスト x 制御パラメータ → 音声 の一括生成パイプライン。

    asyncio.Semaphore で同時リクエスト数を制御し、チェックポイント機能で
    中断・再開に対応する。
    """

    def __init__(self, config: Phase1Config) -> None:
        self._config = config
        self._encoder = EmotionEncoder(
            checkpoint_path=config.phase0_checkpoint,
            device=config.device,
        )
        self._sampler = GridSampler(config.grid)
        self._selector = TextSelector(config.text_selection, self._encoder)
        self._validator = AudioValidator(config.validation)
        self._client = VoicevoxClient(
            base_url=config.voicevox.base_url,
            timeout=config.voicevox.timeout,
            max_retries=config.voicevox.max_retries,
            retry_delay=config.voicevox.retry_delay,
        )
        self._adapter = VoicevoxAdapter(config.control_space)
        self._output_dir = Path(config.generation.output_dir)
        self._audio_dir = self._output_dir / config.generation.audio_subdir

    def _style_ids(self) -> list[int]:
        configured = self._config.voicevox.speaker_ids
        if configured:
            # 順序保持で重複除去
            return list(dict.fromkeys(int(style_id) for style_id in configured))
        return [int(self._config.voicevox.default_speaker_id)]

    async def run(self, splits: dict[str, PreparedSplit]) -> GenerationReport:
        """パイプライン全体を実行する。

        1. テキスト選定 (TextSelector)
        2. パラメータグリッド生成 (GridSampler)
        3. 非同期バッチ音声合成 (VoicevoxClient + VoicevoxAdapter)
        4. 品質チェック (AudioValidator)
        5. データセット書き出し (dataset.py)
        6. レポート生成

        Args:
            splits: build_phase0_splits() の出力。

        Returns:
            GenerationReport。

        """
        start_time = time.monotonic()

        # 1. テキスト選定
        logger.info("テキスト選定を開始...")
        selected_texts = self._selector.select(splits)

        # 2. タスク生成
        logger.info("生成タスクを構築中...")
        completed_ids = self._load_checkpoint()
        style_ids = self._style_ids()
        tasks: list[GenerationTask] = []
        skipped_count = 0

        logger.info("対象style_id: %s", style_ids)

        for sel_text in selected_texts:
            text_dir = self._audio_dir / f"{sel_text.text_id:04d}"
            text_dir.mkdir(parents=True, exist_ok=True)

            params_array = self._sampler.sample(sel_text.text_id)

            for style_id in style_ids:
                for param_idx in range(len(params_array)):
                    task_id = f"{sel_text.text_id:04d}_{param_idx:04d}_s{style_id:04d}"

                    # チェックポイントまたは既存ファイルでスキップ
                    if completed_ids is not None and task_id in completed_ids:
                        skipped_count += 1
                        continue

                    output_path = text_dir / f"{task_id}.wav"
                    if self._config.generation.skip_existing and output_path.exists():
                        skipped_count += 1
                        continue

                    ctrl_row = params_array[param_idx]
                    control_vector = ControlVector.from_numpy(ctrl_row)

                    tasks.append(
                        GenerationTask(
                            task_id=task_id,
                            text_id=sel_text.text_id,
                            text=sel_text.text,
                            emotion_vec=sel_text.emotion_vec,
                            control_vector=control_vector,
                            style_id=style_id,
                            output_path=output_path,
                        ),
                    )

        total_tasks = len(tasks) + skipped_count
        logger.info(
            "総タスク数: %d (新規: %d, スキップ: %d)",
            total_tasks,
            len(tasks),
            skipped_count,
        )

        # 3. 非同期バッチ音声合成
        logger.info("音声生成を開始...")
        results = await self._generate_batch(tasks)

        # 4. 品質チェック + レコード作成
        logger.info("品質検証とデータセット構築中...")
        records: list[TripletRecord] = []
        completed = 0
        failed = 0
        invalid = 0
        failure_reasons: dict[str, int] = {}
        total_audio_duration = 0.0
        total_audio_size = 0
        synthesis_times: list[float] = []
        emotion_dist: dict[str, int] = {}

        for result in results:
            task = result.task
            dominant_emotion = self._get_dominant_emotion(task.emotion_vec)
            emotion_dist[dominant_emotion] = emotion_dist.get(dominant_emotion, 0) + 1

            if not result.success:
                failed += 1
                reason = result.error or "unknown"
                failure_reasons[reason] = failure_reasons.get(reason, 0) + 1
                continue

            completed += 1

            # 品質検証
            validation = self._validator.validate(task.output_path)
            if not validation.is_valid:
                invalid += 1

            total_audio_duration += validation.duration_seconds
            total_audio_size += validation.file_size_bytes

            # 相対パスで記録
            try:
                rel_path = task.output_path.relative_to(self._output_dir)
            except ValueError:
                rel_path = task.output_path

            records.append(
                TripletRecord(
                    text_id=task.text_id,
                    text=task.text,
                    emotion_vec=task.emotion_vec.tolist(),
                    dominant_emotion=dominant_emotion,
                    control_params=task.control_vector.to_numpy().tolist(),
                    audio_path=str(rel_path),
                    audio_duration_seconds=validation.duration_seconds,
                    audio_file_size_bytes=validation.file_size_bytes,
                    sample_rate=validation.sample_rate,
                    style_id=task.style_id,
                    voicevox_params=result.voicevox_params or {},
                    is_valid=validation.is_valid,
                    generation_timestamp=datetime.now(tz=UTC).isoformat(),
                ),
            )

        elapsed = time.monotonic() - start_time

        # 5. データセット書き出し
        if records:
            dataset_dir = self._output_dir / "dataset"
            dataset_path = dataset_dir / "triplet_dataset.parquet"
            save_dataset(records, dataset_path)

            metadata = {
                "version": "1.0.0",
                "phase": "phase1",
                "created_at": datetime.now(tz=UTC).isoformat(),
                "config": self._config.to_dict(),
                "statistics": {
                    "num_texts": len(selected_texts),
                    "num_params_per_text": self._sampler.total_samples_per_text,
                    "total_records": len(records),
                    "valid_records": len(records) - invalid,
                    "invalid_records": invalid,
                    "total_audio_duration_seconds": total_audio_duration,
                    "total_audio_size_bytes": total_audio_size,
                    "dominant_emotion_distribution": emotion_dist,
                },
                "phase0_checkpoint": self._config.phase0_checkpoint,
            }
            save_metadata(metadata, dataset_dir / "metadata.json")

        # 6. レポート生成
        avg_synthesis_time = (
            (sum(synthesis_times) / len(synthesis_times))
            if synthesis_times
            else (elapsed / max(completed, 1))
        )

        report = GenerationReport(
            total_tasks=total_tasks,
            completed=completed,
            failed=failed,
            skipped=skipped_count,
            invalid=invalid,
            elapsed_seconds=elapsed,
            avg_synthesis_time_seconds=avg_synthesis_time,
            total_audio_duration_seconds=total_audio_duration,
            total_audio_size_bytes=total_audio_size,
            dominant_emotion_distribution=emotion_dist,
            failure_reasons=failure_reasons,
            config_snapshot=self._config.to_dict(),
        )

        # レポート保存
        reports_dir = self._output_dir / "reports"
        reports_dir.mkdir(parents=True, exist_ok=True)
        report_path = reports_dir / "generation_report.json"
        with report_path.open("w", encoding="utf-8") as f:
            json.dump(report.to_dict(), f, ensure_ascii=False, indent=2)

        logger.info(
            "生成完了: %d/%d 成功, %d 失敗, %d スキップ, %d 無効 (%.1f秒)",
            completed,
            total_tasks,
            failed,
            skipped_count,
            invalid,
            elapsed,
        )

        return report

    async def _generate_one(self, task: GenerationTask) -> GenerationResult:
        """単一タスクの音声生成。

        VoicevoxClientでaudio_queryを取得し、VoicevoxAdapterで制御パラメータを
        適用した後、synthesisでWAVを生成して保存する。

        Args:
            task: 生成タスク。

        Returns:
            GenerationResult。

        """
        try:
            # audio_query取得
            audio_query = await self._client.audio_query(
                text=task.text,
                speaker_id=task.style_id,
            )

            # 制御パラメータ適用
            modified_query = self._adapter.apply(audio_query, task.control_vector)

            # VOICEVOX実パラメータを記録
            voicevox_params = self._extract_voicevox_params(modified_query)

            # 音声合成
            wav_data = await self._client.synthesis(
                audio_query=modified_query,
                speaker_id=task.style_id,
            )

            # ファイル保存
            task.output_path.parent.mkdir(parents=True, exist_ok=True)
            task.output_path.write_bytes(wav_data)

            # 検証結果から duration と size を取得
            file_size = task.output_path.stat().st_size

            return GenerationResult(
                task=task,
                success=True,
                audio_file_size_bytes=file_size,
                voicevox_params=voicevox_params,
            )

        except Exception as e:
            logger.warning(
                "タスク %s 失敗: %s",
                task.task_id,
                e,
            )
            return GenerationResult(
                task=task,
                success=False,
                error=str(e),
            )

    async def _generate_batch(
        self,
        tasks: list[GenerationTask],
    ) -> list[GenerationResult]:
        """Semaphore で同時実行数を制御しつつ非同期バッチ生成。

        checkpoint_interval ごとにチェックポイントを保存し、
        進捗をログ出力する。

        Args:
            tasks: 生成タスクのリスト。

        Returns:
            GenerationResultのリスト。

        """
        if not tasks:
            return []

        semaphore = asyncio.Semaphore(self._config.generation.max_concurrent_requests)
        completed_ids: set[str] = set()
        results: list[GenerationResult] = []
        results_lock = asyncio.Lock()
        total = len(tasks)

        async def _run_one(task: GenerationTask) -> GenerationResult:
            async with semaphore:
                result = await self._generate_one(task)
                async with results_lock:
                    results.append(result)
                    if result.success:
                        completed_ids.add(task.task_id)
                    done = len(results)
                    # 進捗ログ
                    if done % self._config.generation.checkpoint_interval == 0:
                        logger.info(
                            "進捗: %d/%d (%.1f%%)",
                            done,
                            total,
                            done / total * 100,
                        )
                        self._save_checkpoint_data(completed_ids)
                return result

        await asyncio.gather(*[_run_one(t) for t in tasks])

        # 最終チェックポイント保存
        self._save_checkpoint_data(completed_ids)

        return results

    def _save_checkpoint(self) -> None:
        """チェックポイントを保存する（互換性のためのラッパー）。"""
        self._save_checkpoint_data(set())

    def _save_checkpoint_data(self, completed_ids: set[str]) -> None:
        """進捗状態をチェックポイントファイルに保存する。

        Args:
            completed_ids: 完了したタスクIDの集合。

        """
        checkpoint_dir = self._output_dir / "checkpoints"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        checkpoint_path = checkpoint_dir / "generation_progress.json"

        data = {
            "version": "1.0.0",
            "completed_task_ids": sorted(completed_ids),
            "total_completed": len(completed_ids),
            "last_updated": datetime.now(tz=UTC).isoformat(),
        }

        with checkpoint_path.open("w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    def _load_checkpoint(self) -> set[str] | None:
        """既存のチェックポイントを読み込む。

        Returns:
            完了済みタスクIDの集合。チェックポイントが存在しない場合はNone。

        """
        checkpoint_path = self._output_dir / "checkpoints" / "generation_progress.json"
        if not checkpoint_path.exists():
            return None

        try:
            with checkpoint_path.open("r", encoding="utf-8") as f:
                data = json.load(f)
            completed = set(data.get("completed_task_ids", []))
            logger.info(
                "チェックポイント読み込み: %d タスク完了済み",
                len(completed),
            )
            return completed
        except Exception as e:
            logger.warning("チェックポイント読み込み失敗: %s", e)
            return None

    @staticmethod
    def _get_dominant_emotion(emotion_vec: np.ndarray) -> str:
        """感情ベクトルからdominant emotionを取得する。

        Args:
            emotion_vec: shape (8,) の感情ベクトル。

        Returns:
            dominant emotionのラベル。

        """
        from emotionbridge.constants import EMOTION_LABELS

        dominant_idx = int(np.argmax(emotion_vec))
        return EMOTION_LABELS[dominant_idx]

    @staticmethod
    def _extract_voicevox_params(audio_query: AudioQuery) -> dict[str, float]:
        """AudioQueryからVOICEVOXの実パラメータを抽出する。

        Args:
            audio_query: VoicevoxAdapter.apply() の返値。

        Returns:
            パラメータ名→値の辞書。

        """
        params: dict[str, float] = {}
        for attr in [
            "speedScale",
            "pitchScale",
            "intonationScale",
            "volumeScale",
            "prePhonemeLength",
            "postPhonemeLength",
            "pauseLengthScale",
        ]:
            val = getattr(audio_query, attr, None)
            if val is not None:
                params[attr] = float(val)
        return params
