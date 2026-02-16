"""Phase 2 Approach B+ の三つ組スコア付与パイプライン。"""

import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from emotionbridge.alignment.category_mapper import cosine_similarity_common6
from emotionbridge.analysis.emotion2vec import EMOTION2VEC_LABELS, Emotion2vecExtractor
from emotionbridge.config import Phase2TripletConfig, save_effective_config
from emotionbridge.generation.dataset import load_dataset, save_dataset

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class TripletScoringSummary:
    total_records: int
    scored_records: int
    missing_audio_records: int
    output_dataset_path: str
    output_embeddings_path: str
    output_metadata_path: str

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


class Phase2TripletScorer:
    """Phase 1 生成済みデータセットへ SER スコアを付与する。"""

    def __init__(self, config: Phase2TripletConfig) -> None:
        self._config = config

    def _resolve_dataset_path(self) -> Path:
        raw_path = Path(self._config.phase1_dataset_path)

        candidates: list[Path] = [raw_path]
        if not raw_path.is_absolute():
            candidates.append(Path.cwd() / raw_path)

        checked: list[Path] = []
        seen: set[str] = set()
        for candidate in candidates:
            key = str(candidate)
            if key in seen:
                continue
            seen.add(key)
            checked.append(candidate)
            if candidate.exists():
                return candidate

        discovered = list(Path.cwd().glob("artifacts/**/triplet_dataset.parquet"))
        if discovered:
            best = max(discovered, key=lambda path: path.stat().st_mtime)
            logger.warning(
                "phase1_dataset_path が見つからなかったため、自動検出結果を使用します: %s",
                best,
            )
            return best

        checked_text = "\n".join(f"- {path}" for path in checked)
        msg = (
            "Phase 1 dataset not found.\n"
            f"configured phase1_dataset_path: {self._config.phase1_dataset_path}\n"
            f"checked paths:\n{checked_text}\n"
            "まず Phase 1 のデータ生成を実行してください:\n"
            "  uv run python main.py generate-samples --config configs/phase1.yaml\n"
            "その後、score-triplets を再実行してください。"
        )
        raise FileNotFoundError(msg)

    @staticmethod
    def _resolve_audio_path(
        audio_path: str,
        *,
        phase1_output_dir: Path,
        dataset_root: Path,
    ) -> Path | None:
        path = Path(audio_path)
        if path.is_absolute():
            return path if path.exists() else None

        candidates = [
            phase1_output_dir / path,
            dataset_root / path,
            Path.cwd() / path,
        ]

        for candidate in candidates:
            if candidate.exists():
                return candidate
        return None

    def run(self) -> TripletScoringSummary:
        dataset_path = self._resolve_dataset_path()

        records = load_dataset(dataset_path)
        if self._config.max_records is not None:
            records = records[: self._config.max_records]

        total_records = len(records)
        if total_records == 0:
            msg = "No records found in input dataset"
            raise ValueError(msg)

        output_dir = Path(self._config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        save_effective_config(self._config, output_dir / "effective_config.yaml")

        phase1_output_dir = Path(self._config.phase1_output_dir)
        if not phase1_output_dir.exists():
            inferred_phase1_dir = dataset_path.parent.parent
            if inferred_phase1_dir.exists():
                logger.warning(
                    "phase1_output_dir=%s が存在しないため、推定パスを使用します: %s",
                    phase1_output_dir,
                    inferred_phase1_dir,
                )
                phase1_output_dir = inferred_phase1_dir

        dataset_root = dataset_path.parent.parent
        resolved_paths: list[str] = []
        valid_indices: list[int] = []
        missing_indices: list[int] = []

        for idx, record in enumerate(records):
            resolved = self._resolve_audio_path(
                record.audio_path,
                phase1_output_dir=phase1_output_dir,
                dataset_root=dataset_root,
            )

            if resolved is not None:
                valid_indices.append(idx)
                resolved_paths.append(str(resolved))
            else:
                missing_indices.append(idx)

        logger.info(
            "Triplet scoring input: total=%d, valid_audio=%d, missing_audio=%d",
            total_records,
            len(valid_indices),
            len(missing_indices),
        )

        feats = np.full((total_records, 1024), np.nan, dtype=np.float32)
        logits = np.full(
            (total_records, len(EMOTION2VEC_LABELS)),
            np.nan,
            dtype=np.float32,
        )

        scored_records = 0
        if valid_indices:
            extractor = Emotion2vecExtractor(
                model_id=self._config.ser_model,
                device=self._config.device,
            )
            extracted = extractor.extract(
                resolved_paths,
                batch_size=self._config.batch_size,
            )

            extracted_feats = extracted["feats"].astype(np.float32, copy=False)
            extracted_logits = extracted["logits"].astype(np.float32, copy=False)

            if extracted_feats.shape[0] != len(valid_indices):
                msg = (
                    "Mismatch between extracted embeddings and valid inputs: "
                    f"{extracted_feats.shape[0]} vs {len(valid_indices)}"
                )
                raise RuntimeError(msg)

            index_array = np.asarray(valid_indices, dtype=np.int64)
            feats[index_array] = extracted_feats
            logits[index_array] = extracted_logits

            for offset, record_idx in enumerate(valid_indices):
                score = cosine_similarity_common6(
                    records[record_idx].emotion_vec,
                    extracted_logits[offset],
                )
                records[record_idx].ser_score = score
                scored_records += 1

        for idx in missing_indices:
            records[idx].ser_score = None

        output_dataset_path = output_dir / "triplet_dataset_scored.parquet"
        output_embeddings_path = output_dir / "ser_embeddings.npz"
        output_metadata_path = output_dir / "metadata.json"

        save_dataset(records, output_dataset_path)
        np.savez_compressed(
            output_embeddings_path,
            feats=feats,
            logits=logits,
        )

        metadata = {
            "version": "1.0.0",
            "phase": "phase2_triplet_scoring",
            "input_dataset_path": str(dataset_path),
            "phase1_output_dir": str(phase1_output_dir),
            "ser_model": self._config.ser_model,
            "total_records": total_records,
            "scored_records": scored_records,
            "missing_audio_records": len(missing_indices),
            "embedding_order": "row-aligned with triplet_dataset_scored.parquet",
            "embedding_shapes": {
                "feats": [total_records, 1024],
                "logits": [total_records, len(EMOTION2VEC_LABELS)],
            },
            "emotion2vec_labels": EMOTION2VEC_LABELS,
        }
        with output_metadata_path.open("w", encoding="utf-8") as file:
            json.dump(metadata, file, ensure_ascii=False, indent=2)

        return TripletScoringSummary(
            total_records=total_records,
            scored_records=scored_records,
            missing_audio_records=len(missing_indices),
            output_dataset_path=str(output_dataset_path),
            output_embeddings_path=str(output_embeddings_path),
            output_metadata_path=str(output_metadata_path),
        )
