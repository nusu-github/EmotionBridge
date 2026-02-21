import argparse
import asyncio
import csv
import json
import logging
import random
from pathlib import Path
from typing import Any

import pandas as pd

from emotionbridge.constants import JVNV_EMOTION_LABELS
from emotionbridge.inference import RuleBasedStyleSelector, create_pipeline
from emotionbridge.tts.voicevox_client import VoicevoxClient

logger = logging.getLogger(__name__)


def _configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="パイロット主観評価用の音声刺激と評価シートを生成する",
    )
    parser.add_argument(
        "--dataset-path",
        default="artifacts/audio_gen_multistyle_smoke/dataset/triplet_dataset.parquet",
        help="テキスト候補を読むデータセットparquet",
    )
    parser.add_argument(
        "--output-dir",
        default="artifacts/prosody/subjective_eval/pilot",
        help="主観評価成果物の出力先",
    )
    parser.add_argument(
        "--character",
        default="zundamon",
        help="style_mapping.json内のキャラクターキー",
    )
    parser.add_argument(
        "--classifier-checkpoint",
        default="artifacts/classifier/checkpoints/best_model",
        help="分類器チェックポイント",
    )
    parser.add_argument(
        "--generator-checkpoint",
        default="artifacts/generator/checkpoints/best_generator.pt",
        help="生成器チェックポイント",
    )
    parser.add_argument(
        "--style-mapping",
        default="artifacts/prosody/style_mapping.json",
        help="スタイルマッピングJSON",
    )
    parser.add_argument(
        "--voicevox-url",
        default="http://127.0.0.1:50021",
        help="VOICEVOX Engine URL",
    )
    parser.add_argument(
        "--device",
        default="cuda",
        help="cuda or cpu",
    )
    parser.add_argument(
        "--fallback-threshold",
        type=float,
        default=0.3,
        help="bridgeフォールバック閾値",
    )
    parser.add_argument(
        "--enable-dsp",
        action="store_true",
        help="WORLDベースDSP後処理を有効化する",
    )
    parser.add_argument(
        "--dsp-features-path",
        default="artifacts/prosody/v01/jvnv_egemaps_normalized.parquet",
        help="EmotionDSPMapperの初期化に使うJVNV eGeMAPS parquet",
    )
    parser.add_argument(
        "--samples-per-emotion",
        type=int,
        default=1,
        help="感情ごとに抽出するテキスト件数",
    )
    parser.add_argument(
        "--target-emotions",
        default=",".join(JVNV_EMOTION_LABELS),
        help="対象感情（カンマ区切り、common6）",
    )
    parser.add_argument("--random-seed", type=int, default=42)
    return parser


def _canon_common6(label: str) -> str | None:
    lowered = str(label).strip().lower()
    mapping = {
        "anger": "anger",
        "angry": "anger",
        "ang": "anger",
        "disgust": "disgust",
        "disgusted": "disgust",
        "dis": "disgust",
        "fear": "fear",
        "fearful": "fear",
        "fea": "fear",
        "happy": "happy",
        "joy": "happy",
        "hap": "happy",
        "sad": "sad",
        "sadness": "sad",
        "surprise": "surprise",
        "surprised": "surprise",
        "sur": "surprise",
    }
    return mapping.get(lowered)


def _parse_target_emotions(raw: str) -> list[str]:
    values = [chunk.strip() for chunk in raw.split(",") if chunk.strip()]
    unique = list(dict.fromkeys(values))
    invalid = [emotion for emotion in unique if emotion not in JVNV_EMOTION_LABELS]
    if invalid:
        msg = f"Invalid target emotions: {invalid}. allowed={JVNV_EMOTION_LABELS}"
        raise ValueError(msg)
    return unique


def _pick_text_samples(
    *,
    dataset_path: Path,
    target_emotions: list[str],
    samples_per_emotion: int,
    random_seed: int,
) -> list[dict[str, str]]:
    if not dataset_path.exists():
        msg = f"dataset not found: {dataset_path}"
        raise FileNotFoundError(msg)

    df = pd.read_parquet(dataset_path)
    required = ["text", "dominant_emotion"]
    missing = [name for name in required if name not in df.columns]
    if missing:
        msg = f"dataset missing required columns: {missing}"
        raise ValueError(msg)

    dedup = df[["text", "dominant_emotion"]].drop_duplicates().copy()
    dedup["emotion_common6"] = dedup["dominant_emotion"].map(_canon_common6)
    dedup = dedup[dedup["emotion_common6"].notna()].reset_index(drop=True)

    rng = random.Random(random_seed)
    selected: list[dict[str, str]] = []

    for emotion in target_emotions:
        subset = dedup[dedup["emotion_common6"] == emotion]
        rows = subset.to_dict(orient="records")
        if len(rows) < samples_per_emotion:
            msg = f"Not enough text samples for emotion={emotion}: required={samples_per_emotion}, available={len(rows)}"
            raise RuntimeError(msg)

        rng.shuffle(rows)
        selected.extend(
            {
                "text": str(row["text"]),
                "target_emotion": emotion,
            }
            for row in rows[:samples_per_emotion]
        )

    return selected


def _write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


async def _synthesize_baseline(
    *,
    client: VoicevoxClient,
    text: str,
    style_id: int,
) -> bytes:
    query = await client.audio_query(text=text, speaker_id=style_id)
    return await client.synthesis(query, speaker_id=style_id)


def _make_instructions(output_path: Path) -> None:
    lines = [
        "# Subjective Evaluation Instructions (Pilot)",
        "",
        "- 対象人数: 5〜10名（パイロット）",
        "- 評価手法: A/Bテスト, MOS(5段階), 感情識別テスト",
        "",
        "## 1) A/Bテスト",
        "- `manifests/ab_stimuli.csv` を使い、A/Bのどちらが指定感情をより表現しているか回答する。",
        "- 回答は `responses/ab_responses_template.csv` を複製して記入する。",
        "",
        "## 2) MOS評価",
        "- `manifests/mos_stimuli.csv` の音声を聴取し、自然さ(1-5)と感情適合度(1-5)を評価する。",
        "- 回答は `responses/mos_responses_template.csv` に記入する。",
        "",
        "## 3) 感情識別テスト",
        "- `manifests/emotion_identification_stimuli.csv` の音声について、知覚した感情を6カテゴリから選ぶ。",
        "- 回答は `responses/emotion_identification_responses_template.csv` に記入する。",
        "",
        "## 4) 集計",
        "- 回答CSVを `responses/` に配置し、以下を実行:",
        "  `uv run python -m emotionbridge.scripts.analyze_subjective_eval --eval-dir <this_dir>`",
    ]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines), encoding="utf-8")


async def run_prepare(
    *,
    dataset_path: str,
    output_dir: str,
    character: str,
    classifier_checkpoint: str,
    generator_checkpoint: str,
    style_mapping: str,
    voicevox_url: str,
    device: str,
    fallback_threshold: float,
    enable_dsp: bool,
    dsp_features_path: str,
    samples_per_emotion: int,
    target_emotions_raw: str,
    random_seed: int,
) -> dict[str, Any]:
    target_emotions = _parse_target_emotions(target_emotions_raw)
    selected_texts = _pick_text_samples(
        dataset_path=Path(dataset_path),
        target_emotions=target_emotions,
        samples_per_emotion=samples_per_emotion,
        random_seed=random_seed,
    )

    output_root = Path(output_dir)
    audio_dir = output_root / "audio"
    manifests_dir = output_root / "manifests"
    responses_dir = output_root / "responses"
    audio_dir.mkdir(parents=True, exist_ok=True)
    manifests_dir.mkdir(parents=True, exist_ok=True)
    responses_dir.mkdir(parents=True, exist_ok=True)

    style_selector = RuleBasedStyleSelector(style_mapping)
    default_style_id, default_style_name = style_selector.default_style(character)

    rng = random.Random(random_seed)

    pipeline = await create_pipeline(
        classifier_checkpoint=classifier_checkpoint,
        generator_checkpoint=generator_checkpoint,
        style_mapping=style_mapping,
        voicevox_url=voicevox_url,
        character=character,
        fallback_threshold=fallback_threshold,
        enable_dsp=enable_dsp,
        dsp_features_path=dsp_features_path,
        device=device,
    )

    baseline_client = VoicevoxClient(base_url=voicevox_url)
    if not await baseline_client.health_check():
        await pipeline.close()
        await baseline_client.close()
        msg = f"VOICEVOX Engine に接続できません: {voicevox_url}"
        raise ConnectionError(msg)

    try:
        key_rows: list[dict[str, Any]] = []
        ab_rows: list[dict[str, Any]] = []
        mos_rows: list[dict[str, Any]] = []
        emotion_rows: list[dict[str, Any]] = []

        for index, sample in enumerate(selected_texts, start=1):
            sample_id = f"S{index:03d}"
            text = sample["text"]
            target_emotion = sample["target_emotion"]

            baseline_bytes = await _synthesize_baseline(
                client=baseline_client,
                text=text,
                style_id=default_style_id,
            )
            bridge_result = await pipeline.synthesize(text=text)

            baseline_path = audio_dir / f"{sample_id}_baseline.wav"
            bridge_path = audio_dir / f"{sample_id}_bridge.wav"
            baseline_path.write_bytes(baseline_bytes)
            bridge_path.write_bytes(bridge_result.audio_bytes)

            if rng.random() < 0.5:
                a_path = audio_dir / f"{sample_id}_A.wav"
                b_path = audio_dir / f"{sample_id}_B.wav"
                a_path.write_bytes(baseline_bytes)
                b_path.write_bytes(bridge_result.audio_bytes)
                condition_a = "baseline"
                condition_b = "emotionbridge"
            else:
                a_path = audio_dir / f"{sample_id}_A.wav"
                b_path = audio_dir / f"{sample_id}_B.wav"
                a_path.write_bytes(bridge_result.audio_bytes)
                b_path.write_bytes(baseline_bytes)
                condition_a = "emotionbridge"
                condition_b = "baseline"

            key_rows.append(
                {
                    "sample_id": sample_id,
                    "text": text,
                    "target_emotion": target_emotion,
                    "audio_a_path": str(a_path),
                    "audio_b_path": str(b_path),
                    "condition_a": condition_a,
                    "condition_b": condition_b,
                    "baseline_audio_path": str(baseline_path),
                    "bridge_audio_path": str(bridge_path),
                    "bridge_predicted_emotion": bridge_result.dominant_emotion,
                    "bridge_confidence": bridge_result.confidence,
                    "bridge_style_id": bridge_result.style_id,
                    "bridge_style_name": bridge_result.style_name,
                    "bridge_is_fallback": bridge_result.is_fallback,
                    "bridge_control_params": json.dumps(
                        bridge_result.control_params,
                        ensure_ascii=False,
                    ),
                },
            )

            ab_rows.append(
                {
                    "sample_id": sample_id,
                    "target_emotion": target_emotion,
                    "text": text,
                    "audio_a_path": str(a_path),
                    "audio_b_path": str(b_path),
                },
            )
            mos_rows.append(
                {
                    "sample_id": sample_id,
                    "target_emotion": target_emotion,
                    "text": text,
                    "audio_path": str(bridge_path),
                },
            )
            emotion_rows.append(
                {
                    "sample_id": sample_id,
                    "audio_path": str(bridge_path),
                },
            )

        _write_csv(
            manifests_dir / "stimuli_key.csv",
            key_rows,
            [
                "sample_id",
                "text",
                "target_emotion",
                "audio_a_path",
                "audio_b_path",
                "condition_a",
                "condition_b",
                "baseline_audio_path",
                "bridge_audio_path",
                "bridge_predicted_emotion",
                "bridge_confidence",
                "bridge_style_id",
                "bridge_style_name",
                "bridge_is_fallback",
                "bridge_control_params",
            ],
        )
        _write_csv(
            manifests_dir / "ab_stimuli.csv",
            ab_rows,
            [
                "sample_id",
                "target_emotion",
                "text",
                "audio_a_path",
                "audio_b_path",
            ],
        )
        _write_csv(
            manifests_dir / "mos_stimuli.csv",
            mos_rows,
            ["sample_id", "target_emotion", "text", "audio_path"],
        )
        _write_csv(
            manifests_dir / "emotion_identification_stimuli.csv",
            emotion_rows,
            ["sample_id", "audio_path"],
        )

        _write_csv(
            responses_dir / "ab_responses_template.csv",
            [],
            [
                "participant_id",
                "sample_id",
                "preferred",
                "confidence_1to5",
                "comment",
            ],
        )
        _write_csv(
            responses_dir / "mos_responses_template.csv",
            [],
            [
                "participant_id",
                "sample_id",
                "naturalness_mos_1to5",
                "emotion_fit_mos_1to5",
                "comment",
            ],
        )
        _write_csv(
            responses_dir / "emotion_identification_responses_template.csv",
            [],
            [
                "participant_id",
                "sample_id",
                "predicted_emotion",
                "confidence_1to5",
                "comment",
            ],
        )

        _make_instructions(output_root / "instructions.md")

        summary = {
            "output_dir": str(output_root),
            "num_samples": len(key_rows),
            "target_emotions": target_emotions,
            "samples_per_emotion": samples_per_emotion,
            "character": character,
            "default_style": {
                "style_id": default_style_id,
                "style_name": default_style_name,
            },
            "files": {
                "stimuli_key": str(manifests_dir / "stimuli_key.csv"),
                "ab_stimuli": str(manifests_dir / "ab_stimuli.csv"),
                "mos_stimuli": str(manifests_dir / "mos_stimuli.csv"),
                "emotion_identification_stimuli": str(
                    manifests_dir / "emotion_identification_stimuli.csv",
                ),
                "instructions": str(output_root / "instructions.md"),
            },
        }

        with (output_root / "summary.json").open("w", encoding="utf-8") as file:
            json.dump(summary, file, ensure_ascii=False, indent=2)

        return summary

    finally:
        await pipeline.close()
        await baseline_client.close()


def main() -> None:
    _configure_logging()
    args = _build_parser().parse_args()

    result = asyncio.run(
        run_prepare(
            dataset_path=args.dataset_path,
            output_dir=args.output_dir,
            character=args.character,
            classifier_checkpoint=args.classifier_checkpoint,
            generator_checkpoint=args.generator_checkpoint,
            style_mapping=args.style_mapping,
            voicevox_url=args.voicevox_url,
            device=args.device,
            fallback_threshold=args.fallback_threshold,
            enable_dsp=args.enable_dsp,
            dsp_features_path=args.dsp_features_path,
            samples_per_emotion=args.samples_per_emotion,
            target_emotions_raw=args.target_emotions,
            random_seed=args.random_seed,
        ),
    )
    logger.info("主観評価バッチ生成完了: %s", result["output_dir"])


if __name__ == "__main__":
    main()
