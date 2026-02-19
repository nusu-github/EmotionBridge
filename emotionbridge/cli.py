import argparse
import asyncio
import json
import logging
from pathlib import Path

from emotionbridge.config import (
    AudioGenConfig,
    ClassifierConfig,
    load_config,
)
from emotionbridge.data import (
    build_classifier_data_report,
    build_classifier_splits,
)
from emotionbridge.inference import EmotionEncoder
from emotionbridge.training import train_classifier

logger = logging.getLogger(__name__)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="emotionbridge",
        description="EmotionBridge CLI",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # --- 分類器コマンド ---

    train_parser = subparsers.add_parser(
        "train",
        help="Train text emotion classifier",
    )
    train_parser.add_argument(
        "--config",
        default="configs/classifier.yaml",
        help="Path to YAML config",
    )

    analyze_parser = subparsers.add_parser(
        "analyze-data",
        help="Analyze WRIME data and split statistics",
    )
    analyze_parser.add_argument(
        "--config",
        default="configs/classifier.yaml",
        help="Path to YAML config",
    )
    analyze_parser.add_argument(
        "--output-dir",
        default="artifacts/classifier/reports",
        help="Directory for plots and report",
    )

    encode_parser = subparsers.add_parser(
        "encode",
        help="Run inference from a trained checkpoint",
    )
    encode_parser.add_argument(
        "--checkpoint",
        required=True,
        help="Path to checkpoint directory",
    )
    encode_parser.add_argument("--text", required=True, help="Input text")
    encode_parser.add_argument("--device", default="cuda", help="cuda or cpu")

    # --- 音声生成コマンド ---

    gen_parser = subparsers.add_parser(
        "generate-samples",
        help="Generate audio samples with parameter grid search",
    )
    gen_parser.add_argument(
        "--config",
        default="configs/audio_gen.yaml",
        help="Path to audio generation YAML config",
    )

    speakers_parser = subparsers.add_parser(
        "list-speakers",
        help="List available VOICEVOX speakers and styles",
    )
    speakers_parser.add_argument(
        "--config",
        default="configs/audio_gen.yaml",
        help="Path to audio generation YAML config",
    )

    bridge_parser = subparsers.add_parser(
        "bridge",
        help="Synthesize emotional speech with classifier+generator+style mapping",
    )
    bridge_parser.add_argument("--text", required=True, help="Input text")
    bridge_parser.add_argument(
        "--output",
        default="output.wav",
        help="Output WAV file path",
    )
    bridge_parser.add_argument(
        "--character",
        default="zundamon",
        help="Character key in style mapping JSON (e.g., zundamon)",
    )
    bridge_parser.add_argument(
        "--classifier-checkpoint",
        default="artifacts/classifier/checkpoints/best_model",
        help="Path to classifier checkpoint directory",
    )
    bridge_parser.add_argument(
        "--generator-checkpoint",
        default="artifacts/generator/checkpoints/best_generator.pt",
        help="Path to generator checkpoint",
    )
    bridge_parser.add_argument(
        "--style-mapping",
        default="artifacts/prosody/style_mapping.json",
        help="Path to style mapping JSON",
    )
    bridge_parser.add_argument(
        "--voicevox-url",
        default="http://127.0.0.1:50021",
        help="VOICEVOX Engine URL",
    )
    bridge_parser.add_argument(
        "--fallback-threshold",
        type=float,
        default=0.3,
        help="Fallback threshold for max emotion probability",
    )
    bridge_parser.add_argument("--device", default="cuda", help="cuda or cpu")

    return parser


def _cmd_train(config_path: str) -> None:
    config = load_config(config_path)
    if not isinstance(config, ClassifierConfig):
        msg = f"Expected ClassifierConfig but got {type(config).__name__}. Check config file."
        raise SystemExit(msg)

    result = train_classifier(config)

    if result is not None:
        print(json.dumps(result, ensure_ascii=False, indent=2))


def _cmd_analyze_data(config_path: str, output_dir: str) -> None:
    config = load_config(config_path)
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    if not isinstance(config, ClassifierConfig):
        msg = f"Expected ClassifierConfig but got {type(config).__name__}. Check config file."
        raise SystemExit(msg)

    splits, metadata = build_classifier_splits(config.data)
    report = build_classifier_data_report(splits, metadata)

    with (out / "data_report.json").open("w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"\nReport saved to {out / 'data_report.json'}")


def _cmd_encode(checkpoint: str, text: str, device: str) -> None:
    encoder = EmotionEncoder(checkpoint, device=device)
    vector = encoder.encode(text)
    payload = {
        label: float(value) for label, value in zip(encoder.label_names, vector, strict=False)
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2))


def _cmd_generate_samples(config_path: str) -> None:
    from emotionbridge.config import save_effective_config
    from emotionbridge.generation.pipeline import GenerationPipeline
    from emotionbridge.generation.text_selector import load_jvnv_texts

    config = load_config(config_path)
    if not isinstance(config, AudioGenConfig):
        msg = f"Expected AudioGenConfig but got {type(config).__name__}. Check config file."
        raise SystemExit(msg)

    # 設定スナップショット保存
    out_dir = Path(config.generation.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    save_effective_config(config, out_dir / "effective_config.yaml")

    # JVNVテキスト読み込み
    encoder = EmotionEncoder(config.classifier_checkpoint, device=config.device)
    selected_texts = load_jvnv_texts(
        config.text_selection.jvnv_transcription_path, encoder, config.text_selection
    )

    # パイプライン実行
    pipeline = GenerationPipeline(config)

    async def _run() -> None:
        async with pipeline._client:
            report = await pipeline.run(selected_texts)
        print(json.dumps(report.to_dict(), ensure_ascii=False, indent=2))

    asyncio.run(_run())


def _cmd_list_speakers(config_path: str) -> None:
    from emotionbridge.tts.voicevox_client import VoicevoxClient

    config = load_config(config_path)
    if not isinstance(config, AudioGenConfig):
        msg = f"Expected AudioGenConfig but got {type(config).__name__}. Check config file."
        raise SystemExit(msg)

    async def _run() -> None:
        async with VoicevoxClient(
            base_url=config.voicevox.base_url,
            timeout=config.voicevox.timeout,
        ) as client:
            ok = await client.health_check()
            if not ok:
                msg = f"VOICEVOX Engine に接続できません: {config.voicevox.base_url}"
                raise SystemExit(
                    msg,
                )

            speakers = await client.speakers()

        print(f"利用可能なキャラクター: {len(speakers)}件\n")
        for speaker in speakers:
            talk_styles = [s for s in speaker.styles if s.style_type == "talk"]
            if not talk_styles:
                continue
            print(f"  {speaker.name} (UUID: {speaker.speaker_uuid})")
            for style in talk_styles:
                print(f"    - {style.name} (ID: {style.id})")
            print()

    asyncio.run(_run())


def _cmd_bridge(
    text: str,
    output: str,
    character: str,
    classifier_checkpoint: str,
    generator_checkpoint: str,
    style_mapping: str,
    voicevox_url: str,
    fallback_threshold: float,
    device: str,
) -> None:
    from emotionbridge.inference import create_pipeline

    async def _run() -> None:
        pipeline = await create_pipeline(
            classifier_checkpoint=classifier_checkpoint,
            generator_checkpoint=generator_checkpoint,
            style_mapping=style_mapping,
            voicevox_url=voicevox_url,
            character=character,
            fallback_threshold=fallback_threshold,
            device=device,
        )
        try:
            result = await pipeline.synthesize(text=text, output_path=output)
        finally:
            await pipeline.close()

        payload = {
            "audio_path": str(result.audio_path) if result.audio_path else None,
            "dominant_emotion": result.dominant_emotion,
            "emotion_probs": result.emotion_probs,
            "control_params": result.control_params,
            "style_id": result.style_id,
            "style_name": result.style_name,
            "confidence": result.confidence,
            "is_fallback": result.is_fallback,
            "metadata": result.metadata,
        }
        print(json.dumps(payload, ensure_ascii=False, indent=2))

    asyncio.run(_run())


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    parser = _build_parser()
    args = parser.parse_args()

    if args.command == "train":
        _cmd_train(args.config)
        return
    if args.command == "analyze-data":
        _cmd_analyze_data(args.config, args.output_dir)
        return
    if args.command == "encode":
        _cmd_encode(args.checkpoint, args.text, args.device)
        return
    if args.command == "generate-samples":
        _cmd_generate_samples(args.config)
        return
    if args.command == "list-speakers":
        _cmd_list_speakers(args.config)
        return
    if args.command == "bridge":
        _cmd_bridge(
            args.text,
            args.output,
            args.character,
            args.classifier_checkpoint,
            args.generator_checkpoint,
            args.style_mapping,
            args.voicevox_url,
            args.fallback_threshold,
            args.device,
        )
        return
    parser.error("Unknown command")
