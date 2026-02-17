import argparse
import asyncio
import json
import logging
from pathlib import Path

from transformers import AutoTokenizer

from emotionbridge.config import (
    Phase0Config,
    Phase1Config,
    load_config,
)
from emotionbridge.data import (
    build_data_report,
    build_phase0_splits,
    estimate_unk_ratio,
    plot_cooccurrence_matrix,
    plot_emotion_means_comparison,
    plot_intensity_histograms,
)
from emotionbridge.inference import EmotionEncoder
from emotionbridge.training import train_phase0

logger = logging.getLogger(__name__)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="emotionbridge",
        description="EmotionBridge CLI",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # --- Phase 0 コマンド ---

    train_parser = subparsers.add_parser(
        "train",
        help="Train Phase 0 text emotion encoder",
    )
    train_parser.add_argument(
        "--config",
        default="configs/phase0.yaml",
        help="Path to YAML config",
    )

    analyze_parser = subparsers.add_parser(
        "analyze-data",
        help="Analyze WRIME data and split statistics",
    )
    analyze_parser.add_argument(
        "--config",
        default="configs/phase0.yaml",
        help="Path to YAML config",
    )
    analyze_parser.add_argument(
        "--output-dir",
        default="artifacts/phase0/reports",
        help="Directory for plots and report",
    )

    encode_parser = subparsers.add_parser(
        "encode",
        help="Run inference from a trained checkpoint",
    )
    encode_parser.add_argument(
        "--checkpoint",
        required=True,
        help="Path to checkpoint (.pt)",
    )
    encode_parser.add_argument("--text", required=True, help="Input text")
    encode_parser.add_argument("--device", default="cuda", help="cuda or cpu")
    encode_parser.add_argument(
        "--output-format",
        choices=["emotion8d", "av2d", "both"],
        default="emotion8d",
        help="Inference output format",
    )

    # --- Phase 1 コマンド ---

    gen_parser = subparsers.add_parser(
        "generate-samples",
        help="Generate audio samples with parameter grid search",
    )
    gen_parser.add_argument(
        "--config",
        default="configs/phase1.yaml",
        help="Path to Phase 1 YAML config",
    )

    speakers_parser = subparsers.add_parser(
        "list-speakers",
        help="List available VOICEVOX speakers and styles",
    )
    speakers_parser.add_argument(
        "--config",
        default="configs/phase1.yaml",
        help="Path to Phase 1 YAML config",
    )

    synth_parser = subparsers.add_parser(
        "synthesize",
        help="Synthesize emotional speech from text (heuristic mapping)",
    )
    synth_parser.add_argument(
        "--config",
        default="configs/phase1.yaml",
        help="Path to Phase 1 YAML config",
    )
    synth_parser.add_argument("--text", required=True, help="Input text")
    synth_parser.add_argument(
        "--output",
        default="output.wav",
        help="Output WAV file path",
    )
    synth_parser.add_argument(
        "--speaker-id",
        type=int,
        default=None,
        help="VOICEVOX speaker style ID (overrides config)",
    )
    synth_parser.add_argument(
        "--mapping-space",
        choices=["emotion8d", "av2d"],
        default="av2d",
        help="Control mapping space",
    )

    return parser


def _cmd_train(config_path: str) -> None:
    config = load_config(config_path)
    if not isinstance(config, Phase0Config):
        msg = (
            f"Expected Phase0Config but got {type(config).__name__}. Check config file."
        )
        raise SystemExit(msg)
    result = train_phase0(config)
    if result is not None:
        print(json.dumps(result, ensure_ascii=False, indent=2))


def _cmd_analyze_data(config_path: str, output_dir: str) -> None:
    config = load_config(config_path)
    if not isinstance(config, Phase0Config):
        msg = (
            f"Expected Phase0Config but got {type(config).__name__}. Check config file."
        )
        raise SystemExit(msg)
    splits, metadata = build_phase0_splits(config.data)
    tokenizer = AutoTokenizer.from_pretrained(config.model.pretrained_model_name)
    report = build_data_report(splits, metadata)
    report["unk_ratio_train"] = estimate_unk_ratio(tokenizer, splits["train"].texts)

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    with (out / "data_report.json").open("w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    plot_intensity_histograms(splits, out / "intensity_histograms.png")
    plot_cooccurrence_matrix(splits["train"], out / "cooccurrence_matrix.png")
    plot_emotion_means_comparison(splits, out / "emotion_means.png")

    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"\nPlots saved to {out}/")
    print("  - intensity_histograms.png")
    print("  - cooccurrence_matrix.png")
    print("  - emotion_means.png")


def _cmd_encode(checkpoint: str, text: str, device: str, output_format: str) -> None:
    from emotionbridge.constants import CIRCUMPLEX_AXIS_NAMES, EMOTION_LABELS

    encoder = EmotionEncoder(checkpoint, device=device)
    if output_format == "emotion8d":
        vector = encoder.encode(text)
        print(json.dumps(vector.tolist(), ensure_ascii=False))
        return

    if output_format == "av2d":
        av = encoder.encode_av(text)
        print(json.dumps(av.tolist(), ensure_ascii=False))
        return

    vector = encoder.encode(text)
    av = encoder.encode_av(text)
    payload = {
        "emotion8d": {
            label: float(value)
            for label, value in zip(EMOTION_LABELS, vector, strict=False)
        },
        "continuous_axes": {
            CIRCUMPLEX_AXIS_NAMES[0]: float(av[0]),
            CIRCUMPLEX_AXIS_NAMES[1]: float(av[1]),
        },
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2))


def _cmd_generate_samples(config_path: str) -> None:
    from emotionbridge.config import save_effective_config
    from emotionbridge.generation.pipeline import GenerationPipeline

    config = load_config(config_path)
    if not isinstance(config, Phase1Config):
        msg = (
            f"Expected Phase1Config but got {type(config).__name__}. Check config file."
        )
        raise SystemExit(msg)

    # 設定スナップショット保存
    out_dir = Path(config.generation.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    save_effective_config(config, out_dir / "effective_config.yaml")

    # WRIMEデータ読み込み
    splits, _ = build_phase0_splits(config.data)

    # パイプライン実行
    pipeline = GenerationPipeline(config)

    async def _run() -> None:
        async with pipeline._client:
            report = await pipeline.run(splits)
        print(json.dumps(report.to_dict(), ensure_ascii=False, indent=2))

    asyncio.run(_run())


def _cmd_list_speakers(config_path: str) -> None:
    from emotionbridge.tts.voicevox_client import VoicevoxClient

    config = load_config(config_path)
    if not isinstance(config, Phase1Config):
        msg = (
            f"Expected Phase1Config but got {type(config).__name__}. Check config file."
        )
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


def _cmd_synthesize(
    config_path: str,
    text: str,
    output: str,
    speaker_id: int | None,
    mapping_space: str,
) -> None:
    from emotionbridge.inference.axes import emotion8d_to_av
    from emotionbridge.tts.adapter import VoicevoxAdapter
    from emotionbridge.tts.heuristic_mapper import heuristic_map, heuristic_map_from_av
    from emotionbridge.tts.voicevox_client import VoicevoxClient

    config = load_config(config_path)
    if not isinstance(config, Phase1Config):
        msg = (
            f"Expected Phase1Config but got {type(config).__name__}. Check config file."
        )
        raise SystemExit(msg)

    # 感情エンコード
    encoder = EmotionEncoder(config.phase0_checkpoint, device=config.device)
    emotion_vec = encoder.encode(text)
    av = emotion8d_to_av(emotion_vec)

    # ヒューリスティックマッピング
    if mapping_space == "av2d":
        control = heuristic_map_from_av(arousal=float(av[0]), valence=float(av[1]))
    else:
        control = heuristic_map(emotion_vec)

    # 感情情報を表示
    from emotionbridge.constants import EMOTION_LABELS

    emotion_dict = {
        label: float(f"{v:.3f}")
        for label, v in zip(EMOTION_LABELS, emotion_vec, strict=False)
    }
    print(f"感情ベクトル: {json.dumps(emotion_dict, ensure_ascii=False)}")
    print(f"連続軸: arousal={av[0]:.3f}, valence={av[1]:.3f}")
    print(f"マッピング空間: {mapping_space}")
    print(
        f"制御ベクトル: pitch_shift={control.pitch_shift:.3f}, "
        f"pitch_range={control.pitch_range:.3f}, speed={control.speed:.3f}, "
        f"energy={control.energy:.3f}, pause_weight={control.pause_weight:.3f}",
    )

    # 音声合成
    spk_id = (
        speaker_id if speaker_id is not None else config.voicevox.default_speaker_id
    )
    adapter = VoicevoxAdapter(config.control_space)

    async def _run() -> None:
        async with VoicevoxClient(
            base_url=config.voicevox.base_url,
            timeout=config.voicevox.timeout,
            max_retries=config.voicevox.max_retries,
            retry_delay=config.voicevox.retry_delay,
        ) as client:
            query = await client.audio_query(text, spk_id)
            modified = adapter.apply(query, control)
            wav_data = await client.synthesis(modified, spk_id)

        out_path = Path(output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_bytes(wav_data)
        print(f"音声を保存しました: {out_path} ({len(wav_data)} bytes)")

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
        _cmd_encode(args.checkpoint, args.text, args.device, args.output_format)
        return
    if args.command == "generate-samples":
        _cmd_generate_samples(args.config)
        return
    if args.command == "list-speakers":
        _cmd_list_speakers(args.config)
        return
    if args.command == "synthesize":
        _cmd_synthesize(
            args.config,
            args.text,
            args.output,
            args.speaker_id,
            args.mapping_space,
        )
        return
    parser.error("Unknown command")
