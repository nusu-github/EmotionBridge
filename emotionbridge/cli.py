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
        "--generator-model-dir",
        default="artifacts/generator/checkpoints/best_generator",
        help="Path to generator model directory",
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
    bridge_parser.add_argument(
        "--enable-dsp",
        action="store_true",
        help="Enable WORLD-based DSP post-processing",
    )
    bridge_parser.add_argument(
        "--dsp-features-path",
        default="artifacts/prosody/v01/jvnv_egemaps_normalized.parquet",
        help="Path to JVNV eGeMAPS parquet used for DSP mapper initialization",
    )
    bridge_parser.add_argument(
        "--dsp-f0-extractor",
        choices=["dio", "harvest"],
        default="dio",
        help="F0 extractor for WORLD analysis in DSP stage",
    )
    bridge_parser.add_argument("--device", default="cuda", help="cuda or cpu")

    # --- 評価コマンド ---

    evaluate_parser = subparsers.add_parser(
        "evaluate",
        help="Run evaluation pipelines",
    )
    eval_subparsers = evaluate_parser.add_subparsers(dest="eval_command", required=True)

    eval_roundtrip_parser = eval_subparsers.add_parser(
        "roundtrip",
        help="Roundtrip定量評価（PESQ/MCD/F0 RMSE）",
    )
    eval_roundtrip_parser.add_argument(
        "--baseline-manifest",
        default="demo/v2/manifest.json",
        help="baseline側 manifest JSON",
    )
    eval_roundtrip_parser.add_argument(
        "--candidate-manifest",
        default="demo/v2-dsp/manifest.json",
        help="candidate側 manifest JSON",
    )
    eval_roundtrip_parser.add_argument(
        "--output-dir",
        default="artifacts/prosody/roundtrip_eval/v2_dsp",
        help="評価レポート出力先ディレクトリ",
    )

    eval_resp_parser = eval_subparsers.add_parser(
        "responsiveness",
        help="V-02 VOICEVOX 韻律応答性評価",
    )
    eval_resp_parser.add_argument(
        "--config",
        default="configs/experiment_config.yaml",
        help="実験設定ファイル",
    )
    eval_resp_parser.add_argument(
        "--input-path",
        default=None,
        help="正規化済みVOICEVOX特徴量parquet",
    )
    eval_resp_parser.add_argument(
        "--gate-policy",
        choices=["feature_only", "feature_and_av"],
        default="feature_only",
        help="Go/No-Go 判定ポリシー",
    )

    eval_domain_gap_parser = eval_subparsers.add_parser(
        "domain-gap",
        help="V-03 ドメインギャップ評価",
    )
    eval_domain_gap_parser.add_argument(
        "--config",
        default="configs/experiment_config.yaml",
        help="実験設定ファイル",
    )
    eval_domain_gap_parser.add_argument("--jvnv-normalized", default=None, help="JVNV 正規化特徴量")
    eval_domain_gap_parser.add_argument(
        "--voicevox-normalized",
        default=None,
        help="VOICEVOX 正規化特徴量",
    )
    eval_domain_gap_parser.add_argument("--jvnv-raw", default=None, help="JVNV 生特徴量")
    eval_domain_gap_parser.add_argument("--voicevox-raw", default=None, help="VOICEVOX 生特徴量")

    eval_sep_parser = eval_subparsers.add_parser(
        "separation",
        help="V-01 感情分離性評価",
    )
    eval_sep_parser.add_argument(
        "--config",
        default="configs/experiment_config.yaml",
        help="実験設定ファイル",
    )
    eval_sep_parser.add_argument(
        "--input-path",
        default=None,
        help="正規化済みJVNV特徴量parquet",
    )
    eval_sep_parser.add_argument(
        "--with-nv-input-path",
        default=None,
        help="NV除外なし版（比較用、任意）",
    )
    eval_sep_parser.add_argument(
        "--permutations",
        type=int,
        default=499,
        help="PERMANOVA permutation回数",
    )

    eval_axes_parser = eval_subparsers.add_parser(
        "continuous-axes",
        help="V-01 連続軸（Arousal/Valence）評価",
    )
    eval_axes_parser.add_argument(
        "--config",
        default="configs/experiment_config.yaml",
        help="実験設定ファイル",
    )
    eval_axes_parser.add_argument(
        "--input-path",
        default=None,
        help="JVNV 正規化特徴量 parquet",
    )
    eval_axes_parser.add_argument(
        "--anchors-json",
        default=None,
        help="感情アンカー値を上書きする JSON ファイル",
    )
    eval_axes_parser.add_argument(
        "--arousal-r2-threshold",
        type=float,
        default=0.30,
        help="Conditional Go 判定の arousal R2 下限",
    )
    eval_axes_parser.add_argument(
        "--valence-r2-threshold",
        type=float,
        default=0.15,
        help="Conditional Go 判定の valence R2 下限",
    )

    eval_style_influence_parser = eval_subparsers.add_parser(
        "style-influence",
        help="V-03 style_id主効果/交互作用評価",
    )
    eval_style_influence_parser.add_argument(
        "--config",
        default="configs/experiment_config.yaml",
        help="実験設定ファイル",
    )
    eval_style_influence_parser.add_argument(
        "--voicevox-raw",
        default=None,
        help="VOICEVOX生特徴量parquet",
    )
    eval_style_influence_parser.add_argument(
        "--jvnv-normalized",
        default=None,
        help="JVNV正規化特徴量parquet",
    )
    eval_style_influence_parser.add_argument(
        "--target-style-ids",
        default=None,
        help="対象style_idをカンマ区切りで指定",
    )
    eval_style_influence_parser.add_argument(
        "--output-dir",
        default=None,
        help="出力先ディレクトリ（未指定時はv03）",
    )

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
        config.text_selection.jvnv_transcription_path,
        encoder,
        config.text_selection,
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
    generator_model_dir: str,
    style_mapping: str,
    voicevox_url: str,
    fallback_threshold: float,
    enable_dsp: bool,
    dsp_features_path: str,
    dsp_f0_extractor: str,
    device: str,
) -> None:
    from emotionbridge.inference import create_pipeline

    async def _run() -> None:
        pipeline = await create_pipeline(
            classifier_checkpoint=classifier_checkpoint,
            generator_model_dir=generator_model_dir,
            style_mapping=style_mapping,
            voicevox_url=voicevox_url,
            character=character,
            fallback_threshold=fallback_threshold,
            enable_dsp=enable_dsp,
            dsp_features_path=dsp_features_path,
            dsp_f0_extractor=dsp_f0_extractor,
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
            "dsp_params": result.dsp_params,
            "dsp_applied": result.dsp_applied,
            "dsp_seed": result.dsp_seed,
            "style_id": result.style_id,
            "style_name": result.style_name,
            "confidence": result.confidence,
            "is_fallback": result.is_fallback,
            "metadata": result.metadata,
        }
        print(json.dumps(payload, ensure_ascii=False, indent=2))

    asyncio.run(_run())


def _cmd_evaluate_roundtrip(
    baseline_manifest: str,
    candidate_manifest: str,
    output_dir: str,
) -> None:
    from emotionbridge.scripts.evaluate_roundtrip import run_evaluation

    result = run_evaluation(
        baseline_manifest=baseline_manifest,
        candidate_manifest=candidate_manifest,
        output_dir=output_dir,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


def _cmd_evaluate_responsiveness(
    config: str,
    input_path: str | None,
    gate_policy: str,
) -> None:
    from emotionbridge.scripts.evaluate_responsiveness import run_evaluation

    result = run_evaluation(config, input_path, gate_policy)
    print(json.dumps(result, ensure_ascii=False, indent=2))


def _cmd_evaluate_domain_gap(
    config: str,
    jvnv_normalized: str | None,
    voicevox_normalized: str | None,
    jvnv_raw: str | None,
    voicevox_raw: str | None,
) -> None:
    from emotionbridge.scripts.evaluate_domain_gap import run_evaluation

    result = run_evaluation(
        config_path=config,
        jvnv_normalized=jvnv_normalized,
        voicevox_normalized=voicevox_normalized,
        jvnv_raw=jvnv_raw,
        voicevox_raw=voicevox_raw,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


def _cmd_evaluate_separation(
    config: str,
    input_path: str | None,
    with_nv_input_path: str | None,
    permutations: int,
) -> None:
    from emotionbridge.scripts.evaluate_separation import run_evaluation

    result = run_evaluation(
        config_path=config,
        input_path=input_path,
        with_nv_input_path=with_nv_input_path,
        permutations=permutations,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


def _cmd_evaluate_continuous_axes(
    config: str,
    input_path: str | None,
    anchors_json: str | None,
    arousal_r2_threshold: float,
    valence_r2_threshold: float,
) -> None:
    from emotionbridge.scripts.evaluate_continuous_axes import run_evaluation

    result = run_evaluation(
        config_path=config,
        input_path=input_path,
        anchors_json=anchors_json,
        arousal_r2_threshold=arousal_r2_threshold,
        valence_r2_threshold=valence_r2_threshold,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


def _cmd_evaluate_style_influence(
    config: str,
    voicevox_raw: str | None,
    jvnv_normalized: str | None,
    target_style_ids: str | None,
    output_dir: str | None,
) -> None:
    from emotionbridge.scripts.evaluate_style_influence import run_evaluation

    result = run_evaluation(
        config_path=config,
        voicevox_raw=voicevox_raw,
        jvnv_normalized=jvnv_normalized,
        target_style_ids_raw=target_style_ids,
        output_dir=output_dir,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


def _dispatch_evaluate(args: argparse.Namespace, parser: argparse.ArgumentParser) -> None:
    handlers = {
        "roundtrip": lambda: _cmd_evaluate_roundtrip(
            args.baseline_manifest,
            args.candidate_manifest,
            args.output_dir,
        ),
        "responsiveness": lambda: _cmd_evaluate_responsiveness(
            args.config,
            args.input_path,
            args.gate_policy,
        ),
        "domain-gap": lambda: _cmd_evaluate_domain_gap(
            args.config,
            args.jvnv_normalized,
            args.voicevox_normalized,
            args.jvnv_raw,
            args.voicevox_raw,
        ),
        "separation": lambda: _cmd_evaluate_separation(
            args.config,
            args.input_path,
            args.with_nv_input_path,
            args.permutations,
        ),
        "continuous-axes": lambda: _cmd_evaluate_continuous_axes(
            args.config,
            args.input_path,
            args.anchors_json,
            args.arousal_r2_threshold,
            args.valence_r2_threshold,
        ),
        "style-influence": lambda: _cmd_evaluate_style_influence(
            args.config,
            args.voicevox_raw,
            args.jvnv_normalized,
            args.target_style_ids,
            args.output_dir,
        ),
    }

    handler = handlers.get(args.eval_command)
    if handler is None:
        parser.error("Unknown evaluate command")
        return
    handler()


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    parser = _build_parser()
    args = parser.parse_args()

    handlers = {
        "train": lambda: _cmd_train(args.config),
        "analyze-data": lambda: _cmd_analyze_data(args.config, args.output_dir),
        "encode": lambda: _cmd_encode(args.checkpoint, args.text, args.device),
        "generate-samples": lambda: _cmd_generate_samples(args.config),
        "list-speakers": lambda: _cmd_list_speakers(args.config),
        "bridge": lambda: _cmd_bridge(
            args.text,
            args.output,
            args.character,
            args.classifier_checkpoint,
            args.generator_model_dir,
            args.style_mapping,
            args.voicevox_url,
            args.fallback_threshold,
            args.enable_dsp,
            args.dsp_features_path,
            args.dsp_f0_extractor,
            args.device,
        ),
        "evaluate": lambda: _dispatch_evaluate(args, parser),
    }

    handler = handlers.get(args.command)
    if handler is None:
        parser.error("Unknown command")
        return
    handler()
