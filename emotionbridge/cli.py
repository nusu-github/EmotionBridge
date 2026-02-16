import argparse
import json

from transformers import AutoTokenizer

from emotionbridge.config import load_config
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


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="emotionbridge",
        description="EmotionBridge Phase 0 CLI",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

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

    return parser


def _cmd_train(config_path: str) -> None:
    config = load_config(config_path)
    result = train_phase0(config)
    if result is not None:
        print(json.dumps(result, ensure_ascii=False, indent=2))


def _cmd_analyze_data(config_path: str, output_dir: str) -> None:
    from pathlib import Path

    config = load_config(config_path)
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


def _cmd_encode(checkpoint: str, text: str, device: str) -> None:
    encoder = EmotionEncoder(checkpoint, device=device)
    vector = encoder.encode(text)
    print(json.dumps(vector.tolist(), ensure_ascii=False))


def main() -> None:
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

    parser.error("Unknown command")
