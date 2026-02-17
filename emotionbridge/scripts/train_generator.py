from __future__ import annotations

import argparse
import json

from emotionbridge.training.generator_trainer import (
    load_phase3b_config,
    train_phase3b_generator,
)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Phase3b パラメータ生成器を学習する")
    parser.add_argument(
        "--config",
        default="configs/phase3b.yaml",
        help="学習設定ファイル",
    )
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    config = load_phase3b_config(args.config)
    result = train_phase3b_generator(config)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
