from __future__ import annotations

import argparse
import json

from emotionbridge.training.generator_trainer import (
    load_generator_config,
    train_generator,
)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="パラメータ生成器を学習する")
    parser.add_argument(
        "--config",
        default="configs/generator.yaml",
        help="学習設定ファイル",
    )
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    config = load_generator_config(args.config)
    result = train_generator(config)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
