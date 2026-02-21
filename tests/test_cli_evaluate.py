from __future__ import annotations

from unittest.mock import patch

from emotionbridge import cli


def test_build_parser_parses_evaluate_roundtrip_arguments() -> None:
    parser = cli._build_parser()
    args = parser.parse_args(
        [
            "evaluate",
            "roundtrip",
            "--baseline-manifest",
            "demo/v2/manifest.json",
            "--candidate-manifest",
            "demo/v2-dsp/manifest.json",
            "--output-dir",
            "artifacts/prosody/roundtrip_eval/test",
        ],
    )

    assert args.command == "evaluate"
    assert args.eval_command == "roundtrip"
    assert args.baseline_manifest == "demo/v2/manifest.json"
    assert args.candidate_manifest == "demo/v2-dsp/manifest.json"
    assert args.output_dir == "artifacts/prosody/roundtrip_eval/test"


def test_main_dispatches_evaluate_roundtrip() -> None:
    with (
        patch(
            "emotionbridge.cli._cmd_evaluate_roundtrip",
        ) as mock_roundtrip,
        patch(
            "sys.argv",
            [
                "main.py",
                "evaluate",
                "roundtrip",
                "--baseline-manifest",
                "base.json",
                "--candidate-manifest",
                "cand.json",
                "--output-dir",
                "out_dir",
            ],
        ),
    ):
        cli.main()

    mock_roundtrip.assert_called_once_with("base.json", "cand.json", "out_dir")


def test_main_dispatches_evaluate_continuous_axes() -> None:
    with (
        patch(
            "emotionbridge.cli._cmd_evaluate_continuous_axes",
        ) as mock_axes,
        patch(
            "sys.argv",
            [
                "main.py",
                "evaluate",
                "continuous-axes",
                "--config",
                "configs/experiment_config.yaml",
                "--input-path",
                "in.parquet",
                "--anchors-json",
                "anchors.json",
                "--arousal-r2-threshold",
                "0.31",
                "--valence-r2-threshold",
                "0.16",
            ],
        ),
    ):
        cli.main()

    mock_axes.assert_called_once_with(
        "configs/experiment_config.yaml",
        "in.parquet",
        "anchors.json",
        0.31,
        0.16,
    )
