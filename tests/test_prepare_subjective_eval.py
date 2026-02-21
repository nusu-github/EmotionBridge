from emotionbridge.scripts.prepare_subjective_eval import _build_parser


def test_prepare_subjective_eval_style_only_defaults() -> None:
    parser = _build_parser()
    args = parser.parse_args([])

    assert args.include_style_only_ab is True
    assert args.style_only_mode == "mapped_vs_default"
