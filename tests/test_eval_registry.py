from __future__ import annotations

from unittest.mock import patch

from emotionbridge.eval.registry import HFMetricRegistry


def test_load_uses_cache_for_same_spec() -> None:
    with patch("emotionbridge.eval.registry.evaluate.load") as mock_load:
        metric_obj = object()
        mock_load.return_value = metric_obj

        registry = HFMetricRegistry(default_revision="unit-test-rev")
        first = registry.load("accuracy")
        second = registry.load("accuracy")

    assert first is metric_obj
    assert second is metric_obj
    assert mock_load.call_count == 1
    mock_load.assert_called_once_with("accuracy", revision="unit-test-rev")


def test_load_passes_explicit_arguments() -> None:
    with patch("emotionbridge.eval.registry.evaluate.load") as mock_load:
        mock_load.return_value = object()

        registry = HFMetricRegistry(default_revision="default-rev")
        registry.load(
            "f1",
            config_name="multilabel",
            revision="explicit-rev",
            module_type="metric",
        )

    mock_load.assert_called_once_with(
        "f1",
        config_name="multilabel",
        revision="explicit-rev",
        module_type="metric",
    )


def test_load_cache_key_includes_revision() -> None:
    with patch("emotionbridge.eval.registry.evaluate.load") as mock_load:
        mock_load.side_effect = [object(), object()]

        registry = HFMetricRegistry(default_revision="base-rev")
        first = registry.load("precision")
        second = registry.load("precision", revision="another-rev")

    assert first is not second
    assert mock_load.call_count == 2
