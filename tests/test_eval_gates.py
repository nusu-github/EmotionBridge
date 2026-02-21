from __future__ import annotations

import math

from emotionbridge.eval.gates import boolean_check, build_gate_decision, threshold_check


def test_threshold_check_ge_pass() -> None:
    check = threshold_check("silhouette_score", 0.21, 0.15, ">=")

    assert check["pass"] is True
    assert check["operator"] == ">="
    assert check["reason"] is None


def test_threshold_check_le_fail_with_reason() -> None:
    check = threshold_check("mcd_db", 6.7, 6.0, "<=")

    assert check["pass"] is False
    assert check["operator"] == "<="
    assert check["reason"] == "mcd_db 6.7000 > 6.0000"


def test_threshold_check_rejects_non_finite_value() -> None:
    check = threshold_check("pesq_mean", math.nan, 3.5, ">=")

    assert check["pass"] is False
    assert check["reason"] == "pesq_mean is not finite"


def test_build_gate_decision_supports_overall_override() -> None:
    checks = [
        boolean_check("raw_baseline_available", True),
        boolean_check(
            "wasserstein_mean_improved",
            False,
            detail="mean_wasserstein_after did not improve",
        ),
    ]
    decision = build_gate_decision(checks, overall_pass=True)

    assert decision["pass"] is True
    assert decision["label"] == "Go"
    assert decision["checks"]["raw_baseline_available"]["pass"] is True
    assert decision["checks"]["wasserstein_mean_improved"]["pass"] is False
    assert decision["failure_reasons"] == ["mean_wasserstein_after did not improve"]


def test_build_gate_decision_custom_labels() -> None:
    checks = [boolean_check("continuous_axes_gate", False, detail="r2 below threshold")]
    decision = build_gate_decision(
        checks,
        label_pass="Conditional-Go",
        label_fail="No-Go",
    )

    assert decision["pass"] is False
    assert decision["label"] == "No-Go"
