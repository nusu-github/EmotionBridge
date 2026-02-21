from __future__ import annotations

from math import isfinite
from typing import Any


def threshold_check(
    name: str,
    value: float,
    threshold: float,
    operator: str,
) -> dict[str, Any]:
    if operator not in {">=", "<="}:
        msg = f"Unsupported threshold operator: {operator}"
        raise ValueError(msg)

    if not isfinite(value):
        return {
            "name": name,
            "value": value,
            "threshold": threshold,
            "operator": operator,
            "pass": False,
            "reason": f"{name} is not finite",
        }

    if operator == ">=":
        passed = value >= threshold
        reason = None if passed else f"{name} {value:.4f} < {threshold:.4f}"
    else:
        passed = value <= threshold
        reason = None if passed else f"{name} {value:.4f} > {threshold:.4f}"

    return {
        "name": name,
        "value": value,
        "threshold": threshold,
        "operator": operator,
        "pass": passed,
        "reason": reason,
    }


def boolean_check(
    name: str,
    passed: bool,
    *,
    detail: str | None = None,
) -> dict[str, Any]:
    return {
        "name": name,
        "value": bool(passed),
        "operator": "bool",
        "pass": bool(passed),
        "reason": None if passed else detail or f"{name} check failed",
    }


def build_gate_decision(
    checks: list[dict[str, Any]],
    *,
    overall_pass: bool | None = None,
    label_pass: str = "Go",
    label_fail: str = "No-Go",
) -> dict[str, Any]:
    if overall_pass is None:
        gate_pass = all(bool(item.get("pass", False)) for item in checks)
    else:
        gate_pass = bool(overall_pass)

    failure_reasons = [
        str(item["reason"])
        for item in checks
        if not bool(item.get("pass", False)) and item.get("reason")
    ]

    mapped_checks = {
        str(item["name"]): {
            key: value for key, value in item.items() if key not in {"name", "reason"}
        }
        for item in checks
    }

    return {
        "label": label_pass if gate_pass else label_fail,
        "pass": gate_pass,
        "checks": mapped_checks,
        "failure_reasons": failure_reasons,
    }
