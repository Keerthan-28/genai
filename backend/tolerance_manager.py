"""Metric-specific numerical tolerance manager."""

from __future__ import annotations

from backend.metric_registry import MetricRegistry


class ToleranceManager:
    """Computes effective tolerances accounting for metric rules and reporting rounding."""

    def __init__(self, registry: MetricRegistry | None = None) -> None:
        self.registry = registry or MetricRegistry()

    @staticmethod
    def _infer_rounding_half_ulp(claimed_value: float) -> float:
        text = f"{claimed_value}"
        if "." in text:
            decimals = len(text.rstrip("0").split(".")[1])
            if decimals == 0:
                return 0.5
            return 0.5 * (10 ** (-decimals))
        return 0.5

    def get_tolerance(self, metric_id: str | None, actual_value: float, claimed_value: float | None = None) -> float:
        metric_def = self.registry.get(metric_id) if metric_id else None
        base_abs = metric_def.abs_tolerance if metric_def else 1.0
        base_rel = metric_def.rel_tolerance if metric_def else 0.005

        rel_bound = abs(actual_value) * base_rel
        rounding_bound = self._infer_rounding_half_ulp(claimed_value) if claimed_value is not None else 0.0
        # Use max of metric absolute tolerance, relative tolerance, and rounding step
        return max(base_abs, rel_bound, rounding_bound)
