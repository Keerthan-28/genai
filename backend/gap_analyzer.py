"""Phase 8: Numerical and categorical gap analysis engine."""

from __future__ import annotations

from backend.schemas import GapAnalysisResult, GapType


class GapAnalyzer:
    """Calculates signed, absolute, and relative numerical gaps and classifies mismatches."""

    def analyze(
        self,
        claim_id: str,
        claimed_value: float | str | None,
        actual_value: float | str | None,
        unit: str,
        tolerance: float | None,
        operation: str = "value",
        alternative_period_match: str | None = None,
        alternative_metric_match: str | None = None,
    ) -> GapAnalysisResult:
        # Non-comparable when either value is missing
        if claimed_value is None or actual_value is None:
            return GapAnalysisResult(
                claim_id=claim_id,
                claimed_value=claimed_value,
                actual_value=actual_value,
                gap_type=GapType.NOT_COMPARABLE,
                unit=unit,
                tolerance=tolerance,
                comparison_method="missing_value",
                notes="Cannot compute numerical gap without both claimed and actual values.",
            )

        # Categorical / ranking comparison
        if operation == "ranking" or isinstance(claimed_value, str) or isinstance(actual_value, str):
            c_str = str(claimed_value).strip().lower()
            a_str = str(actual_value).strip().lower()
            is_match = c_str == a_str
            return GapAnalysisResult(
                claim_id=claim_id,
                claimed_value=claimed_value,
                actual_value=actual_value,
                gap_type=GapType.EXACT_MATCH if is_match else GapType.METRIC_MISMATCH,
                unit=unit or "rank",
                tolerance=0.0,
                comparison_method="categorical_exact",
                notes=(
                    "Claimed rank matches actual top performer."
                    if is_match
                    else f"Ranking mismatch: claimed '{claimed_value}', actual '{actual_value}'."
                ),
            )

        claimed_f = float(claimed_value)
        actual_f = float(actual_value)
        signed_gap = claimed_f - actual_f
        abs_gap = abs(signed_gap)
        tol = 0.0 if tolerance is None else float(tolerance)

        rel_gap_pct: float | None = None
        if abs(actual_f) > 1e-12:
            rel_gap_pct = (signed_gap / abs(actual_f)) * 100.0

        pp_diff = signed_gap if (unit == "percent" or operation in ("growth_pct", "share_pct")) else None

        if abs_gap <= tol:
            gap_type = GapType.EXACT_MATCH
            note = f"Exact/within tolerance (|gap|={abs_gap:.4f} <= tol={tol:.4f})."
        elif alternative_period_match:
            gap_type = GapType.PERIOD_MISMATCH
            note = (
                f"Claimed value matches period '{alternative_period_match}' instead of the claimed period "
                f"(gap={signed_gap:+.2f})."
            )
        elif alternative_metric_match:
            gap_type = GapType.METRIC_MISMATCH
            note = (
                f"Claimed value matches alternative metric '{alternative_metric_match}' "
                f"(gap={signed_gap:+.2f})."
            )
        elif signed_gap > 0:
            gap_type = GapType.OVERSTATED
            note = f"Claimed value is overstated by {abs_gap:.2f} {unit}."
        else:
            gap_type = GapType.UNDERSTATED
            note = f"Claimed value is understated by {abs_gap:.2f} {unit}."

        return GapAnalysisResult(
            claim_id=claim_id,
            claimed_value=claimed_f,
            actual_value=actual_f,
            signed_gap=signed_gap,
            absolute_gap=abs_gap,
            relative_gap_pct=rel_gap_pct,
            percentage_point_diff=pp_diff,
            gap_type=gap_type,
            unit=unit,
            tolerance=tol,
            comparison_method="percentage_points" if pp_diff is not None else "numeric_tolerance",
            notes=note,
        )
