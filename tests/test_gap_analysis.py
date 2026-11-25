"""Phase 8 tests: gap analysis calculations and classifications."""

from __future__ import annotations

import pytest

from backend.gap_analyzer import GapAnalyzer
from backend.schemas import GapType


def test_gap_analyzer_covers_all_classifications() -> None:
    analyzer = GapAnalyzer()

    exact = analyzer.analyze("G1", 100.4, 100.0, "GBP", tolerance=1.0)
    assert exact.gap_type == GapType.EXACT_MATCH

    over = analyzer.analyze("G2", 12000.0, 10000.0, "GBP", tolerance=1.0)
    assert over.gap_type == GapType.OVERSTATED
    assert over.signed_gap == pytest.approx(2000.0)
    assert over.relative_gap_pct == pytest.approx(20.0)

    under = analyzer.analyze("G3", 8000.0, 10000.0, "GBP", tolerance=1.0)
    assert under.gap_type == GapType.UNDERSTATED
    assert under.signed_gap == pytest.approx(-2000.0)
    assert under.relative_gap_pct == pytest.approx(-20.0)

    pct_gap = analyzer.analyze("G4", 15.0, 10.0, "percent", tolerance=0.25, operation="growth_pct")
    assert pct_gap.percentage_point_diff == pytest.approx(5.0)

    period_mm = analyzer.analyze("G5", 14200.0, 14800.0, "GBP", tolerance=1.0, alternative_period_match="2024-10")
    assert period_mm.gap_type == GapType.PERIOD_MISMATCH

    metric_mm = analyzer.analyze("G6", 43700.0, 42500.0, "GBP", tolerance=1.0, alternative_metric_match="gross_revenue")
    assert metric_mm.gap_type == GapType.METRIC_MISMATCH

    not_comp = analyzer.analyze("G7", 100.0, None, "GBP", tolerance=1.0)
    assert not_comp.gap_type == GapType.NOT_COMPARABLE

    zero_actual = analyzer.analyze("G8", 50.0, 0.0, "GBP", tolerance=1.0)
    assert zero_actual.gap_type == GapType.OVERSTATED
    assert zero_actual.relative_gap_pct is None
