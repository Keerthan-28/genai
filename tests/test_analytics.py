"""Phase 3 tests: metric registry and deterministic retail sales analytics."""

from __future__ import annotations

import pandas as pd
import pytest

from backend.analytics_engine import AnalyticsEngine
from backend.data_loader import DataLoader
from backend.data_profiler import profile_dataset
from backend.metric_registry import MetricRegistry
from backend.preprocessing import preprocess_transactions
from backend.schemas import RevenueMode, TransactionPolicy
from config.settings import PROJECT_ROOT


def test_metric_registry_contains_core_metrics_and_blocks_unsupported() -> None:
    registry = MetricRegistry()
    for mid in [
        "total_revenue",
        "gross_revenue",
        "returns_amount",
        "net_revenue",
        "monthly_revenue",
        "revenue_growth",
        "quantity_sold",
        "product_revenue",
        "top_products",
        "country_revenue",
        "average_order_value",
        "revenue_contribution",
        "cancellation_analysis",
    ]:
        assert registry.is_supported(mid)

    assert registry.is_supported("customer_satisfaction") is False
    assert registry.is_explicitly_unsupported_phrase("Net Profit Margin") is True
    assert registry.is_explicitly_unsupported_phrase("Marketing ROI") is True


def test_full_analytics_calculations_on_sample_dataset() -> None:
    sample_path = PROJECT_ROOT / "data" / "samples" / "sample_retail_transactions.csv"
    raw_df, _, _ = DataLoader().load_from_path(sample_path, persist_raw_copy=False)
    policy = TransactionPolicy(revenue_mode=RevenueMode.NET_REVENUE, deduplicate_exact_rows=True)
    annotated, processed, steps = preprocess_transactions(raw_df, policy)
    profile = profile_dataset(annotated, processed, steps)

    engine = AnalyticsEngine()
    analytics = engine.compute_full_analytics(annotated, processed, profile, policy)
    summary = analytics["summary"]

    # Manual check of sample_retail_transactions.csv (deduplicated, valid date, UnitPrice > 0):
    # Oct: 10*1000 + 50*20 + 30*40 + 20*100 = 10000 + 1000 + 1200 + 2000 = 14200
    # Nov: 8*1000 + 100*20 + 25*100 - 1*1000 + 50*40 = 8000 + 2000 + 2500 - 1000 + 2000 = 13500
    # Dec: 12*1000 + 75*20 + 15*100 - 5*40 (deduped) = 12000 + 1500 + 1500 - 200 = 14800
    # Total net = 14200 + 13500 + 14800 = 42500
    assert summary["net_revenue"] == pytest.approx(42500.0)
    assert summary["gross_revenue"] == pytest.approx(43700.0)
    assert summary["returns_amount"] == pytest.approx(200.0)
    assert summary["cancellations_amount"] == pytest.approx(1000.0)
    assert analytics["top_products"][0]["product"] == "Laptop Pro 15"
    assert analytics["country_revenue"][0]["country"] == "United Kingdom"

    oct_m = [m for m in analytics["monthly_revenue"] if m["period"] == "2024-10"][0]
    nov_m = [m for m in analytics["monthly_revenue"] if m["period"] == "2024-11"][0]
    dec_m = [m for m in analytics["monthly_revenue"] if m["period"] == "2024-12"][0]
    assert oct_m["revenue"] == pytest.approx(14200.0)
    assert nov_m["revenue"] == pytest.approx(13500.0)
    assert dec_m["revenue"] == pytest.approx(14800.0)
    assert dec_m["mom_growth_pct"] == pytest.approx(((14800.0 - 13500.0) / 13500.0) * 100.0)


def test_zero_denominator_growth_handled_cleanly() -> None:
    df = pd.DataFrame(
        {
            "InvoiceNo": ["1", "2", "C3"],
            "StockCode": ["S1", "S1", "S1"],
            "Description": ["Item A", "Item A", "Item A"],
            "Quantity": [10, 5, -5],
            "InvoiceDate": ["2024-10-01", "2024-10-02", "2024-10-03"],
            "UnitPrice": [10.0, 20.0, 20.0],
            "Country": ["UK", "UK", "UK"],
        }
    )
    # Wait: Oct net = 100 + 100 - 100 = 100; let's add a zero-net month in Nov: +100 and -100
    df = pd.DataFrame(
        {
            "InvoiceNo": ["1", "C2", "3"],
            "StockCode": ["S1", "S1", "S1"],
            "Description": ["Item A", "Item A", "Item A"],
            "Quantity": [5, -5, 10],
            "InvoiceDate": ["2024-10-05", "2024-10-06", "2024-11-05"],
            "UnitPrice": [10.0, 10.0, 10.0],
            "Country": ["UK", "UK", "UK"],
        }
    )
    policy = TransactionPolicy(revenue_mode=RevenueMode.NET_REVENUE)
    annotated, processed, steps = preprocess_transactions(df, policy)
    profile = profile_dataset(annotated, processed, steps)
    engine = AnalyticsEngine()

    res = engine.compute_metric(
        "revenue_growth",
        processed,
        annotated,
        policy,
        profile,
        period="2024-11",
        comparison_period="2024-10",
        operation="growth_pct",
    )
    assert res.value is None
    assert "Zero comparison-period revenue" in (res.error_reason or "")
