"""Phase 2 tests: preprocessing, transaction policies, and dataset profiling."""

from __future__ import annotations

import pandas as pd

from backend.data_loader import DataLoader
from backend.data_profiler import profile_dataset
from backend.preprocessing import preprocess_transactions
from backend.schemas import RevenueMode, TransactionPolicy
from config.settings import PROJECT_ROOT


def test_preprocess_preserves_returns_and_cancellations_under_net_revenue() -> None:
    sample_path = PROJECT_ROOT / "data" / "samples" / "sample_retail_transactions.csv"
    raw_df, _, _ = DataLoader().load_from_path(sample_path, persist_raw_copy=False)

    annotated, processed, steps = preprocess_transactions(
        raw_df, TransactionPolicy(revenue_mode=RevenueMode.NET_REVENUE)
    )
    assert len(annotated) == 16
    assert annotated["is_cancellation"].sum() == 1
    assert annotated["is_return"].sum() == 2  # row 13 & duplicate row 14
    assert annotated["is_exact_duplicate"].sum() == 1
    assert (~annotated["is_valid_date"]).sum() == 1
    assert (~annotated["is_valid_price"]).sum() == 1

    # Processed excludes 1 invalid date and 1 zero-price row -> 14 rows remain
    assert len(processed) == 14
    assert len(steps) >= 3


def test_gross_sales_policy_excludes_returns_and_cancellations() -> None:
    sample_path = PROJECT_ROOT / "data" / "samples" / "sample_retail_transactions.csv"
    raw_df, _, _ = DataLoader().load_from_path(sample_path, persist_raw_copy=False)

    _, gross_df, _ = preprocess_transactions(
        raw_df,
        TransactionPolicy(revenue_mode=RevenueMode.GROSS_SALES, deduplicate_exact_rows=True),
    )
    assert (gross_df["QuantityNum"] > 0).all()
    assert (~gross_df["is_cancellation"]).all()
    assert (~gross_df["is_return"]).all()


def test_profile_detects_partial_periods_and_quality_issues() -> None:
    sample_path = PROJECT_ROOT / "data" / "samples" / "sample_retail_transactions.csv"
    raw_df, _, _ = DataLoader().load_from_path(sample_path, persist_raw_copy=False)
    annotated, processed, steps = preprocess_transactions(raw_df)
    profile = profile_dataset(annotated, processed, steps)

    assert profile.total_raw_rows == 16
    assert profile.total_processed_rows == 14
    assert profile.duplicate_rows == 1
    assert profile.invalid_date_rows == 1
    assert profile.invalid_unit_price_rows == 1
    assert profile.cancellation_rows == 1
    assert profile.return_rows == 2
    assert profile.date_min == "2024-10-01"
    assert profile.date_max == "2024-12-12"
    # December only spans Dec 1..Dec 12 and 2024 only spans Oct..Dec -> both partial
    assert "2024-12" in profile.partial_periods
    assert "2024" in profile.partial_periods
    assert 0.0 < profile.data_quality_score < 1.0
