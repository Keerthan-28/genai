"""Dataset profiling and period-coverage analysis."""

from __future__ import annotations

import calendar
from datetime import datetime

import pandas as pd

from backend.schemas import DatasetProfile, PeriodCoverageInfo, TransformationStep


def _compute_period_coverage(valid_dates: pd.Series) -> tuple[list[PeriodCoverageInfo], list[str]]:
    """Compute monthly and yearly coverage metrics to guard against partial-period claims."""
    if valid_dates.empty:
        return [], []

    coverage_list: list[PeriodCoverageInfo] = []
    partial_periods: list[str] = []

    dates_dt = pd.to_datetime(valid_dates.dropna())
    if dates_dt.empty:
        return [], []

    # Monthly coverage
    month_series = dates_dt.dt.strftime("%Y-%m")
    for month_str, group in dates_dt.groupby(month_series):
        year_int, month_int = map(int, str(month_str).split("-"))
        expected_days = calendar.monthrange(year_int, month_int)[1]
        first_dt = group.min()
        last_dt = group.max()
        span_days = int((last_dt.normalize() - first_dt.normalize()).days) + 1
        distinct_days = int(group.dt.normalize().nunique())
        # Consider a month covered only if transactions span at least 80% of the month
        # (prevents treating 1-5 days of data as a full month).
        span_ratio = round(span_days / expected_days, 4)
        is_complete = span_ratio >= 0.80 and distinct_days >= 2
        if not is_complete:
            partial_periods.append(str(month_str))
        coverage_list.append(
            PeriodCoverageInfo(
                period=str(month_str),
                period_type="month",
                days_present=distinct_days,
                expected_days=expected_days,
                coverage_ratio=span_ratio,
                is_complete=is_complete,
                first_date=first_dt.strftime("%Y-%m-%d"),
                last_date=last_dt.strftime("%Y-%m-%d"),
            )
        )

    # Yearly coverage
    year_series = dates_dt.dt.strftime("%Y")
    for year_str, group in dates_dt.groupby(year_series):
        year_int = int(str(year_str))
        expected_days = 366 if calendar.isleap(year_int) else 365
        first_dt = group.min()
        last_dt = group.max()
        distinct_months = int(group.dt.month.nunique())
        distinct_days = int(group.dt.normalize().nunique())
        span_days = int((last_dt.normalize() - first_dt.normalize()).days) + 1
        span_ratio = round(span_days / expected_days, 4)
        is_complete = distinct_months >= 11 and span_ratio >= 0.80
        if not is_complete:
            partial_periods.append(str(year_str))
        coverage_list.append(
            PeriodCoverageInfo(
                period=str(year_str),
                period_type="year",
                days_present=distinct_days,
                expected_days=expected_days,
                coverage_ratio=span_ratio,
                is_complete=is_complete,
                first_date=first_dt.strftime("%Y-%m-%d"),
                last_date=last_dt.strftime("%Y-%m-%d"),
            )
        )

    coverage_list.sort(key=lambda item: (item.period_type, item.period))
    return coverage_list, sorted(set(partial_periods))


def profile_dataset(
    annotated_df: pd.DataFrame,
    processed_df: pd.DataFrame,
    transformations: list[TransformationStep] | None = None,
) -> DatasetProfile:
    """Build a complete data quality and coverage profile from annotated & processed data."""
    total_raw = int(len(annotated_df))
    total_proc = int(len(processed_df))

    core_cols = [
        c for c in ("InvoiceNo", "StockCode", "Description", "Quantity", "InvoiceDate", "UnitPrice", "CustomerID", "Country")
        if c in annotated_df.columns
    ]
    dtypes = {col: str(annotated_df[col].dtype) for col in core_cols}
    missing_values = {col: int(annotated_df[col].isna().sum()) for col in core_cols}
    missing_rates = {
        col: round(missing_values[col] / total_raw, 4) if total_raw > 0 else 0.0
        for col in core_cols
    }

    duplicate_rows = int(annotated_df.get("is_exact_duplicate", pd.Series(dtype=bool)).sum())
    invalid_date_rows = int((~annotated_df.get("is_valid_date", pd.Series(dtype=bool))).sum())
    invalid_numeric_rows = int((~annotated_df.get("is_valid_numeric", pd.Series(dtype=bool))).sum())
    invalid_price_rows = int((~annotated_df.get("is_valid_price", pd.Series(dtype=bool))).sum())
    negative_qty_rows = int(
        (annotated_df["QuantityNum"].notna() & (annotated_df["QuantityNum"] < 0)).sum()
        if "QuantityNum" in annotated_df.columns
        else 0
    )
    return_rows = int(annotated_df.get("is_return", pd.Series(dtype=bool)).sum())
    cancellation_rows = int(annotated_df.get("is_cancellation", pd.Series(dtype=bool)).sum())

    valid_dates = annotated_df.get("ParsedInvoiceDate", pd.Series(dtype="datetime64[ns]")).dropna()
    date_min = valid_dates.min().strftime("%Y-%m-%d") if not valid_dates.empty else None
    date_max = valid_dates.max().strftime("%Y-%m-%d") if not valid_dates.empty else None

    period_coverage, partial_periods = _compute_period_coverage(valid_dates)

    distinct_products = int(processed_df["Description"].dropna().nunique()) if "Description" in processed_df.columns else 0
    distinct_countries = int(processed_df["Country"].dropna().nunique()) if "Country" in processed_df.columns else 0
    distinct_invoices = int(processed_df["InvoiceNo"].dropna().nunique()) if "InvoiceNo" in processed_df.columns else 0

    # Deterministic data quality score (0.0 - 1.0)
    quality_notes: list[str] = []
    if total_raw == 0:
        score = 0.0
        quality_notes.append("Dataset has zero rows.")
    else:
        invalid_rate = (invalid_date_rows + invalid_numeric_rows + invalid_price_rows) / (total_raw * 3)
        dup_rate = duplicate_rows / total_raw
        req_missing_cols = [c for c in core_cols if c != "CustomerID"]
        avg_req_missing = (
            sum(missing_rates[c] for c in req_missing_cols) / len(req_missing_cols)
            if req_missing_cols
            else 0.0
        )
        score = max(0.0, min(1.0, round(1.0 - (0.5 * invalid_rate + 0.3 * avg_req_missing + 0.2 * dup_rate), 4)))

    if invalid_date_rows > 0:
        quality_notes.append(f"{invalid_date_rows} row(s) have invalid InvoiceDate values.")
    if invalid_price_rows > 0:
        quality_notes.append(f"{invalid_price_rows} row(s) have UnitPrice <= 0 or non-numeric.")
    if duplicate_rows > 0:
        quality_notes.append(f"{duplicate_rows} exact duplicate row(s) detected.")
    if cancellation_rows > 0:
        quality_notes.append(f"{cancellation_rows} cancellation transaction(s) ('C' invoice prefix) detected.")
    if return_rows > 0:
        quality_notes.append(f"{return_rows} return transaction(s) (negative quantity) detected.")
    if partial_periods:
        quality_notes.append(
            f"Partial period coverage detected for: {', '.join(partial_periods)}. "
            "Do not claim full-period coverage for these periods."
        )

    return DatasetProfile(
        total_raw_rows=total_raw,
        total_processed_rows=total_proc,
        total_columns=int(len(annotated_df.columns)),
        dtypes=dtypes,
        missing_values=missing_values,
        missing_rates=missing_rates,
        duplicate_rows=duplicate_rows,
        invalid_date_rows=invalid_date_rows,
        invalid_numeric_rows=invalid_numeric_rows,
        invalid_unit_price_rows=invalid_price_rows,
        negative_quantity_rows=negative_qty_rows,
        return_rows=return_rows,
        cancellation_rows=cancellation_rows,
        date_min=date_min,
        date_max=date_max,
        distinct_products=distinct_products,
        distinct_countries=distinct_countries,
        distinct_invoices=distinct_invoices,
        period_coverage=period_coverage,
        partial_periods=partial_periods,
        data_quality_score=score,
        quality_notes=quality_notes,
        transformations=transformations or [],
    )
