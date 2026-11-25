"""Data preprocessing and configurable transaction policy engine.

Preserves original records, classifies cancellations and returns rather than
blindly discarding them, and logs every transformation step.
"""

from __future__ import annotations

import pandas as pd

from backend.schemas import RevenueMode, TransactionPolicy, TransformationStep


def preprocess_transactions(
    raw_df: pd.DataFrame,
    policy: TransactionPolicy | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, list[TransformationStep]]:
    """Preprocess retail transaction records under a documented TransactionPolicy.

    Returns:
        annotated_df: Full copy of raw data with parsed types and quality/classification flags.
        analytics_df: Policy-filtered DataFrame ready for metric calculations.
        transformations: Ordered log of transformations applied.
    """
    policy = policy or TransactionPolicy()
    df = raw_df.copy()
    transformations: list[TransformationStep] = []

    # Ensure expected columns exist defensively
    for col in ("InvoiceNo", "StockCode", "Description", "Quantity", "InvoiceDate", "UnitPrice", "Country"):
        if col not in df.columns:
            df[col] = pd.NA

    # Normalize string fields cleanly & precompute lowercase keys for fast O(1) filtering
    df["InvoiceNo"] = df["InvoiceNo"].astype("string").str.strip()
    df["StockCode"] = df["StockCode"].astype("string").str.strip()
    df["Description"] = df["Description"].astype("string").str.strip()
    df["Country"] = df["Country"].astype("string").str.strip()
    df["_stock_lower"] = df["StockCode"].fillna("").str.lower()
    df["_desc_lower"] = df["Description"].fillna("").str.lower()
    df["_country_lower"] = df["Country"].fillna("").str.lower()

    # Parse dates and numeric values (vectorized, fast on 500k+ rows)
    if pd.api.types.is_datetime64_any_dtype(df["InvoiceDate"]):
        df["ParsedInvoiceDate"] = df["InvoiceDate"]
    else:
        df["ParsedInvoiceDate"] = pd.to_datetime(df["InvoiceDate"], errors="coerce", format="mixed")
    df["QuantityNum"] = pd.to_numeric(df["Quantity"], errors="coerce")
    df["UnitPriceNum"] = pd.to_numeric(df["UnitPrice"], errors="coerce")

    df["is_valid_date"] = df["ParsedInvoiceDate"].notna()
    df["is_valid_numeric"] = df["QuantityNum"].notna() & df["UnitPriceNum"].notna()
    df["is_valid_price"] = df["UnitPriceNum"].notna() & (df["UnitPriceNum"] > 0)

    # Transaction classification flags (never blindly removed)
    invoice_upper = df["InvoiceNo"].fillna("").str.upper()
    df["is_cancellation"] = invoice_upper.str.startswith("C")
    df["is_return"] = (~df["is_cancellation"]) & df["QuantityNum"].notna() & (df["QuantityNum"] < 0)
    df["is_gross_sale"] = (
        (~df["is_cancellation"])
        & (~df["is_return"])
        & df["QuantityNum"].notna()
        & (df["QuantityNum"] > 0)
        & df["is_valid_price"]
    )
    df["is_exact_duplicate"] = df.duplicated(
        subset=[c for c in ("InvoiceNo", "StockCode", "Description", "Quantity", "InvoiceDate", "UnitPrice", "Country") if c in df.columns],
        keep="first",
    )

    # Compute line revenue & vectorized YYYY-MM / YYYY periods (avoids slow per-row strftime)
    df["LineRevenue"] = (df["QuantityNum"] * df["UnitPriceNum"]).astype("float64")
    df["PeriodMonth"] = df["ParsedInvoiceDate"].dt.to_period("M").astype("string")
    df["PeriodYear"] = df["ParsedInvoiceDate"].dt.year.astype("Int64").astype("string")

    working = df.copy()
    # Filter invalid numeric rows first (cannot compute revenue without numbers)
    before = len(working)
    working = working[working["is_valid_numeric"]].copy()
    transformations.append(
        TransformationStep(
            step_name="filter_invalid_numerics",
            rows_before=before,
            rows_after=len(working),
            affected_rows=before - len(working),
            reason="Excluded rows with non-numeric Quantity or UnitPrice.",
        )
    )

    if policy.drop_invalid_dates:
        before = len(working)
        working = working[working["is_valid_date"]].copy()
        transformations.append(
            TransformationStep(
                step_name="filter_invalid_dates",
                rows_before=before,
                rows_after=len(working),
                affected_rows=before - len(working),
                reason="Excluded rows with unparseable InvoiceDate values.",
            )
        )

    if policy.drop_non_positive_unit_price:
        before = len(working)
        working = working[working["is_valid_price"]].copy()
        transformations.append(
            TransformationStep(
                step_name="filter_non_positive_unit_price",
                rows_before=before,
                rows_after=len(working),
                affected_rows=before - len(working),
                reason="Excluded rows with UnitPrice <= 0.",
            )
        )

    if policy.deduplicate_exact_rows:
        before = len(working)
        working = working[~working["is_exact_duplicate"]].copy()
        transformations.append(
            TransformationStep(
                step_name="deduplicate_exact_rows",
                rows_before=before,
                rows_after=len(working),
                affected_rows=before - len(working),
                reason="Removed exact duplicate transaction rows per policy.",
            )
        )

    # Apply transaction revenue mode policy
    before = len(working)
    if policy.revenue_mode == RevenueMode.GROSS_SALES:
        working = working[working["is_gross_sale"]].copy()
        mode_reason = "Retained positive gross sales only (excluded returns and cancellations)."
    elif policy.revenue_mode == RevenueMode.RETURNS_ONLY:
        working = working[working["is_return"]].copy()
        mode_reason = "Retained return transactions only (Quantity < 0 without 'C' prefix)."
    elif policy.revenue_mode == RevenueMode.CANCELLATIONS_ONLY:
        working = working[working["is_cancellation"]].copy()
        mode_reason = "Retained cancellation transactions only (InvoiceNo starting with 'C')."
    else:  # NET_REVENUE
        mask = pd.Series(True, index=working.index)
        if not policy.include_returns:
            mask &= ~working["is_return"]
        if not policy.include_cancellations:
            mask &= ~working["is_cancellation"]
        working = working[mask].copy()
        mode_reason = (
            f"Net revenue mode (include_returns={policy.include_returns}, "
            f"include_cancellations={policy.include_cancellations})."
        )

    transformations.append(
        TransformationStep(
            step_name=f"apply_revenue_mode_{policy.revenue_mode.value}",
            rows_before=before,
            rows_after=len(working),
            affected_rows=before - len(working),
            reason=mode_reason,
        )
    )

    return df, working, transformations
