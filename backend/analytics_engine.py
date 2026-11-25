"""Deterministic Python & Pandas retail sales analytics engine.

All numerical calculations are computed strictly from processed transaction records.
Unrounded float values are preserved internally.
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from backend.metric_registry import MetricRegistry
from backend.schemas import (
    DatasetProfile,
    MetricComputationResult,
    TransactionPolicy,
)


def _filter_by_period(df: pd.DataFrame, period: str | None) -> pd.DataFrame:
    if not period:
        return df
    p = str(period).strip()
    if len(p) == 7 and "-" in p and "PeriodMonth" in df.columns:
        return df[df["PeriodMonth"] == p]
    if len(p) == 4 and p.isdigit() and "PeriodYear" in df.columns:
        return df[df["PeriodYear"] == p]
    return df.iloc[0:0]


def _filter_by_entity(
    df: pd.DataFrame,
    entity: str | None,
    entity_type: str | None,
) -> pd.DataFrame:
    if not entity or entity_type in (None, "overall"):
        return df
    norm = str(entity).strip().lower()
    if entity_type == "product":
        if "_desc_lower" in df.columns and "_stock_lower" in df.columns:
            return df[(df["_desc_lower"] == norm) | (df["_stock_lower"] == norm)]
        desc_match = (
            df["Description"].fillna("").str.strip().str.lower() == norm
            if "Description" in df.columns
            else pd.Series(False, index=df.index)
        )
        code_match = (
            df["StockCode"].fillna("").str.strip().str.lower() == norm
            if "StockCode" in df.columns
            else pd.Series(False, index=df.index)
        )
        return df[desc_match | code_match]
    if entity_type == "country" and "Country" in df.columns:
        if "_country_lower" in df.columns:
            return df[df["_country_lower"] == norm]
        return df[df["Country"].fillna("").str.strip().str.lower() == norm]
    return df.iloc[0:0]


class AnalyticsEngine:
    """Calculates all retail metrics deterministically from transaction records."""

    def __init__(self, registry: MetricRegistry | None = None) -> None:
        self.registry = registry or MetricRegistry()

    def compute_full_analytics(
        self,
        annotated_df: pd.DataFrame,
        processed_df: pd.DataFrame,
        profile: DatasetProfile,
        policy: TransactionPolicy | None = None,
    ) -> dict[str, Any]:
        """Compute the full suite of 14 retail analytics outputs."""
        policy = policy or TransactionPolicy()

        # Valid rows with numeric values & valid dates for gross/returns/cancellations breakdown
        valid_base = annotated_df[
            annotated_df["is_valid_numeric"]
            & (annotated_df["is_valid_date"] if policy.drop_invalid_dates else True)
            & (annotated_df["is_valid_price"] if policy.drop_non_positive_unit_price else True)
        ].copy()
        if policy.deduplicate_exact_rows:
            valid_base = valid_base[~valid_base["is_exact_duplicate"]].copy()

        total_revenue = float(processed_df["LineRevenue"].sum()) if not processed_df.empty else 0.0
        gross_df = valid_base[valid_base["is_gross_sale"]]
        returns_df = valid_base[valid_base["is_return"]]
        cancels_df = valid_base[valid_base["is_cancellation"]]

        gross_revenue = float(gross_df["LineRevenue"].sum()) if not gross_df.empty else 0.0
        returns_signed = float(returns_df["LineRevenue"].sum()) if not returns_df.empty else 0.0
        returns_abs = abs(returns_signed)
        cancellations_signed = float(cancels_df["LineRevenue"].sum()) if not cancels_df.empty else 0.0
        cancellations_abs = abs(cancellations_signed)
        net_revenue = float(valid_base["LineRevenue"].sum()) if not valid_base.empty else 0.0

        quantity_sold = float(processed_df["QuantityNum"].sum()) if not processed_df.empty else 0.0
        distinct_orders = int(processed_df["InvoiceNo"].dropna().nunique()) if not processed_df.empty else 0
        aov = (total_revenue / distinct_orders) if distinct_orders > 0 else 0.0

        # Monthly revenue & trends & MoM growth
        monthly_records: list[dict[str, Any]] = []
        if not processed_df.empty and "PeriodMonth" in processed_df.columns:
            coverage_lookup = {c.period: c for c in profile.period_coverage if c.period_type == "month"}
            grouped_m = (
                processed_df.groupby("PeriodMonth", dropna=True)
                .agg(
                    revenue=("LineRevenue", "sum"),
                    quantity=("QuantityNum", "sum"),
                    orders=("InvoiceNo", "nunique"),
                    transactions=("InvoiceNo", "count"),
                )
                .sort_index()
            )
            prev_rev: float | None = None
            prev_month: str | None = None
            for month_key, row in grouped_m.iterrows():
                rev = float(row["revenue"])
                growth_pct: float | None = None
                if prev_rev is not None and abs(prev_rev) > 1e-12:
                    growth_pct = ((rev - prev_rev) / abs(prev_rev)) * 100.0
                cov = coverage_lookup.get(str(month_key))
                monthly_records.append(
                    {
                        "period": str(month_key),
                        "revenue": rev,
                        "quantity": float(row["quantity"]),
                        "orders": int(row["orders"]),
                        "transactions": int(row["transactions"]),
                        "previous_period": prev_month,
                        "mom_growth_pct": growth_pct,
                        "revenue_share_pct": ((rev / total_revenue) * 100.0) if abs(total_revenue) > 1e-12 else None,
                        "is_complete_period": cov.is_complete if cov else False,
                    }
                )
                prev_rev = rev
                prev_month = str(month_key)

        # Overall growth (latest month vs previous month)
        latest_growth: dict[str, Any] | None = None
        for rec in reversed(monthly_records):
            if rec["mom_growth_pct"] is not None:
                latest_growth = {
                    "period": rec["period"],
                    "comparison_period": rec["previous_period"],
                    "growth_pct": rec["mom_growth_pct"],
                }
                break

        # Product-wise revenue & top-performing products
        product_records: list[dict[str, Any]] = []
        if not processed_df.empty and "Description" in processed_df.columns:
            grouped_p = (
                processed_df.dropna(subset=["Description"])
                .groupby("Description")
                .agg(
                    revenue=("LineRevenue", "sum"),
                    quantity=("QuantityNum", "sum"),
                    orders=("InvoiceNo", "nunique"),
                )
                .sort_values("revenue", ascending=False)
            )
            for rank_idx, (prod, row) in enumerate(grouped_p.iterrows(), start=1):
                rev = float(row["revenue"])
                product_records.append(
                    {
                        "rank": rank_idx,
                        "product": str(prod),
                        "revenue": rev,
                        "quantity": float(row["quantity"]),
                        "orders": int(row["orders"]),
                        "revenue_contribution_pct": ((rev / total_revenue) * 100.0) if abs(total_revenue) > 1e-12 else None,
                    }
                )

        # Country-wise revenue
        country_records: list[dict[str, Any]] = []
        if not processed_df.empty and "Country" in processed_df.columns:
            grouped_c = (
                processed_df.dropna(subset=["Country"])
                .groupby("Country")
                .agg(
                    revenue=("LineRevenue", "sum"),
                    quantity=("QuantityNum", "sum"),
                    orders=("InvoiceNo", "nunique"),
                )
                .sort_values("revenue", ascending=False)
            )
            for rank_idx, (country, row) in enumerate(grouped_c.iterrows(), start=1):
                rev = float(row["revenue"])
                country_records.append(
                    {
                        "rank": rank_idx,
                        "country": str(country),
                        "revenue": rev,
                        "quantity": float(row["quantity"]),
                        "orders": int(row["orders"]),
                        "revenue_contribution_pct": ((rev / total_revenue) * 100.0) if abs(total_revenue) > 1e-12 else None,
                    }
                )

        # Monthly trends summary
        trend_direction = "insufficient_periods"
        if len(monthly_records) >= 2:
            first_m = monthly_records[0]["revenue"]
            last_m = monthly_records[-1]["revenue"]
            if last_m > first_m:
                trend_direction = "upward"
            elif last_m < first_m:
                trend_direction = "downward"
            else:
                trend_direction = "flat"

        cancellation_rate_pct = (
            (len(cancels_df) / len(valid_base)) * 100.0 if len(valid_base) > 0 else 0.0
        )

        return {
            "policy": policy.model_dump(mode="json"),
            "currency_symbol": policy.currency_symbol,
            "currency_code": policy.currency_code,
            "summary": {
                "total_revenue": total_revenue,
                "gross_revenue": gross_revenue,
                "returns_amount": returns_abs,
                "returns_signed": returns_signed,
                "cancellations_amount": cancellations_abs,
                "cancellations_signed": cancellations_signed,
                "net_revenue": net_revenue,
                "quantity_sold": quantity_sold,
                "distinct_orders": distinct_orders,
                "average_order_value": aov,
                "distinct_products": len(product_records),
                "distinct_countries": len(country_records),
            },
            "monthly_revenue": monthly_records,
            "latest_revenue_growth": latest_growth,
            "monthly_trends": {
                "direction": trend_direction,
                "months_observed": len(monthly_records),
                "partial_periods": profile.partial_periods,
            },
            "product_performance": product_records,
            "top_products": product_records[:10],
            "country_revenue": country_records,
            "cancellation_analysis": {
                "cancellation_transactions": int(len(cancels_df)),
                "cancellation_invoices": int(cancels_df["InvoiceNo"].nunique()) if not cancels_df.empty else 0,
                "cancellation_amount": cancellations_abs,
                "cancellation_rate_pct": cancellation_rate_pct,
                "return_transactions": int(len(returns_df)),
                "return_amount": returns_abs,
            },
        }

    def compute_metric(
        self,
        metric_id: str,
        processed_df: pd.DataFrame,
        annotated_df: pd.DataFrame,
        policy: TransactionPolicy,
        profile: DatasetProfile,
        period: str | None = None,
        comparison_period: str | None = None,
        entity: str | None = None,
        entity_type: str | None = None,
        operation: str = "value",
        rank_k: int = 1,
    ) -> MetricComputationResult:
        """Independently compute a single normalized metric for verification."""
        metric_def = self.registry.get(metric_id)
        if metric_def is None:
            return MetricComputationResult(
                metric_id=metric_id,
                operation=operation,
                error_reason=f"Unsupported metric '{metric_id}'.",
            )

        # Select base dataframe according to metric semantics
        valid_base = annotated_df[
            annotated_df["is_valid_numeric"]
            & (annotated_df["is_valid_date"] if policy.drop_invalid_dates else True)
            & (annotated_df["is_valid_price"] if policy.drop_non_positive_unit_price else True)
        ].copy()
        if policy.deduplicate_exact_rows:
            valid_base = valid_base[~valid_base["is_exact_duplicate"]].copy()

        if metric_id == "gross_revenue":
            base_df = valid_base[valid_base["is_gross_sale"]].copy()
        elif metric_id == "net_revenue":
            base_df = valid_base
        elif metric_id == "returns_amount":
            base_df = valid_base[valid_base["is_return"]].copy()
        elif metric_id == "cancellation_analysis":
            base_df = valid_base[valid_base["is_cancellation"]].copy()
        else:
            base_df = processed_df

        filters_applied: dict[str, Any] = {}
        if period:
            filters_applied["period"] = period
        if comparison_period:
            filters_applied["comparison_period"] = comparison_period
        if entity:
            filters_applied["entity"] = entity
            filters_applied["entity_type"] = entity_type

        is_partial = bool(period and period in profile.partial_periods)

        # Growth calculation
        if operation == "growth_pct" or metric_id == "revenue_growth":
            curr_df = _filter_by_entity(_filter_by_period(base_df, period), entity, entity_type)
            prev_df = _filter_by_entity(_filter_by_period(base_df, comparison_period), entity, entity_type)
            if curr_df.empty or prev_df.empty:
                return MetricComputationResult(
                    metric_id=metric_id,
                    operation="growth_pct",
                    matching_rows=int(len(curr_df)),
                    filters_applied=filters_applied,
                    is_partial_period=is_partial,
                    error_reason="Missing records in current or comparison period for growth calculation.",
                )
            curr_val = float(curr_df["LineRevenue"].sum())
            prev_val = float(prev_df["LineRevenue"].sum())
            if abs(prev_val) <= 1e-12:
                return MetricComputationResult(
                    metric_id=metric_id,
                    operation="growth_pct",
                    matching_rows=int(len(curr_df)),
                    filters_applied=filters_applied,
                    is_partial_period=is_partial,
                    error_reason="Zero comparison-period revenue denominator prevents growth percentage calculation.",
                    metadata={"current_value": curr_val, "previous_value": prev_val},
                )
            growth = ((curr_val - prev_val) / abs(prev_val)) * 100.0
            return MetricComputationResult(
                metric_id=metric_id,
                operation="growth_pct",
                value=growth,
                unit="percent",
                matching_rows=int(len(curr_df) + len(prev_df)),
                formula_used=metric_def.formula,
                filters_applied=filters_applied,
                is_partial_period=is_partial,
                metadata={"current_value": curr_val, "previous_value": prev_val},
            )

        scoped = _filter_by_period(base_df, period)

        # Share / contribution percentage
        if operation == "share_pct" or metric_id == "revenue_contribution":
            entity_df = _filter_by_entity(scoped, entity, entity_type)
            total_rev = float(scoped["LineRevenue"].sum()) if not scoped.empty else 0.0
            if abs(total_rev) <= 1e-12:
                return MetricComputationResult(
                    metric_id=metric_id,
                    operation="share_pct",
                    matching_rows=int(len(entity_df)),
                    filters_applied=filters_applied,
                    is_partial_period=is_partial,
                    error_reason="Zero total revenue denominator prevents contribution share calculation.",
                )
            if metric_id == "cancellation_analysis":
                denom = _filter_by_period(valid_base, period)
                if len(denom) == 0:
                    return MetricComputationResult(
                        metric_id=metric_id,
                        operation="share_pct",
                        matching_rows=0,
                        filters_applied=filters_applied,
                        error_reason="Zero denominator records for cancellation rate.",
                    )
                rate = (len(entity_df) / len(denom)) * 100.0
                return MetricComputationResult(
                    metric_id=metric_id,
                    operation="share_pct",
                    value=rate,
                    unit="percent",
                    matching_rows=int(len(entity_df)),
                    formula_used="(Cancellation Rows / Total Valid Rows) * 100",
                    filters_applied=filters_applied,
                    is_partial_period=is_partial,
                )
            entity_rev = float(entity_df["LineRevenue"].sum()) if not entity_df.empty else 0.0
            share = (entity_rev / total_rev) * 100.0
            return MetricComputationResult(
                metric_id=metric_id,
                operation="share_pct",
                value=share,
                unit="percent",
                matching_rows=int(len(entity_df)),
                formula_used=metric_def.formula,
                filters_applied=filters_applied,
                is_partial_period=is_partial,
                metadata={"entity_revenue": entity_rev, "total_revenue": total_rev},
            )

        # Ranking
        if operation == "ranking" or metric_id == "top_products":
            group_col = "Country" if entity_type == "country" else "Description"
            grouped = (
                scoped.dropna(subset=[group_col])
                .groupby(group_col)["LineRevenue"]
                .sum()
                .sort_values(ascending=False)
            )
            if grouped.empty:
                return MetricComputationResult(
                    metric_id=metric_id,
                    operation="ranking",
                    matching_rows=0,
                    filters_applied=filters_applied,
                    is_partial_period=is_partial,
                    error_reason="No matching records for ranking.",
                )
            idx = max(0, min(rank_k - 1, len(grouped) - 1))
            top_name = str(grouped.index[idx])
            return MetricComputationResult(
                metric_id=metric_id,
                operation="ranking",
                value=top_name,
                unit="rank",
                matching_rows=int(len(scoped)),
                formula_used=metric_def.formula,
                filters_applied=filters_applied,
                is_partial_period=is_partial,
                metadata={"ranking": [(str(k), float(v)) for k, v in grouped.items()]},
            )

        filtered = _filter_by_entity(scoped, entity, entity_type)

        if metric_id == "quantity_sold":
            val = float(filtered["QuantityNum"].sum()) if not filtered.empty else 0.0
            return MetricComputationResult(
                metric_id=metric_id,
                operation="value",
                value=val,
                unit="units",
                matching_rows=int(len(filtered)),
                formula_used=metric_def.formula,
                filters_applied=filters_applied,
                is_partial_period=is_partial,
            )

        if metric_id == "order_count":
            cnt = int(filtered["InvoiceNo"].dropna().nunique()) if not filtered.empty else 0
            return MetricComputationResult(
                metric_id=metric_id,
                operation="count",
                value=float(cnt),
                unit="count",
                matching_rows=int(len(filtered)),
                formula_used=metric_def.formula,
                filters_applied=filters_applied,
                is_partial_period=is_partial,
            )

        if metric_id == "average_order_value":
            orders = int(filtered["InvoiceNo"].dropna().nunique()) if not filtered.empty else 0
            if orders == 0:
                return MetricComputationResult(
                    metric_id=metric_id,
                    operation="value",
                    matching_rows=0,
                    filters_applied=filters_applied,
                    is_partial_period=is_partial,
                    error_reason="Zero distinct invoices prevent Average Order Value calculation.",
                )
            val = float(filtered["LineRevenue"].sum()) / orders
            return MetricComputationResult(
                metric_id=metric_id,
                operation="value",
                value=val,
                unit="currency",
                matching_rows=int(len(filtered)),
                formula_used=metric_def.formula,
                filters_applied=filters_applied,
                is_partial_period=is_partial,
            )

        if metric_id == "returns_amount":
            if operation == "count":
                val = float(len(filtered))
                unit = "count"
            else:
                val = abs(float(filtered["LineRevenue"].sum())) if not filtered.empty else 0.0
                unit = "currency"
            return MetricComputationResult(
                metric_id=metric_id,
                operation=operation,
                value=val,
                unit=unit,
                matching_rows=int(len(filtered)),
                formula_used=metric_def.formula,
                filters_applied=filters_applied,
                is_partial_period=is_partial,
            )

        if metric_id == "cancellation_analysis":
            if operation == "count":
                val = float(len(filtered))
                unit = "count"
            else:
                val = abs(float(filtered["LineRevenue"].sum())) if not filtered.empty else 0.0
                unit = "currency"
            return MetricComputationResult(
                metric_id=metric_id,
                operation=operation,
                value=val,
                unit=unit,
                matching_rows=int(len(filtered)),
                formula_used=metric_def.formula,
                filters_applied=filters_applied,
                is_partial_period=is_partial,
            )

        # Default revenue metrics: total_revenue, gross_revenue, net_revenue, monthly_revenue, product_revenue, country_revenue
        val = float(filtered["LineRevenue"].sum()) if not filtered.empty else 0.0
        return MetricComputationResult(
            metric_id=metric_id,
            operation="value",
            value=val,
            unit="currency",
            matching_rows=int(len(filtered)),
            formula_used=metric_def.formula,
            filters_applied=filters_applied,
            is_partial_period=is_partial,
        )
