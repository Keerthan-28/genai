"""Configurable column mapping for retail transaction datasets."""

from __future__ import annotations

import re

from backend.schemas import ColumnMappingResult

CANONICAL_REQUIRED_COLUMNS: tuple[str, ...] = (
    "InvoiceNo",
    "StockCode",
    "Description",
    "Quantity",
    "InvoiceDate",
    "UnitPrice",
    "Country",
)

CANONICAL_OPTIONAL_COLUMNS: tuple[str, ...] = ("CustomerID",)

CANONICAL_ALL_COLUMNS: tuple[str, ...] = CANONICAL_REQUIRED_COLUMNS + CANONICAL_OPTIONAL_COLUMNS

DEFAULT_COLUMN_ALIASES: dict[str, list[str]] = {
    "InvoiceNo": [
        "invoiceno",
        "invoice_no",
        "invoice",
        "invoicenumber",
        "invoice_id",
        "orderid",
        "order_id",
        "transaction_id",
        "bill_no",
    ],
    "StockCode": [
        "stockcode",
        "stock_code",
        "sku",
        "product_id",
        "product_code",
        "item_code",
        "item_id",
    ],
    "Description": [
        "description",
        "product",
        "product_name",
        "item_name",
        "item_description",
        "product_description",
        "name",
    ],
    "Quantity": [
        "quantity",
        "qty",
        "units",
        "units_sold",
        "item_qty",
        "count",
    ],
    "InvoiceDate": [
        "invoicedate",
        "invoice_date",
        "date",
        "order_date",
        "transaction_date",
        "datetime",
        "timestamp",
    ],
    "UnitPrice": [
        "unitprice",
        "unit_price",
        "price",
        "unit_cost",
        "price_per_unit",
        "rate",
    ],
    "CustomerID": [
        "customerid",
        "customer_id",
        "cust_id",
        "client_id",
        "buyer_id",
    ],
    "Country": [
        "country",
        "region",
        "market",
        "nation",
        "territory",
    ],
}


def _normalize_col_key(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(name).strip().lower())


def suggest_column_mapping(
    raw_columns: list[str],
    user_mapping: dict[str, str] | None = None,
) -> ColumnMappingResult:
    """Map raw dataset columns to canonical retail column names.

    Args:
        raw_columns: Exact column names present in the uploaded file.
        user_mapping: Optional explicit override `{canonical_name: raw_column_name}`.
    """
    raw_lookup = {col: col for col in raw_columns}
    norm_to_raw = {_normalize_col_key(col): col for col in raw_columns}

    resolved: dict[str, str] = {}
    used_raw: set[str] = set()

    # 1. Apply explicit user-supplied overrides first
    if user_mapping:
        for canonical, raw_col in user_mapping.items():
            if canonical in CANONICAL_ALL_COLUMNS and raw_col in raw_lookup:
                resolved[canonical] = raw_col
                used_raw.add(raw_col)

    # 2. Auto-detect exact and alias matches for remaining canonical columns
    for canonical in CANONICAL_ALL_COLUMNS:
        if canonical in resolved:
            continue
        if canonical in raw_lookup and canonical not in used_raw:
            resolved[canonical] = canonical
            used_raw.add(canonical)
            continue

        aliases = DEFAULT_COLUMN_ALIASES.get(canonical, [])
        for alias in aliases:
            candidate = norm_to_raw.get(_normalize_col_key(alias))
            if candidate and candidate not in used_raw:
                resolved[canonical] = candidate
                used_raw.add(candidate)
                break

    missing_required = [c for c in CANONICAL_REQUIRED_COLUMNS if c not in resolved]
    unmapped_raw = [c for c in raw_columns if c not in used_raw]

    return ColumnMappingResult(
        mapping=resolved,
        missing_required_columns=missing_required,
        unmapped_raw_columns=unmapped_raw,
        is_valid=len(missing_required) == 0,
    )


def apply_column_mapping(df, mapping: dict[str, str]):
    """Return a copy of `df` with mapped columns renamed to canonical names."""
    rename_map = {raw: canonical for canonical, raw in mapping.items() if raw in df.columns}
    return df.rename(columns=rename_map).copy()
