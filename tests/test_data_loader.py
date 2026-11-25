"""Phase 1 tests: dataset upload, validation, sanitization, and column mapping."""

from __future__ import annotations

import io

import pandas as pd
import pytest

from backend.column_mapper import suggest_column_mapping
from backend.data_loader import DataLoader, DatasetValidationError, sanitize_filename
from config.settings import PROJECT_ROOT


def test_load_sample_csv_dataset() -> None:
    loader = DataLoader()
    sample_path = PROJECT_ROOT / "data" / "samples" / "sample_retail_transactions.csv"
    df, metadata, mapping = loader.load_from_path(sample_path, persist_raw_copy=False)

    assert mapping.is_valid is True
    assert metadata.raw_row_count == 16
    assert metadata.file_format == "csv"
    assert len(metadata.dataset_hash) == 64
    assert "InvoiceNo" in df.columns
    assert "UnitPrice" in df.columns


def test_load_excel_dataset_and_alias_mapping(tmp_path) -> None:
    raw = pd.DataFrame(
        {
            "Order_ID": ["1001", "1002"],
            "SKU": ["A1", "A2"],
            "Product_Name": ["Widget", "Gadget"],
            "Qty": [2, 3],
            "Order_Date": ["2024-11-01", "2024-11-02"],
            "Price": [10.0, 20.0],
            "Region": ["UK", "France"],
        }
    )
    xlsx_path = tmp_path / "retail_export.xlsx"
    raw.to_excel(xlsx_path, index=False)

    loader = DataLoader()
    df, metadata, mapping = loader.load_from_path(xlsx_path, persist_raw_copy=False)

    assert mapping.is_valid is True
    assert mapping.mapping["InvoiceNo"] == "Order_ID"
    assert mapping.mapping["Description"] == "Product_Name"
    assert mapping.mapping["UnitPrice"] == "Price"
    assert mapping.mapping["Country"] == "Region"
    assert "InvoiceNo" in df.columns
    assert metadata.file_format == "xlsx"


def test_missing_required_columns_flagged() -> None:
    mapping = suggest_column_mapping(["InvoiceNo", "Quantity"])
    assert mapping.is_valid is False
    assert set(["StockCode", "Description", "InvoiceDate", "UnitPrice", "Country"]).issubset(
        set(mapping.missing_required_columns)
    )


def test_reject_unsupported_extension_and_empty_file() -> None:
    loader = DataLoader()
    with pytest.raises(DatasetValidationError):
        loader.load_from_bytes(b"col1,col2\n1,2\n", "bad.json", persist_raw_copy=False)
    with pytest.raises(DatasetValidationError):
        loader.load_from_bytes(b"", "empty.csv", persist_raw_copy=False)


def test_sanitize_filename_blocks_traversal() -> None:
    cleaned = sanitize_filename("../../etc/passwd;drop.csv")
    assert "/" not in cleaned
    assert "\\" not in cleaned
    assert ".." not in cleaned
    assert cleaned.endswith(".csv")


def test_load_from_stream() -> None:
    loader = DataLoader()
    csv_bytes = (
        b"InvoiceNo,StockCode,Description,Quantity,InvoiceDate,UnitPrice,Country\n"
        b"100,S1,Item,2,2024-10-01,15.5,UK\n"
    )
    df, metadata, mapping = loader.load_from_stream(
        io.BytesIO(csv_bytes), "stream_test.csv", persist_raw_copy=False
    )
    assert mapping.is_valid is True
    assert len(df) == 1
    assert metadata.dataset_id.startswith("ds_")
