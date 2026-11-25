"""Dataset upload, validation, safe storage, and canonical loading."""

from __future__ import annotations

import hashlib
import io
import re
import uuid
from pathlib import Path
from typing import BinaryIO

import pandas as pd

from backend.column_mapper import apply_column_mapping, suggest_column_mapping
from backend.schemas import ColumnMappingResult, DatasetMetadata
from config.settings import Settings, get_settings


class DatasetValidationError(ValueError):
    """Raised when an uploaded dataset fails format, size, or schema validation."""


def sanitize_filename(filename: str) -> str:
    """Produce a safe single-segment filename without directory traversal characters."""
    base = Path(filename).name
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", base).strip("._")
    return cleaned or f"dataset_{uuid.uuid4().hex[:8]}.csv"


def compute_bytes_sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


class DataLoader:
    """Loads and validates CSV/Excel retail transaction datasets safely."""

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()

    def validate_file_properties(self, filename: str, file_size_bytes: int) -> str:
        ext = Path(filename).suffix.lower()
        if ext not in self.settings.allowed_extensions:
            raise DatasetValidationError(
                f"Unsupported file format '{ext}'. Allowed: {self.settings.allowed_extensions}"
            )
        if file_size_bytes <= 0:
            raise DatasetValidationError("Uploaded file is empty (0 bytes).")
        max_bytes = self.settings.max_upload_size_mb * 1024 * 1024
        if file_size_bytes > max_bytes:
            raise DatasetValidationError(
                f"File size ({file_size_bytes} bytes) exceeds limit of {self.settings.max_upload_size_mb} MB."
            )
        return ext

    @staticmethod
    def _read_dataframe_from_bytes(content: bytes, ext: str) -> pd.DataFrame:
        buffer = io.BytesIO(content)
        if ext == ".csv":
            for encoding in ("utf-8-sig", "utf-8", "latin-1", "cp1252"):
                try:
                    buffer.seek(0)
                    return pd.read_csv(buffer, encoding=encoding, low_memory=False)
                except UnicodeDecodeError:
                    continue
            raise DatasetValidationError("Could not decode CSV file with supported encodings.")
        if ext == ".xlsx":
            import openpyxl

            buffer.seek(0)
            wb = openpyxl.load_workbook(buffer, read_only=True, data_only=True)
            try:
                ws = wb.active
                rows_iter = ws.iter_rows(values_only=True)
                try:
                    header_row = next(rows_iter)
                except StopIteration:
                    return pd.DataFrame()
                headers = [
                    str(h).strip() if h is not None else f"col_{idx}"
                    for idx, h in enumerate(header_row)
                ]
                n_cols = len(headers)
                data = [
                    (row[:n_cols] if len(row) >= n_cols else row + (None,) * (n_cols - len(row)))
                    for row in rows_iter
                    if row and any(v is not None for v in row)
                ]
                return pd.DataFrame(data, columns=headers)
            finally:
                wb.close()
        if ext == ".xls":
            buffer.seek(0)
            return pd.read_excel(buffer)
        raise DatasetValidationError(f"Unsupported extension: {ext}")

    def load_from_path(
        self,
        file_path: str | Path,
        user_mapping: dict[str, str] | None = None,
        persist_raw_copy: bool = False,
    ) -> tuple[pd.DataFrame, DatasetMetadata, ColumnMappingResult]:
        path = Path(file_path)
        if not path.is_file():
            raise DatasetValidationError(f"Dataset file not found: {path}")
        content = path.read_bytes()
        return self.load_from_bytes(
            content=content,
            filename=path.name,
            user_mapping=user_mapping,
            persist_raw_copy=persist_raw_copy,
        )

    def load_from_stream(
        self,
        stream: BinaryIO,
        filename: str,
        user_mapping: dict[str, str] | None = None,
        persist_raw_copy: bool = True,
    ) -> tuple[pd.DataFrame, DatasetMetadata, ColumnMappingResult]:
        stream.seek(0)
        content = stream.read()
        return self.load_from_bytes(
            content=content,
            filename=filename,
            user_mapping=user_mapping,
            persist_raw_copy=persist_raw_copy,
        )

    def load_from_bytes(
        self,
        content: bytes,
        filename: str,
        user_mapping: dict[str, str] | None = None,
        persist_raw_copy: bool = True,
    ) -> tuple[pd.DataFrame, DatasetMetadata, ColumnMappingResult]:
        ext = self.validate_file_properties(filename, len(content))
        safe_name = sanitize_filename(filename)
        dataset_hash = compute_bytes_sha256(content)
        dataset_id = f"ds_{dataset_hash[:12]}"

        raw_df = self._read_dataframe_from_bytes(content, ext)
        if raw_df.empty or len(raw_df.columns) == 0:
            raise DatasetValidationError("Dataset contains no rows or columns.")

        # Strip surrounding whitespace from column headers
        raw_df.columns = [str(c).strip() for c in raw_df.columns]
        raw_columns = list(raw_df.columns)

        mapping_result = suggest_column_mapping(raw_columns, user_mapping=user_mapping)
        warnings: list[str] = []
        if not mapping_result.is_valid:
            warnings.append(
                f"Missing required mapped columns: {mapping_result.missing_required_columns}"
            )
        if "CustomerID" not in mapping_result.mapping:
            warnings.append("Optional column 'CustomerID' not present or unmapped.")

        if persist_raw_copy:
            self.settings.raw_data_dir.mkdir(parents=True, exist_ok=True)
            raw_copy_path = self.settings.raw_data_dir / f"{dataset_id}_{safe_name}"
            if not raw_copy_path.exists():
                raw_copy_path.write_bytes(content)

        canonical_df = apply_column_mapping(raw_df, mapping_result.mapping)

        metadata = DatasetMetadata(
            dataset_id=dataset_id,
            filename=Path(filename).name,
            safe_filename=safe_name,
            file_format=ext.lstrip("."),
            file_size_bytes=len(content),
            dataset_hash=dataset_hash,
            raw_row_count=int(len(raw_df)),
            raw_column_count=int(len(raw_columns)),
            raw_columns=raw_columns,
            column_mapping=mapping_result.mapping,
            warnings=warnings,
        )
        return canonical_df, metadata, mapping_result
