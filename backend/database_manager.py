"""Phase 11: SQLite audit database schema and connection manager (11 tables)."""

from __future__ import annotations

from contextlib import contextmanager
import sqlite3
from pathlib import Path
from typing import Iterator

from config.settings import Settings, get_settings

SCHEMA_DDL = """
PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS audit_runs (
    run_id TEXT PRIMARY KEY,
    dataset_id TEXT NOT NULL,
    report_id TEXT,
    transaction_policy_json TEXT NOT NULL,
    system_version TEXT NOT NULL,
    metric_registry_version TEXT NOT NULL,
    total_claims INTEGER NOT NULL DEFAULT 0,
    verified_count INTEGER NOT NULL DEFAULT 0,
    contradicted_count INTEGER NOT NULL DEFAULT 0,
    insufficient_count INTEGER NOT NULL DEFAULT 0,
    needs_review_count INTEGER NOT NULL DEFAULT 0,
    partially_verified_count INTEGER NOT NULL DEFAULT 0,
    processing_time_seconds REAL NOT NULL DEFAULT 0.0,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS dataset_metadata (
    dataset_id TEXT PRIMARY KEY,
    filename TEXT NOT NULL,
    safe_filename TEXT NOT NULL,
    file_format TEXT NOT NULL,
    file_size_bytes INTEGER NOT NULL,
    dataset_hash TEXT NOT NULL,
    raw_row_count INTEGER NOT NULL,
    processed_row_count INTEGER NOT NULL,
    column_mapping_json TEXT NOT NULL,
    data_quality_score REAL NOT NULL,
    date_min TEXT,
    date_max TEXT,
    partial_periods_json TEXT NOT NULL,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS generated_reports (
    report_id TEXT PRIMARY KEY,
    run_id TEXT NOT NULL,
    dataset_id TEXT NOT NULL,
    report_hash TEXT NOT NULL,
    model_used TEXT NOT NULL,
    generation_mode TEXT NOT NULL,
    generation_time_seconds REAL NOT NULL,
    title TEXT NOT NULL,
    full_text TEXT NOT NULL,
    txt_path TEXT,
    docx_path TEXT,
    pdf_path TEXT,
    created_at TEXT NOT NULL,
    FOREIGN KEY (run_id) REFERENCES audit_runs(run_id) ON DELETE CASCADE,
    FOREIGN KEY (dataset_id) REFERENCES dataset_metadata(dataset_id)
);

CREATE TABLE IF NOT EXISTS claims (
    claim_uid TEXT PRIMARY KEY,
    run_id TEXT NOT NULL,
    claim_id TEXT NOT NULL,
    claim_text TEXT NOT NULL,
    source_sentence TEXT NOT NULL,
    report_section TEXT NOT NULL,
    raw_metric TEXT,
    metric_id TEXT,
    raw_entity TEXT,
    normalized_entity TEXT,
    entity_type TEXT,
    raw_period TEXT,
    normalized_period TEXT,
    normalized_comparison_period TEXT,
    operation TEXT NOT NULL,
    claimed_value TEXT,
    normalized_unit TEXT,
    extraction_confidence REAL NOT NULL,
    ambiguity_flag INTEGER NOT NULL DEFAULT 0,
    FOREIGN KEY (run_id) REFERENCES audit_runs(run_id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS claim_subclaims (
    subclaim_uid TEXT PRIMARY KEY,
    run_id TEXT NOT NULL,
    parent_claim_uid TEXT NOT NULL,
    subclaim_id TEXT NOT NULL,
    claim_text TEXT NOT NULL,
    metric_id TEXT,
    normalized_entity TEXT,
    entity_type TEXT,
    normalized_period TEXT,
    normalized_comparison_period TEXT,
    operation TEXT NOT NULL,
    claimed_value TEXT,
    normalized_unit TEXT,
    ambiguity_flag INTEGER NOT NULL DEFAULT 0,
    FOREIGN KEY (run_id) REFERENCES audit_runs(run_id) ON DELETE CASCADE,
    FOREIGN KEY (parent_claim_uid) REFERENCES claims(claim_uid) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS verification_results (
    verification_uid TEXT PRIMARY KEY,
    run_id TEXT NOT NULL,
    claim_uid TEXT NOT NULL,
    subclaim_uid TEXT,
    claim_id TEXT NOT NULL,
    status TEXT NOT NULL,
    original_machine_status TEXT NOT NULL,
    actual_value TEXT,
    formula_used TEXT,
    applied_filters_json TEXT NOT NULL,
    explanation TEXT NOT NULL,
    explanation_source TEXT NOT NULL,
    needs_human_review INTEGER NOT NULL DEFAULT 0,
    review_reason TEXT,
    verified_at TEXT NOT NULL,
    FOREIGN KEY (run_id) REFERENCES audit_runs(run_id) ON DELETE CASCADE,
    FOREIGN KEY (claim_uid) REFERENCES claims(claim_uid) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS evidence_records (
    evidence_uid TEXT PRIMARY KEY,
    run_id TEXT NOT NULL,
    claim_uid TEXT NOT NULL,
    claim_id TEXT NOT NULL,
    has_sufficient_evidence INTEGER NOT NULL,
    reason_code TEXT NOT NULL,
    reason_detail TEXT NOT NULL,
    matching_records_count INTEGER NOT NULL,
    is_partial_period INTEGER NOT NULL,
    units_compatible INTEGER NOT NULL,
    FOREIGN KEY (run_id) REFERENCES audit_runs(run_id) ON DELETE CASCADE,
    FOREIGN KEY (claim_uid) REFERENCES claims(claim_uid) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS gap_analysis (
    gap_uid TEXT PRIMARY KEY,
    run_id TEXT NOT NULL,
    claim_uid TEXT NOT NULL,
    claim_id TEXT NOT NULL,
    claimed_value TEXT,
    actual_value TEXT,
    signed_gap REAL,
    absolute_gap REAL,
    relative_gap_pct REAL,
    percentage_point_diff REAL,
    gap_type TEXT NOT NULL,
    unit TEXT,
    tolerance REAL,
    comparison_method TEXT NOT NULL,
    notes TEXT,
    FOREIGN KEY (run_id) REFERENCES audit_runs(run_id) ON DELETE CASCADE,
    FOREIGN KEY (claim_uid) REFERENCES claims(claim_uid) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS evidence_quality (
    quality_uid TEXT PRIMARY KEY,
    run_id TEXT NOT NULL,
    claim_uid TEXT NOT NULL,
    claim_id TEXT NOT NULL,
    quality_level TEXT NOT NULL,
    score REAL NOT NULL,
    dimension_findings_json TEXT NOT NULL,
    explanation TEXT NOT NULL,
    FOREIGN KEY (run_id) REFERENCES audit_runs(run_id) ON DELETE CASCADE,
    FOREIGN KEY (claim_uid) REFERENCES claims(claim_uid) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS consistency_findings (
    finding_uid TEXT PRIMARY KEY,
    run_id TEXT NOT NULL,
    finding_id TEXT NOT NULL,
    related_claim_ids_json TEXT NOT NULL,
    relationship_type TEXT NOT NULL,
    expected_value TEXT,
    observed_value TEXT,
    difference REAL,
    is_consistent INTEGER NOT NULL,
    explanation TEXT NOT NULL,
    FOREIGN KEY (run_id) REFERENCES audit_runs(run_id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS review_actions (
    review_id TEXT PRIMARY KEY,
    run_id TEXT NOT NULL,
    claim_id TEXT NOT NULL,
    reviewer TEXT NOT NULL,
    action_type TEXT NOT NULL,
    previous_status TEXT NOT NULL,
    new_status TEXT NOT NULL,
    corrected_metric_id TEXT,
    corrected_period TEXT,
    corrected_entity TEXT,
    external_evidence_ref TEXT,
    notes TEXT NOT NULL,
    created_at TEXT NOT NULL,
    FOREIGN KEY (run_id) REFERENCES audit_runs(run_id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_claims_run ON claims(run_id);
CREATE INDEX IF NOT EXISTS idx_ver_run_status ON verification_results(run_id, status);
CREATE INDEX IF NOT EXISTS idx_gap_run_type ON gap_analysis(run_id, gap_type);
CREATE INDEX IF NOT EXISTS idx_review_run_claim ON review_actions(run_id, claim_id);
"""


class DatabaseManager:
    """Manages SQLite connections and schema initialization."""

    def __init__(self, db_path: Path | str | None = None, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        self.db_path = Path(db_path) if db_path else self.settings.database_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.initialize_schema()

    @contextmanager
    def connect(self) -> Iterator[sqlite3.Connection]:
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON;")
        try:
            with conn:
                yield conn
        finally:
            conn.close()

    def initialize_schema(self) -> None:
        with self.connect() as conn:
            conn.executescript(SCHEMA_DDL)

    def list_tables(self) -> list[str]:
        with self.connect() as conn:
            rows = conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%' ORDER BY name"
            ).fetchall()
        return [str(r["name"]) for r in rows]
