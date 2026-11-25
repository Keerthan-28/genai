# VeriReport AI — Architecture Documentation

## 1. Core Architectural Principle

VeriReport AI separates **natural-language generation/parsing** from **numerical computation and verification**:

1. **Local LLM (Ollama)** drafts human-readable sales reports from structured Python metrics, parses unstructured report sentences into candidate structured claims, and writes readable explanations from verification engine outputs.
2. **Python & Pandas (`analytics_engine.py`, `preprocessing.py`)** execute every numerical calculation deterministically from the processed transaction records.
3. **Deterministic Verification Engine (`evidence_gate.py`, `claim_verifier.py`, `gap_analyzer.py`, `status_engine.py`)** independently recalculates actual values and assigns one of the five verification statuses. The LLM **never** evaluates or overrides its own numerical claims.

## 2. 22-Step Processing Pipeline

1. **Dataset Upload & Validation (`data_loader.py`, `column_mapper.py`)**: Validates extension (`.csv`, `.xlsx`, `.xls`), file size, filename safety, SHA-256 hash, and maps columns via configurable aliases.
2. **Preprocessing & Profiling (`preprocessing.py`, `data_profiler.py`)**: Preserves raw records, classifies cancellations (`C` invoice prefix) and returns (`Quantity < 0`), logs every transformation under a `TransactionPolicy`, and computes monthly/yearly coverage to flag partial periods.
3. **Analytics & Metric Registry (`metric_registry.py`, `analytics_engine.py`)**: Defines formulas, required columns, supported operations, and tolerances; computes all 14 retail analytics metrics deterministically.
4. **Report Generation (`report_generator.py`, `report_templates.py`)**: Sends only structured Python analytics to Ollama (with deterministic fallback), producing an 11-section report with preserved sentence references and PDF (`PyMuPDF`), DOCX (`python-docx`), and TXT exports.
5. **Claim Extraction & Validation (`claim_extractor.py`, `claim_validator.py`)**: Extracts atomic & compound claims (`subclaims`), defends against prompt injection via `<UNTRUSTED_REPORT>` boundaries, and validates with Pydantic v2.
6. **Normalization & Mapping (`claim_normalizer.py`, `claim_mapper.py`)**: Canonicalizes units, periods (`YYYY-MM`), entities against dataset vocabulary, and maps claims to `MetricRegistry` IDs.
7. **Evidence Sufficiency Gate (`evidence_gate.py`)**: Checks required columns, metric support, entity identifiability, period & comparison-period coverage, matching records, unit compatibility, and data quality before numerical comparison.
8. **Independent Verification & Gap Analysis (`claim_verifier.py`, `tolerance_manager.py`, `gap_analyzer.py`)**: Recalculates actual values, applies metric-specific tolerances, and classifies `EXACT_MATCH`, `OVERSTATED`, `UNDERSTATED`, `PERIOD_MISMATCH`, `METRIC_MISMATCH`, or `NOT_COMPARABLE`.
9. **Evidence Quality & Status Engine (`evidence_quality.py`, `status_engine.py`)**: Evaluates 7 evidence-quality dimensions (`HIGH`, `MEDIUM`, `LOW`) independently from claim truth, and assigns `VERIFIED`, `CONTRADICTED`, `INSUFFICIENT_EVIDENCE`, `NEEDS_REVIEW`, or `PARTIALLY_VERIFIED`.
10. **Cross-Claim Consistency & Human Review (`consistency_checker.py`, `review_manager.py`)**: Flags arithmetic/ranking contradictions across claims and manages human review overrides while preserving `original_machine_status`.
11. **Explanations & SQLite Audit Logging (`explanation_generator.py`, `database_manager.py`, `audit_logger.py`)**: Generates explanations grounded in verification results and persists the full audit trail across 11 SQLite tables without storing unnecessary customer identifiers.
12. **Interactive Dashboard (`frontend/app.py`)**: 10-page Streamlit UI with Plotly visualizations, interactive review workflow, and CSV/JSON/Markdown/PDF/DOCX/TXT exports.

## 3. SQLite Audit Schema (11 Tables)

- `audit_runs`
- `dataset_metadata`
- `generated_reports`
- `claims`
- `claim_subclaims`
- `verification_results`
- `evidence_records`
- `gap_analysis`
- `evidence_quality`
- `consistency_findings`
- `review_actions`
