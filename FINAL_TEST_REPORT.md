# VeriReport AI — Final Test Report

## 1. Test Execution Summary

| Suite / Command | Passed | Failed | Skipped | Duration | Environment |
|---|---|---|---|---|---|
| `.\.venv\Scripts\python.exe -m pytest -v` | 41 | 0 | 1 | 30.51s | Sandboxed offline suite |
| `RUN_LIVE_OLLAMA_TESTS=1 .\.venv\Scripts\python.exe -m pytest tests/test_integration.py -k "test_live_ollama_end_to_end" -v` | 1 | 0 | 0 | 151.64s | Live local Ollama (`llama3.2:latest`) |
| **Combined Verified Total** | **42** | **0** | **0** | — | All 42 tests verified |

## 2. Test Coverage by Module & Phase

- **Phase 0 (`tests/test_phase0_setup.py` — 8 passed)**: Directory structure, `.gitignore` secret protection, package imports (`pandas`, `numpy`, `requests`, `pydantic`, `pymupdf`, `docx`, `openpyxl`, `streamlit`, `plotly`, `pytest`, `dotenv`, `sqlite3`), Ollama client unreachable fallback, model verification, and embedding-model filtering.
- **Phase 1 (`tests/test_data_loader.py` — 6 passed)**: CSV/Excel loading, alias column mapping, missing required column detection, invalid extension/empty file rejection, filename traversal sanitization, stream ingestion.
- **Phase 2 (`tests/test_preprocessing.py` — 3 passed)**: Return/cancellation preservation under `NET_REVENUE`, `GROSS_SALES` policy filtering, duplicate/invalid date/zero-price detection, and partial-period coverage detection.
- **Phase 3 (`tests/test_analytics.py` — 3 passed)**: `MetricRegistry` coverage & unsupported metric blocking, deterministic revenue/growth/product/country calculations, and zero-denominator growth handling.
- **Phase 4 (`tests/test_report_generator.py` — 2 passed)**: 11-section deterministic fallback report, TXT/DOCX/PDF export generation & read-back verification, and mocked Ollama report generation.
- **Phase 5 (`tests/test_claim_extraction.py` — 3 passed)**: Compound sentence splitting (`subclaims`), malformed JSON recovery, empty input handling, and untrusted report prompt-injection boundary preservation.
- **Phase 6 (`tests/test_claim_mapping.py` — 2 passed)**: Normalization of entities/periods/units and mapping to `MetricRegistry` (including explicit unsupported vs ambiguous routing).
- **Phase 7 (`tests/test_evidence_gate.py` — 2 passed)**: Evidence sufficiency acceptance with partial-period flagging, plus rejection of unsupported metrics, uncovered periods, and incompatible currency units.
- **Phase 8 (`tests/test_verification.py` + `tests/test_gap_analysis.py` — 2 passed)**: Independent recalculation, `EXACT_MATCH`, `OVERSTATED`, `UNDERSTATED`, `PERIOD_MISMATCH`, `METRIC_MISMATCH`, `NOT_COMPARABLE`, percentage-point gaps, and zero-actual handling.
- **Phase 9 (`tests/test_evidence_quality.py` + `tests/test_status_engine.py` — 2 passed)**: 7-dimension evidence quality (`HIGH`/`MEDIUM`/`LOW`) decoupled from truth, plus independent verification of all 5 statuses (`VERIFIED`, `CONTRADICTED`, `INSUFFICIENT_EVIDENCE`, `NEEDS_REVIEW`, `PARTIALLY_VERIFIED`).
- **Phase 10 (`tests/test_consistency.py` + `tests/test_review.py` — 2 passed)**: Cross-claim checks (product sum vs total, contribution sum >100%, MoM growth vs monthly revenues, ranking conflicts) and non-destructive human review preserving `original_machine_status`.
- **Phase 11 (`tests/test_database.py` — 2 passed)**: All 11 SQLite tables verified, full audit persistence, review action history preservation, CSV/JSON export, and proof that LLM explanations cannot alter Python statuses or numbers.
- **Phase 13 (`tests/test_integration.py` — 5 passed across offline + live runs)**: Sample dataset E2E, synthetic 10-claim benchmark, missing-column/poor-quality dataset handling, Streamlit dashboard module compilation & 10-page helper check, and live Ollama (`llama3.2:latest`) E2E report generation + extraction + verification + SQLite persistence.

## 3. Observed Synthetic Benchmark Metrics

Measured directly against `data/samples/synthetic_audit_dataset.csv` and `data/samples/synthetic_sales_report.md` (10 ground-truth claims covering all 5 statuses, compound claims, period mismatch, metric mismatch, zero comparison-period denominator, unsupported metric, and ambiguous metric):

- **Claim Extraction**: Precision = `1.00`, Recall = `1.00`, F1 = `1.00` (10 / 10 sentences extracted)
- **Verification Classification Accuracy**: `1.00` (10 / 10)
- **Per-Status Precision & Recall**:
  - `VERIFIED`: Precision `1.00`, Recall `1.00` (`tp=2, fp=0, fn=0`)
  - `CONTRADICTED`: Precision `1.00`, Recall `1.00` (`tp=4, fp=0, fn=0`)
  - `INSUFFICIENT_EVIDENCE`: Precision `1.00`, Recall `1.00` (`tp=2, fp=0, fn=0`)
  - `NEEDS_REVIEW`: Precision `1.00`, Recall `1.00` (`tp=1, fp=0, fn=0`)
  - `PARTIALLY_VERIFIED`: Precision `1.00`, Recall `1.00` (`tp=1, fp=0, fn=0`)
- **Evidence Quality Distribution**: `{'MEDIUM': 5, 'HIGH': 2, 'LOW': 3}`
- **Human Review Frequency**: `0.10` (1 / 10)
- **End-to-End Deterministic Benchmark Runtime**: `0.1728s`
- **Offline Sample CLI Runtime (`main.py --sample --no-llm`)**: `0.2746s` (26 extracted claims, 13 `VERIFIED`, 1 `CONTRADICTED`, 2 `INSUFFICIENT_EVIDENCE`, 8 `NEEDS_REVIEW`, 2 `PARTIALLY_VERIFIED`, 1 consistency finding)
- **Live Ollama E2E Test Runtime (`llama3.2:latest`)**: `151.64s`
