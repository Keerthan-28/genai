"""Phase 11: Audit logger for persisting, querying, and exporting verification runs."""

from __future__ import annotations

import csv
import json
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any

from backend.database_manager import DatabaseManager
from backend.schemas import (
    ClaimVerificationResult,
    ConsistencyFinding,
    DatasetMetadata,
    DatasetProfile,
    GeneratedReport,
    ReviewAction,
    TransactionPolicy,
    VerificationStatus,
)
from config.settings import Settings, get_settings


class AuditLogger:
    """Persists full audit trails and generates CSV/JSON/Markdown exports."""

    def __init__(
        self,
        db_manager: DatabaseManager | None = None,
        settings: Settings | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.db = db_manager or DatabaseManager(settings=self.settings)

    def save_full_audit_run(
        self,
        metadata: DatasetMetadata,
        profile: DatasetProfile,
        policy: TransactionPolicy,
        report: GeneratedReport,
        results: list[ClaimVerificationResult],
        consistency_findings: list[ConsistencyFinding],
        processing_time_seconds: float = 0.0,
        run_id: str | None = None,
    ) -> str:
        run_id = run_id or f"run_{uuid.uuid4().hex[:10]}"
        now = datetime.utcnow().isoformat()

        counts = {s.value: 0 for s in VerificationStatus}
        for r in results:
            counts[r.status.value] += 1

        with self.db.connect() as conn:
            # 1. dataset_metadata (upsert by dataset_id)
            conn.execute(
                """
                INSERT OR REPLACE INTO dataset_metadata (
                    dataset_id, filename, safe_filename, file_format, file_size_bytes,
                    dataset_hash, raw_row_count, processed_row_count, column_mapping_json,
                    data_quality_score, date_min, date_max, partial_periods_json, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    metadata.dataset_id,
                    metadata.filename,
                    metadata.safe_filename,
                    metadata.file_format,
                    metadata.file_size_bytes,
                    metadata.dataset_hash,
                    profile.total_raw_rows,
                    profile.total_processed_rows,
                    json.dumps(metadata.column_mapping),
                    profile.data_quality_score,
                    profile.date_min,
                    profile.date_max,
                    json.dumps(profile.partial_periods),
                    metadata.loaded_at,
                ),
            )

            # 2. audit_runs
            conn.execute(
                """
                INSERT INTO audit_runs (
                    run_id, dataset_id, report_id, transaction_policy_json,
                    system_version, metric_registry_version, total_claims,
                    verified_count, contradicted_count, insufficient_count,
                    needs_review_count, partially_verified_count, processing_time_seconds, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    run_id,
                    metadata.dataset_id,
                    report.report_id,
                    json.dumps(policy.model_dump(mode="json")),
                    self.settings.app_version,
                    "v1.0",
                    len(results),
                    counts[VerificationStatus.VERIFIED.value],
                    counts[VerificationStatus.CONTRADICTED.value],
                    counts[VerificationStatus.INSUFFICIENT_EVIDENCE.value],
                    counts[VerificationStatus.NEEDS_REVIEW.value],
                    counts[VerificationStatus.PARTIALLY_VERIFIED.value],
                    processing_time_seconds,
                    now,
                ),
            )

            # 3. generated_reports
            conn.execute(
                """
                INSERT INTO generated_reports (
                    report_id, run_id, dataset_id, report_hash, model_used,
                    generation_mode, generation_time_seconds, title, full_text,
                    txt_path, docx_path, pdf_path, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    report.report_id,
                    run_id,
                    metadata.dataset_id,
                    report.report_hash,
                    report.model_used,
                    report.generation_mode,
                    report.generation_time_seconds,
                    report.title,
                    report.full_text,
                    report.txt_path,
                    report.docx_path,
                    report.pdf_path,
                    report.generated_at,
                ),
            )

            # 4-9. claims, subclaims, verification_results, evidence_records, gap_analysis, evidence_quality
            for r in results:
                c = r.normalized_claim
                claim_uid = f"{run_id}_{r.claim_id}"
                conn.execute(
                    """
                    INSERT INTO claims (
                        claim_uid, run_id, claim_id, claim_text, source_sentence, report_section,
                        raw_metric, metric_id, raw_entity, normalized_entity, entity_type,
                        raw_period, normalized_period, normalized_comparison_period, operation,
                        claimed_value, normalized_unit, extraction_confidence, ambiguity_flag
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        claim_uid,
                        run_id,
                        r.claim_id,
                        c.claim_text,
                        c.source_sentence,
                        c.report_section,
                        c.raw_metric,
                        c.metric_id,
                        c.raw_entity,
                        c.normalized_entity,
                        c.entity_type,
                        c.raw_period,
                        c.normalized_period,
                        c.normalized_comparison_period,
                        c.operation,
                        None if c.claimed_value is None else str(c.claimed_value),
                        c.normalized_unit,
                        c.extraction_confidence,
                        int(c.ambiguity_flag),
                    ),
                )

                self._insert_verification_artifacts(conn, run_id, claim_uid, None, r, now)

                for sub in r.subclaim_results:
                    sc = sub.normalized_claim
                    sub_uid = f"{run_id}_{sub.claim_id}"
                    conn.execute(
                        """
                        INSERT INTO claim_subclaims (
                            subclaim_uid, run_id, parent_claim_uid, subclaim_id, claim_text,
                            metric_id, normalized_entity, entity_type, normalized_period,
                            normalized_comparison_period, operation, claimed_value,
                            normalized_unit, ambiguity_flag
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                        (
                            sub_uid,
                            run_id,
                            claim_uid,
                            sub.claim_id,
                            sc.claim_text,
                            sc.metric_id,
                            sc.normalized_entity,
                            sc.entity_type,
                            sc.normalized_period,
                            sc.normalized_comparison_period,
                            sc.operation,
                            None if sc.claimed_value is None else str(sc.claimed_value),
                            sc.normalized_unit,
                            int(sc.ambiguity_flag),
                        ),
                    )
                    self._insert_verification_artifacts(conn, run_id, claim_uid, sub_uid, sub, now)

            # 10. consistency_findings
            for f in consistency_findings:
                conn.execute(
                    """
                    INSERT INTO consistency_findings (
                        finding_uid, run_id, finding_id, related_claim_ids_json,
                        relationship_type, expected_value, observed_value,
                        difference, is_consistent, explanation
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        f"{run_id}_{f.finding_id}",
                        run_id,
                        f.finding_id,
                        json.dumps(f.related_claim_ids),
                        f.relationship_type,
                        None if f.expected_value is None else str(f.expected_value),
                        None if f.observed_value is None else str(f.observed_value),
                        f.difference,
                        int(f.is_consistent),
                        f.explanation,
                    ),
                )

        return run_id

    @staticmethod
    def _insert_verification_artifacts(
        conn,
        run_id: str,
        claim_uid: str,
        subclaim_uid: str | None,
        r: ClaimVerificationResult,
        now: str,
    ) -> None:
        key_suffix = subclaim_uid or claim_uid
        conn.execute(
            """
            INSERT INTO verification_results (
                verification_uid, run_id, claim_uid, subclaim_uid, claim_id,
                status, original_machine_status, actual_value, formula_used,
                applied_filters_json, explanation, explanation_source,
                needs_human_review, review_reason, verified_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                f"ver_{key_suffix}",
                run_id,
                claim_uid,
                subclaim_uid,
                r.claim_id,
                r.status.value,
                r.original_machine_status.value,
                None if r.actual_value is None else str(r.actual_value),
                r.formula_used,
                json.dumps(r.applied_filters),
                r.explanation,
                r.explanation_source,
                int(r.needs_human_review),
                r.review_reason,
                now,
            ),
        )

        eg = r.evidence_gate
        conn.execute(
            """
            INSERT INTO evidence_records (
                evidence_uid, run_id, claim_uid, claim_id, has_sufficient_evidence,
                reason_code, reason_detail, matching_records_count,
                is_partial_period, units_compatible
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                f"ev_{key_suffix}",
                run_id,
                claim_uid,
                r.claim_id,
                int(eg.has_sufficient_evidence),
                eg.reason_code.value,
                eg.reason_detail,
                eg.matching_records_count,
                int(eg.is_partial_period),
                int(eg.units_compatible),
            ),
        )

        gap = r.gap_analysis
        conn.execute(
            """
            INSERT INTO gap_analysis (
                gap_uid, run_id, claim_uid, claim_id, claimed_value, actual_value,
                signed_gap, absolute_gap, relative_gap_pct, percentage_point_diff,
                gap_type, unit, tolerance, comparison_method, notes
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                f"gap_{key_suffix}",
                run_id,
                claim_uid,
                r.claim_id,
                None if gap.claimed_value is None else str(gap.claimed_value),
                None if gap.actual_value is None else str(gap.actual_value),
                gap.signed_gap,
                gap.absolute_gap,
                gap.relative_gap_pct,
                gap.percentage_point_diff,
                gap.gap_type.value,
                gap.unit,
                gap.tolerance,
                gap.comparison_method,
                gap.notes,
            ),
        )

        eq = r.evidence_quality
        conn.execute(
            """
            INSERT INTO evidence_quality (
                quality_uid, run_id, claim_uid, claim_id, quality_level,
                score, dimension_findings_json, explanation
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                f"qual_{key_suffix}",
                run_id,
                claim_uid,
                r.claim_id,
                eq.quality_level.value,
                eq.score,
                json.dumps(eq.dimension_findings),
                eq.explanation,
            ),
        )

    def record_review_action(
        self,
        run_id: str,
        updated_result: ClaimVerificationResult,
        action: ReviewAction,
    ) -> None:
        """Persist human review action and update current status while preserving original_machine_status."""
        with self.db.connect() as conn:
            conn.execute(
                """
                INSERT INTO review_actions (
                    review_id, run_id, claim_id, reviewer, action_type,
                    previous_status, new_status, corrected_metric_id,
                    corrected_period, corrected_entity, external_evidence_ref,
                    notes, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    action.review_id,
                    run_id,
                    action.claim_id,
                    action.reviewer,
                    action.action_type,
                    action.previous_status.value,
                    action.new_status.value,
                    action.corrected_metric_id,
                    action.corrected_period,
                    action.corrected_entity,
                    action.external_evidence_ref,
                    action.notes,
                    action.created_at,
                ),
            )
            conn.execute(
                """
                UPDATE verification_results
                SET status = ?,
                    actual_value = ?,
                    needs_human_review = ?,
                    review_reason = ?,
                    explanation = ?
                WHERE run_id = ? AND claim_id = ? AND subclaim_uid IS NULL
                """,
                (
                    updated_result.status.value,
                    None if updated_result.actual_value is None else str(updated_result.actual_value),
                    int(updated_result.needs_human_review),
                    updated_result.review_reason,
                    updated_result.explanation,
                    run_id,
                    updated_result.claim_id,
                ),
            )
            # Refresh run summary counts
            rows = conn.execute(
                "SELECT status FROM verification_results WHERE run_id = ? AND subclaim_uid IS NULL",
                (run_id,),
            ).fetchall()
            counts = {s.value: 0 for s in VerificationStatus}
            for row in rows:
                counts[row["status"]] = counts.get(row["status"], 0) + 1
            conn.execute(
                """
                UPDATE audit_runs
                SET verified_count = ?, contradicted_count = ?, insufficient_count = ?,
                    needs_review_count = ?, partially_verified_count = ?
                WHERE run_id = ?
                """,
                (
                    counts[VerificationStatus.VERIFIED.value],
                    counts[VerificationStatus.CONTRADICTED.value],
                    counts[VerificationStatus.INSUFFICIENT_EVIDENCE.value],
                    counts[VerificationStatus.NEEDS_REVIEW.value],
                    counts[VerificationStatus.PARTIALLY_VERIFIED.value],
                    run_id,
                ),
            )

    def list_audit_runs(self) -> list[dict[str, Any]]:
        with self.db.connect() as conn:
            rows = conn.execute(
                """
                SELECT a.*, d.filename, d.dataset_hash, r.title AS report_title, r.model_used
                FROM audit_runs a
                LEFT JOIN dataset_metadata d ON a.dataset_id = d.dataset_id
                LEFT JOIN generated_reports r ON a.report_id = r.report_id
                ORDER BY a.created_at DESC
                """
            ).fetchall()
        return [dict(r) for r in rows]

    def get_run_details(self, run_id: str) -> dict[str, Any]:
        with self.db.connect() as conn:
            run_row = conn.execute("SELECT * FROM audit_runs WHERE run_id = ?", (run_id,)).fetchone()
            ver_rows = conn.execute(
                """
                SELECT v.*, c.claim_text, c.source_sentence, c.report_section, c.metric_id,
                       c.normalized_period, c.normalized_entity,
                       g.claimed_value, g.signed_gap, g.absolute_gap, g.relative_gap_pct, g.gap_type, g.unit,
                       eq.quality_level, eq.score AS quality_score,
                       ev.reason_code, ev.reason_detail, ev.matching_records_count
                FROM verification_results v
                JOIN claims c ON v.claim_uid = c.claim_uid
                LEFT JOIN gap_analysis g ON g.run_id = v.run_id AND g.claim_id = v.claim_id AND v.subclaim_uid IS NULL
                LEFT JOIN evidence_quality eq ON eq.run_id = v.run_id AND eq.claim_id = v.claim_id AND v.subclaim_uid IS NULL
                LEFT JOIN evidence_records ev ON ev.run_id = v.run_id AND ev.claim_id = v.claim_id AND v.subclaim_uid IS NULL
                WHERE v.run_id = ? AND v.subclaim_uid IS NULL
                ORDER BY v.claim_id
                """,
                (run_id,),
            ).fetchall()
            con_rows = conn.execute(
                "SELECT * FROM consistency_findings WHERE run_id = ?", (run_id,)
            ).fetchall()
            rev_rows = conn.execute(
                "SELECT * FROM review_actions WHERE run_id = ? ORDER BY created_at DESC", (run_id,)
            ).fetchall()

        return {
            "run": dict(run_row) if run_row else None,
            "verifications": [dict(r) for r in ver_rows],
            "consistency_findings": [dict(r) for r in con_rows],
            "review_actions": [dict(r) for r in rev_rows],
        }

    def export_run_to_csv(self, run_id: str, output_path: Path | None = None) -> Path:
        details = self.get_run_details(run_id)
        out = output_path or (self.settings.exports_dir / f"{run_id}_verification.csv")
        out.parent.mkdir(parents=True, exist_ok=True)
        rows = details["verifications"]
        fieldnames = (
            list(rows[0].keys())
            if rows
            else ["claim_id", "status", "original_machine_status", "claimed_value", "actual_value", "gap_type"]
        )
        with out.open("w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        return out

    def export_run_to_json(self, run_id: str, output_path: Path | None = None) -> Path:
        details = self.get_run_details(run_id)
        out = output_path or (self.settings.exports_dir / f"{run_id}_verification.json")
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(details, indent=2, default=str), encoding="utf-8")
        return out
