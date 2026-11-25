"""End-to-end VeriReport AI pipeline orchestrator."""

from __future__ import annotations

import hashlib
import time
import uuid
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

from backend.analytics_engine import AnalyticsEngine
from backend.audit_logger import AuditLogger
from backend.claim_extractor import ClaimExtractor
from backend.claim_mapper import ClaimMapper
from backend.claim_normalizer import ClaimNormalizer
from backend.consistency_checker import ConsistencyChecker
from backend.data_loader import DataLoader
from backend.data_profiler import profile_dataset
from backend.database_manager import DatabaseManager
from backend.explanation_generator import ExplanationGenerator
from backend.llm_client import OllamaClientWrapper
from backend.metric_registry import MetricRegistry
from backend.preprocessing import preprocess_transactions
from backend.report_generator import ReportGenerator
from backend.report_templates import parse_markdown_sections, split_sentences
from backend.review_manager import ReviewManager
from backend.schemas import (
    ClaimVerificationResult,
    ColumnMappingResult,
    ConsistencyFinding,
    DatasetMetadata,
    DatasetProfile,
    GeneratedReport,
    ReportSection,
    TransactionPolicy,
)
from backend.status_engine import StatusEngine
from config.settings import Settings, get_settings


@dataclass
class PipelineRunOutput:
    run_id: str
    metadata: DatasetMetadata
    mapping_result: ColumnMappingResult
    policy: TransactionPolicy
    raw_df: pd.DataFrame
    annotated_df: pd.DataFrame
    processed_df: pd.DataFrame
    profile: DatasetProfile
    analytics: dict[str, Any]
    report: GeneratedReport
    results: list[ClaimVerificationResult]
    consistency_findings: list[ConsistencyFinding]
    extraction_warnings: list[str]
    total_time_seconds: float


class VeriReportPipeline:
    """Orchestrates dataset ingestion, analytics, report generation, verification, and audit logging."""

    def __init__(
        self,
        settings: Settings | None = None,
        db_manager: DatabaseManager | None = None,
        llm_client: OllamaClientWrapper | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.db_manager = db_manager or DatabaseManager(settings=self.settings)
        self.llm_client = llm_client or OllamaClientWrapper(settings=self.settings)
        self.registry = MetricRegistry()
        self.data_loader = DataLoader(self.settings)
        self.analytics_engine = AnalyticsEngine(self.registry)
        self.report_generator = ReportGenerator(self.llm_client, self.settings)
        self.claim_extractor = ClaimExtractor(self.llm_client, self.settings)
        self.claim_mapper = ClaimMapper(self.registry)
        self.status_engine = StatusEngine()
        self.consistency_checker = ConsistencyChecker()
        self.review_manager = ReviewManager(self.status_engine)
        self.explanation_generator = ExplanationGenerator(self.llm_client, self.settings)
        self.audit_logger = AuditLogger(self.db_manager, self.settings)

    def run_from_bytes(
        self,
        content: bytes,
        filename: str,
        policy: TransactionPolicy | None = None,
        user_mapping: dict[str, str] | None = None,
        use_llm_report: bool = True,
        use_llm_extraction: bool = True,
        use_llm_explanations: bool = False,
        custom_report_text: str | None = None,
        save_report_files: bool = True,
        persist_audit: bool = True,
        progress_callback: Any | None = None,
    ) -> PipelineRunOutput:
        start_total = time.perf_counter()
        policy = policy or TransactionPolicy()

        def _notify(msg: str) -> None:
            if progress_callback is not None:
                progress_callback(msg)

        # Phases 1-3: Load, validate, preprocess, profile, compute analytics
        _notify(f"Loading and validating `{filename}`...")
        raw_df, metadata, mapping_result = self.data_loader.load_from_bytes(
            content=content,
            filename=filename,
            user_mapping=user_mapping,
            persist_raw_copy=persist_audit,
        )
        _notify(f"Preprocessing {metadata.raw_row_count:,} transaction rows...")
        annotated_df, processed_df, transformations = preprocess_transactions(raw_df, policy)
        profile = profile_dataset(annotated_df, processed_df, transformations)
        _notify(f"Computing 14 deterministic retail analytics across {profile.total_processed_rows:,} valid rows...")
        analytics = self.analytics_engine.compute_full_analytics(
            annotated_df, processed_df, profile, policy
        )

        # Phase 4: Report generation (or wrap supplied report text for verification)
        _notify("Generating structured 11-section sales report...")
        if custom_report_text is not None:
            sections = parse_markdown_sections(custom_report_text)
            if not any(s.content for s in sections):
                sections = [
                    ReportSection(
                        section_number=1,
                        title="Executive Summary",
                        content=custom_report_text,
                        sentences=split_sentences(custom_report_text),
                    )
                ]
            rep_id = f"rep_{uuid.uuid4().hex[:10]}"
            report = GeneratedReport(
                report_id=rep_id,
                dataset_id=metadata.dataset_id,
                report_hash=hashlib.sha256(custom_report_text.encode("utf-8")).hexdigest(),
                model_used="supplied-report",
                generation_mode="external",
                generated_at=datetime.utcnow().isoformat(),
                generation_time_seconds=0.0,
                title=f"Verified Report — {metadata.filename}",
                currency_code=policy.currency_code,
                currency_symbol=policy.currency_symbol,
                sections=sections,
                full_text=custom_report_text,
            )
        else:
            report = self.report_generator.generate_report(
                metadata=metadata,
                profile=profile,
                analytics=analytics,
                use_llm=use_llm_report,
                save_files=save_report_files,
            )

        # Phases 5-11: Extract, normalize, map, verify, quality, status, consistency, explain, audit
        _notify("Extracting and mapping factual/numerical claims...")
        extracted_claims, extraction_warnings = self.claim_extractor.extract_from_report(
            report, use_llm=use_llm_extraction
        )
        normalizer = ClaimNormalizer(
            known_products=processed_df["Description"].dropna().unique() if "Description" in processed_df.columns else [],
            known_countries=processed_df["Country"].dropna().unique() if "Country" in processed_df.columns else [],
            known_periods=processed_df["PeriodMonth"].dropna().unique() if "PeriodMonth" in processed_df.columns else [],
        )
        normalized_claims = normalizer.normalize_all(extracted_claims)
        mapped_claims = self.claim_mapper.map_claims(normalized_claims)

        _notify(f"Independently verifying {len(mapped_claims)} claim(s) against transaction records...")
        results = self.status_engine.evaluate_all(
            mapped_claims, annotated_df, processed_df, profile, policy
        )
        consistency_findings = self.consistency_checker.check(results)
        results = self.explanation_generator.explain_all(results, use_llm=use_llm_explanations)

        total_elapsed = round(time.perf_counter() - start_total, 4)
        run_id = f"run_{uuid.uuid4().hex[:10]}"
        if persist_audit:
            _notify("Saving full audit trail to SQLite...")
            run_id = self.audit_logger.save_full_audit_run(
                metadata=metadata,
                profile=profile,
                policy=policy,
                report=report,
                results=results,
                consistency_findings=consistency_findings,
                processing_time_seconds=total_elapsed,
                run_id=run_id,
            )

        return PipelineRunOutput(
            run_id=run_id,
            metadata=metadata,
            mapping_result=mapping_result,
            policy=policy,
            raw_df=raw_df,
            annotated_df=annotated_df,
            processed_df=processed_df,
            profile=profile,
            analytics=analytics,
            report=report,
            results=results,
            consistency_findings=consistency_findings,
            extraction_warnings=extraction_warnings,
            total_time_seconds=total_elapsed,
        )

    def run_from_path(
        self,
        file_path: str | Path,
        **kwargs: Any,
    ) -> PipelineRunOutput:
        p = Path(file_path)
        return self.run_from_bytes(content=p.read_bytes(), filename=p.name, **kwargs)
