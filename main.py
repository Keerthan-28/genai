"""CLI entry point for running VeriReport AI end-to-end or launching the dashboard."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

from backend.pipeline import VeriReportPipeline
from config.settings import PROJECT_ROOT


def main() -> int:
    parser = argparse.ArgumentParser(description="VeriReport AI — Evidence-Based Retail Sales Report Verifier")
    parser.add_argument("--file", type=str, help="Path to CSV or Excel retail dataset")
    parser.add_argument("--sample", action="store_true", help="Run on bundled sample retail dataset")
    parser.add_argument("--no-llm", action="store_true", help="Use deterministic fallback instead of Ollama")
    parser.add_argument("--ui", action="store_true", help="Launch the Streamlit dashboard")
    args = parser.parse_args()

    if args.ui or (not args.file and not args.sample):
        app_path = PROJECT_ROOT / "frontend" / "app.py"
        return subprocess.call([sys.executable, "-m", "streamlit", "run", str(app_path)])

    target = Path(args.file) if args.file else PROJECT_ROOT / "data" / "samples" / "sample_retail_transactions.csv"
    pipeline = VeriReportPipeline()
    use_llm = not args.no_llm
    out = pipeline.run_from_path(
        target,
        use_llm_report=use_llm,
        use_llm_extraction=use_llm,
        use_llm_explanations=False,
    )
    summary = {
        "run_id": out.run_id,
        "dataset": out.metadata.filename,
        "report_mode": out.report.generation_mode,
        "total_claims": len(out.results),
        "statuses": {
            s: sum(1 for r in out.results if r.status.value == s)
            for s in ("VERIFIED", "CONTRADICTED", "INSUFFICIENT_EVIDENCE", "NEEDS_REVIEW", "PARTIALLY_VERIFIED")
        },
        "consistency_findings": len(out.consistency_findings),
        "elapsed_seconds": out.total_time_seconds,
    }
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
