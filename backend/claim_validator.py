"""Pydantic validation and malformed-JSON recovery for extracted claims."""

from __future__ import annotations

import json
import re
from typing import Any

from pydantic import ValidationError

from backend.schemas import ExtractedClaim, ExtractedSubclaim


def _coerce_numeric_value(val: Any) -> float | str | None:
    if val is None or isinstance(val, bool):
        return None
    if isinstance(val, (int, float)):
        return float(val)
    text = str(val).strip()
    if not text:
        return None
    # Handle currency / comma / percent strings like "£14,800.00" or "20%"
    cleaned = re.sub(r"^[£$€₹]\s*", "", text)
    cleaned = cleaned.replace(",", "")
    if cleaned.endswith("%"):
        cleaned = cleaned[:-1].strip()
    try:
        return float(cleaned)
    except ValueError:
        return text


def extract_json_payload(raw_text: str) -> Any:
    """Safely parse JSON from LLM output, recovering from markdown fences or surrounding prose."""
    if not raw_text or not raw_text.strip():
        return []
    text = raw_text.strip()
    # Strip markdown code fences if present
    fence_match = re.search(r"```(?:json)?\s*([\s\S]*?)```", text, re.IGNORECASE)
    if fence_match:
        text = fence_match.group(1).strip()

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Try locating outermost JSON array or object
    for open_ch, close_ch in (("[", "]"), ("{", "}")):
        start = text.find(open_ch)
        end = text.rfind(close_ch)
        if start != -1 and end > start:
            snippet = text[start : end + 1]
            # Remove trailing commas before closing brackets
            snippet = re.sub(r",\s*([}\]])", r"\1", snippet)
            try:
                return json.loads(snippet)
            except json.JSONDecodeError:
                continue
    return None


def validate_extracted_claims(
    raw_payload: Any,
    default_section: str = "General",
    default_sentence: str = "",
) -> tuple[list[ExtractedClaim], list[str]]:
    """Validate and sanitize raw claim dicts into `ExtractedClaim` models.

    Returns:
        valid_claims: List of validated ExtractedClaim instances.
        errors: Non-fatal validation warnings/errors encountered.
    """
    errors: list[str] = []
    if isinstance(raw_payload, str):
        parsed = extract_json_payload(raw_payload)
        if parsed is None:
            return [], ["Malformed JSON payload from claim extractor."]
        raw_payload = parsed

    if isinstance(raw_payload, dict):
        items = raw_payload.get("claims", [raw_payload])
    elif isinstance(raw_payload, list):
        items = raw_payload
    else:
        return [], ["Unsupported claim payload type."]

    valid_claims: list[ExtractedClaim] = []
    for idx, item in enumerate(items, start=1):
        if not isinstance(item, dict):
            errors.append(f"Item {idx} is not a dict.")
            continue

        claim_text = str(item.get("claim_text") or "").strip()
        if not claim_text:
            errors.append(f"Item {idx} missing claim_text.")
            continue

        claim_id = str(item.get("claim_id") or f"CLM-{idx:03d}").strip()
        source_sentence = str(item.get("source_sentence") or default_sentence or claim_text).strip()
        report_section = str(item.get("report_section") or default_section).strip()

        subclaims_raw = item.get("subclaims") or []
        validated_subs: list[ExtractedSubclaim] = []
        if isinstance(subclaims_raw, list):
            for s_idx, sub in enumerate(subclaims_raw, start=1):
                if not isinstance(sub, dict) or not str(sub.get("claim_text") or "").strip():
                    continue
                try:
                    validated_subs.append(
                        ExtractedSubclaim(
                            subclaim_id=str(sub.get("subclaim_id") or f"{claim_id}.{s_idx}"),
                            claim_text=str(sub["claim_text"]).strip(),
                            entity=sub.get("entity"),
                            metric=sub.get("metric"),
                            period=sub.get("period"),
                            comparison_period=sub.get("comparison_period"),
                            operation=str(sub.get("operation") or "value"),
                            claimed_value=_coerce_numeric_value(sub.get("claimed_value")),
                            unit=sub.get("unit"),
                            filters=sub.get("filters") if isinstance(sub.get("filters"), dict) else {},
                            extraction_confidence=float(sub.get("extraction_confidence", 0.8)),
                            ambiguity_flag=bool(sub.get("ambiguity_flag", False)),
                        )
                    )
                except (ValidationError, TypeError, ValueError) as exc:
                    errors.append(f"Subclaim validation error in {claim_id}: {exc}")

        try:
            confidence = float(item.get("extraction_confidence", 0.85))
            confidence = max(0.0, min(1.0, confidence))
            claimed_obj = ExtractedClaim(
                claim_id=claim_id,
                claim_text=claim_text,
                source_sentence=source_sentence,
                report_section=report_section,
                entity=item.get("entity"),
                metric=item.get("metric"),
                period=item.get("period"),
                comparison_period=item.get("comparison_period"),
                operation=str(item.get("operation") or "value"),
                claimed_value=_coerce_numeric_value(item.get("claimed_value")),
                unit=item.get("unit"),
                filters=item.get("filters") if isinstance(item.get("filters"), dict) else {},
                extraction_confidence=confidence,
                ambiguity_flag=bool(item.get("ambiguity_flag", False)),
                subclaims=validated_subs,
            )
            valid_claims.append(claimed_obj)
        except (ValidationError, TypeError, ValueError) as exc:
            errors.append(f"Claim validation error at item {idx}: {exc}")

    return valid_claims, errors
