"""
Python dataclasses mirroring the Blue Roads AI database schema (v2, April 2026).
Used by the parser pipeline to produce schema-aligned output before DB writes are wired in.
All UUID and timestamp fields are None at parse time — filled in by the platform on insert.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional


# ---------------------------------------------------------------------------
# Jobs & document storage
# ---------------------------------------------------------------------------

@dataclass
class Job:
    id: Optional[str] = None
    project_id: Optional[str] = None
    organization_id: Optional[str] = None
    uploaded_by_user_id: Optional[str] = None
    job_name: Optional[str] = None
    status: str = "parsed"          # uploaded | parsing | parsed | validated | error
    parse_error: Optional[str] = None
    created_at: Optional[str] = None
    updated_at: Optional[str] = None


@dataclass
class BidParseResults:
    id: Optional[str] = None
    job_id: Optional[str] = None
    bid_number: Optional[str] = None
    project_name: Optional[str] = None
    city: Optional[str] = None
    state: Optional[str] = None
    work_types: Optional[list[str]] = None
    estimated_cost: Optional[float] = None
    bid_due_date: Optional[str] = None
    total_pages: Optional[int] = None
    selected_pages: Optional[int] = None
    selected_page_numbers: Optional[list[int]] = None
    total_streets: Optional[int] = None
    chunks_processed: Optional[int] = None
    created_at: Optional[str] = None


# ---------------------------------------------------------------------------
# Parser pipeline stage logs
# ---------------------------------------------------------------------------

@dataclass
class ParserStageLog:
    id: Optional[str] = None
    job_id: Optional[str] = None
    stage: str = ""                 # page_screen | docai_extract | gemini_extract | dedup | tag_extraction | geocoding
    stage_order: int = 0            # 1–6
    status: str = "success"         # success | error | skipped
    street_count_in: Optional[int] = None
    street_count_out: Optional[int] = None
    streets_dropped: Optional[int] = None
    pages_processed: Optional[int] = None
    pages_selected: Optional[int] = None
    selected_page_numbers: Optional[list[int]] = None
    duration_ms: Optional[int] = None
    error_message: Optional[str] = None
    raw_log_s3_key: Optional[str] = None
    metadata: Optional[dict] = None
    created_at: Optional[str] = None


# ---------------------------------------------------------------------------
# Streets
# ---------------------------------------------------------------------------

@dataclass
class StreetRaw:
    id: Optional[str] = None
    job_id: Optional[str] = None
    main_street: str = ""
    from_street: Optional[str] = None
    to_street: Optional[str] = None
    work_types: Optional[list[str]] = None  # array — dedup merges per-street materials
    page: Optional[int] = None
    source: str = "gemini-pro"          # gemini-pro | haiku | docai | text | manual
    tags: Optional[list[str]] = None
    location: Optional[str] = None
    confidence: str = "high"            # high | low
    low_confidence_reason: Optional[str] = None
    is_active: bool = True
    validated: bool = False
    created_at: Optional[str] = None


@dataclass
class StreetFinal:
    id: Optional[str] = None
    job_id: Optional[str] = None
    street_raw_id: Optional[str] = None
    main_street: str = ""
    from_street: Optional[str] = None
    to_street: Optional[str] = None
    work_type: Optional[str] = None
    tags: Optional[list[str]] = None
    source: str = "gemini-pro"
    validated_by_user_id: Optional[str] = None
    validated_at: Optional[str] = None
    created_at: Optional[str] = None
    updated_at: Optional[str] = None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def street_raw_from_dict(s: dict, job_id: str = None) -> StreetRaw:
    """Convert a raw pipeline street dict to a StreetRaw dataclass."""
    work_types = s.get("work_types") or []
    if not work_types and s.get("work_type"):
        work_types = [s["work_type"]]

    return StreetRaw(
        job_id=job_id,
        main_street=s.get("main_street") or "",
        from_street=s.get("from_street") or None,
        to_street=s.get("to_street") or None,
        work_types=work_types or None,
        page=s.get("page") or None,
        source=s.get("source") or "gemini-pro",
        tags=s.get("tags") or None,
        location=s.get("location") or None,
        confidence=s.get("confidence") or "high",
        low_confidence_reason=s.get("low_confidence_reason") or None,
        is_active=True,
        validated=False,
    )


def parser_stage_log_from_dict(d: dict, job_id: str = None) -> ParserStageLog:
    """Convert a pipeline _stage.finish() dict to a ParserStageLog dataclass."""
    return ParserStageLog(
        job_id=job_id,
        stage=d.get("stage") or "",
        stage_order=d.get("stage_order") or 0,
        status=d.get("status") or "success",
        street_count_in=d.get("street_count_in"),
        street_count_out=d.get("street_count_out"),
        streets_dropped=d.get("streets_dropped"),
        pages_processed=d.get("pages_processed"),
        pages_selected=d.get("pages_selected"),
        selected_page_numbers=d.get("selected_page_numbers"),
        duration_ms=d.get("duration_ms"),
        error_message=d.get("error_message"),
        raw_log_s3_key=d.get("raw_log_s3_key"),
        metadata=d.get("metadata"),
    )
