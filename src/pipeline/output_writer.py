import json
import os
from typing import List

import pandas as pd

from src.pipeline.pipeline import CaseRecord
from src.utils.logger import get_logger

logger = get_logger(__name__)


def write_structured_data(records: List[CaseRecord], output_dir: str = "output/structured_data") -> None:
    """
    Writes one JSON file per document containing its extracted structured
    case data. A record where extraction failed still gets a file — with
    `case` set to null and `error` populated — so every processed document
    has a corresponding output artifact, not just the successful ones.
    """
    os.makedirs(output_dir, exist_ok=True)

    for record in records:
        stem = os.path.splitext(record.filename)[0]
        output_path = os.path.join(output_dir, f"{stem}.json")

        payload = {
            "filename": record.filename,
            "case": record.case.model_dump() if record.case is not None else None,
            "error": record.error,
        }

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)

        logger.info(f"Wrote structured data for {record.filename} -> {output_path}")


def write_customer_emails(records: List[CaseRecord], output_dir: str = "output/customer_emails") -> None:
    """
    Writes one .txt file per document containing its generated customer
    email. A record with no email (extraction or generation failed) still
    gets a file, with a placeholder note explaining why, so the output
    folder makes it obvious which documents were skipped and why rather
    than silently having a missing file.
    """
    os.makedirs(output_dir, exist_ok=True)

    for record in records:
        stem = os.path.splitext(record.filename)[0]
        output_path = os.path.join(output_dir, f"{stem}.txt")

        content = (
            record.email
            if record.email is not None
            else f"[No email generated for {record.filename}]\nReason: {record.error}"
        )

        with open(output_path, "w", encoding="utf-8") as f:
            f.write(content)

        logger.info(f"Wrote customer email for {record.filename} -> {output_path}")


def write_case_summaries(records: List[CaseRecord], output_dir: str = "output/case_summaries") -> None:
    """
    Writes one .txt file per document containing its internal case
    summary. Same "nothing dropped silently" approach as the other
    writers: a record with no summary still gets a file explaining why.
    """
    os.makedirs(output_dir, exist_ok=True)

    for record in records:
        stem = os.path.splitext(record.filename)[0]
        output_path = os.path.join(output_dir, f"{stem}.txt")

        content = (
            record.case_summary
            if record.case_summary is not None
            else f"[No case summary generated for {record.filename}]\nReason: {record.error}"
        )

        with open(output_path, "w", encoding="utf-8") as f:
            f.write(content)

        logger.info(f"Wrote case summary for {record.filename} -> {output_path}")


def write_final_report(records: List[CaseRecord], output_path: str = "output/final_report.csv") -> None:
    """
    Builds the consolidated final report: one row per processed document,
    with key extracted fields, case status, and success/failure flags for
    both the extraction and generation stages. Deliberately excludes long
    free-text fields (issue_description, resolution_provided, the email
    body, the case summary) — those already live in their own output
    files; this report is meant to be scanned quickly, not read as a raw
    data dump.
    """
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    rows = []
    for record in records:
        case = record.case
        rows.append({
            "filename": record.filename,
            "customer_name": case.customer_name if case else None,
            "email": case.email if case else None,
            "complaint_category": case.complaint_category if case else None,
            "is_complaint": case.is_complaint if case else None,
            "escalation_required": case.escalation_required if case else None,
            "supporting_document_available": case.supporting_document_available if case else None,
            "overall_case_status": case.overall_case_status if case else None,
            "extraction_success": case is not None,
            "generation_success": record.email is not None and record.case_summary is not None,
            "error": record.error,
        })

    df = pd.DataFrame(rows)
    df.to_csv(output_path, index=False)

    logger.info(f"Wrote final report ({len(rows)} rows) -> {output_path}")
