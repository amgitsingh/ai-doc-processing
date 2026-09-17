from dataclasses import dataclass
from typing import List, Optional

from langchain_core.runnables import RunnableParallel

from src.loaders.document_loader import LoadedDocument
from src.components.extraction import ComplaintCase, case_to_prompt_dict
from src.utils.logger import get_logger

DEFAULT_MAX_CONCURRENCY = 5

logger = get_logger(__name__)


@dataclass
class CaseRecord:
    """
    Carries one document through the pipeline. A document that fails at
    any stage keeps its record (with `error` set) instead of being dropped
    silently, so the final report can still account for every input file.
    """
    filename: str
    filepath: str
    text: str
    case: Optional[ComplaintCase] = None
    email: Optional[str] = None
    case_summary: Optional[str] = None
    error: Optional[str] = None


def run_extraction(
    docs: List[LoadedDocument],
    extraction_chain,
    max_concurrency: int = DEFAULT_MAX_CONCURRENCY,
) -> List[CaseRecord]:
    """
    Runs structured extraction over every loaded document. Documents are
    batched (chain.batch, not a sequential loop) so their LLM calls run
    concurrently rather than one-at-a-time — real batch-level parallelism,
    not just parallelism within a single document. return_exceptions=True
    keeps the same per-document error isolation a try/except loop would
    give: one failing document is recorded with `error` set and does not
    abort the rest of the batch.
    """
    inputs = [{"document_text": doc.text} for doc in docs]
    results = extraction_chain.batch(
        inputs,
        config={"max_concurrency": max_concurrency},
        return_exceptions=True,
    )

    records: List[CaseRecord] = []
    for doc, result in zip(docs, results):
        record = CaseRecord(filename=doc.filename, filepath=doc.filepath, text=doc.text)
        if isinstance(result, Exception):
            record.error = f"Extraction failed: {result}"
            logger.error(f"Extraction failed for {doc.filename}: {result}")
        else:
            record.case = result
            logger.info(f"Extracted case data for {doc.filename}")
        records.append(record)

    return records


def apply_generation(
    records: List[CaseRecord],
    email_chain,
    summary_chain,
    max_concurrency: int = DEFAULT_MAX_CONCURRENCY,
) -> List[CaseRecord]:
    """
    For every record with a successfully extracted case, generates the
    customer email and internal case summary. Two layers of concurrency:
    within one document, email + summary run together via RunnableParallel
    (they only depend on the extracted case, not on each other); across
    documents, every record's combined (email + summary) call is itself
    batched via chain.batch rather than looped sequentially. Records where
    extraction already failed (record.case is None) are skipped — there is
    nothing to generate from.
    """
    generation_chain = RunnableParallel(email=email_chain, case_summary=summary_chain)

    pending = [r for r in records if r.case is not None]
    if not pending:
        return records

    inputs = []
    for r in pending:
        assert r.case is not None  # guaranteed by the `pending` filter above
        inputs.append(case_to_prompt_dict(r.case))
    results = generation_chain.batch(
        inputs,
        config={"max_concurrency": max_concurrency},
        return_exceptions=True,
    )

    for record, result in zip(pending, results):
        if isinstance(result, Exception):
            record.error = f"Generation failed: {result}"
            logger.error(f"Generation failed for {record.filename}: {result}")
        else:
            record.email = result["email"]
            record.case_summary = result["case_summary"]
            logger.info(f"Generated email + summary for {record.filename}")

    return records
