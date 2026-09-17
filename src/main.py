import os
import sys

# Running this file directly (python src/main.py) puts src/'s own directory
# on sys.path, not the project root — so `from src...` imports can't resolve
# without this. Same fix already used in 12-Capstone-Project2/app/streamlit_app.py.
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.utils.config_loader import load_config
from src.utils.llm_loader import load_llm
from src.utils.logger import get_logger
from src.loaders.document_loader import load_documents
from src.components.extraction import get_extraction_chain
from src.components.email_generator import get_email_chain
from src.components.case_summary import get_case_summary_chain
from src.pipeline.pipeline import run_extraction, apply_generation
from src.pipeline.output_writer import (
    write_structured_data,
    write_customer_emails,
    write_case_summaries,
    write_final_report,
)

logger = get_logger(__name__)

DATA_DIR = "data"


def main():
    logger.info("Starting AI Document Processing batch run")

    config = load_config()
    llm = load_llm(config)

    extraction_chain = get_extraction_chain(llm, config)
    email_chain = get_email_chain(llm)
    summary_chain = get_case_summary_chain(llm)

    docs = load_documents(DATA_DIR)
    if not docs:
        logger.error(f"No documents could be loaded from '{DATA_DIR}'. Nothing to process.")
        return

    records = run_extraction(docs, extraction_chain)
    records = apply_generation(records, email_chain, summary_chain)

    write_structured_data(records)
    write_customer_emails(records)
    write_case_summaries(records)
    write_final_report(records)

    succeeded = sum(1 for r in records if r.error is None)
    failed = len(records) - succeeded
    logger.info(
        f"Batch run complete: {succeeded} succeeded, {failed} failed, "
        f"{len(records)} documents processed (see output/final_report.csv for details)"
    )


if __name__ == "__main__":
    main()
