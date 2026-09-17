from typing import Optional

from langchain_core.prompts import PromptTemplate
from pydantic import BaseModel, Field


class ComplaintCase(BaseModel):
    """Structured schema for information extracted from a customer complaint document."""

    customer_name: Optional[str] = Field(
        None, description="Full name of the customer, if mentioned in the document."
    )
    email: Optional[str] = Field(
        None, description="Customer's email address, if mentioned in the document."
    )
    phone_number: Optional[str] = Field(
        None, description="Customer's phone number, if mentioned in the document."
    )
    complaint_category: str = Field(
        ...,
        description="Category of the complaint. Must be one of the provided category labels.",
    )
    issue_description: str = Field(
        ..., description="A concise description of the customer's issue, based only on the document."
    )
    resolution_provided: Optional[str] = Field(
        None,
        description="The resolution or action taken, if described in the document. Null if no resolution is mentioned.",
    )
    is_complaint: bool = Field(
        ...,
        description="True if this document represents an actual complaint. False if it is a general inquiry or unrelated content.",
    )
    escalation_required: bool = Field(
        ..., description="True if the document indicates the case was, or should be, escalated to a higher support tier."
    )
    supporting_document_available: bool = Field(
        ...,
        description="True if the document mentions supporting evidence being provided (screenshots, invoices, tracking numbers, videos, etc.).",
    )
    overall_case_status: str = Field(
        ...,
        description="Current status of the case as described in the document, e.g. 'Resolved', 'Pending', 'Escalated - awaiting repair'.",
    )


def get_extraction_chain(llm, config):
    labels = config["classification"]["labels"]

    prompt = PromptTemplate(
        template="""
You are an assistant that extracts structured case information from customer
complaint documents for an internal case-processing system.

Classify the complaint into exactly one of the following categories:
{labels}

Read the document below and extract the required fields. Use ONLY information
that is actually present in the document. If a piece of information (such as
the customer's email, phone number, or resolution) is not present, leave that
field blank/null rather than guessing or inventing a value.

Document:
{document_text}
""",
        input_variables=["document_text"],
        partial_variables={"labels": labels},
    )

    return prompt | llm.with_structured_output(ComplaintCase)


def case_to_prompt_dict(case: ComplaintCase) -> dict:
    """
    Converts a ComplaintCase into a prompt-safe dict for downstream chains
    (email_generator, case_summary). Without this, a missing Optional field
    would interpolate into a prompt as the literal text "None", and a bool
    would interpolate as "True"/"False" instead of readable Yes/No.
    """
    data = case.model_dump()

    for key in ("customer_name", "email", "phone_number", "resolution_provided"):
        if data[key] is None:
            data[key] = "Not provided"

    for key in ("is_complaint", "escalation_required", "supporting_document_available"):
        data[key] = "Yes" if data[key] else "No"

    return data
