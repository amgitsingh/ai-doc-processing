from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser


def get_case_summary_chain(llm):
    """
    Builds a chain that produces a concise internal case summary for the
    support/management team, grounded in the extracted case data.
    """
    prompt = PromptTemplate(
        template="""
You are a support operations assistant preparing an internal case summary
for the management team. This is an internal document, not customer-facing.

Use ONLY the case details below to write the summary. Base the recommended
next action on these facts — do not invent information not provided here.

Case Details:
- Customer Name: {customer_name}
- Complaint Category: {complaint_category}
- Issue: {issue_description}
- Resolution / Action Taken So Far: {resolution_provided}
- Case Status: {overall_case_status}
- Escalated: {escalation_required}
- Supporting Documentation Provided: {supporting_document_available}

Write a concise internal case summary with exactly these five sections,
each on its own line prefixed with its label:

Case Overview: <one or two sentence summary of the case>
Key Issue: <the core problem, stated plainly>
Action Taken: <what has been done so far, or "None yet" if nothing has been done>
Current Status: <the case status>
Recommended Next Action: <a specific, practical next step for the support team, based on the status and escalation flag above>
""",
        input_variables=[
            "customer_name",
            "complaint_category",
            "issue_description",
            "resolution_provided",
            "overall_case_status",
            "escalation_required",
            "supporting_document_available",
        ],
    )

    return prompt | llm | StrOutputParser()
