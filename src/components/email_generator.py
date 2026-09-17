from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser


def get_email_chain(llm):
    """
    Builds a chain that drafts a professional customer response email.
    """
    prompt = PromptTemplate(
        template="""
You are a customer support assistant drafting a professional response email
for a customer complaint case.

Use ONLY the case details below. Do not invent facts, names, dates, or any
detail not provided here. If the customer's name is "Not provided", address
the email to "Valued Customer" instead of guessing a name.

Case Details:
- Customer Name: {customer_name}
- Complaint Category: {complaint_category}
- Issue: {issue_description}
- Resolution / Action Taken: {resolution_provided}
- Case Status: {overall_case_status}

Write a short, professional email that:
1. Addresses the customer appropriately.
2. Acknowledges and summarizes their issue.
3. Clearly states the resolution or current status.
4. Maintains a polite, professional tone throughout.
5. Does not mention internal-only details (such as escalation routing) that were not given above.

Respond with only the email body text (no subject line, no explanation).
""",
        input_variables=[
            "customer_name",
            "complaint_category",
            "issue_description",
            "resolution_provided",
            "overall_case_status",
        ],
    )

    return prompt | llm | StrOutputParser()
