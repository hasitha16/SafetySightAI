import os
from pathlib import Path

from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI

from vector_store import retrieve_similar_reports


PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_ROOT / ".env")


# --------------------------------------------------
# Gemini model
# --------------------------------------------------

llm = ChatGoogleGenerativeAI(
    model="gemini-3.6-flash",
    google_api_key=os.getenv("GEMINI_API_KEY")
)


# --------------------------------------------------
# Build grounded context
# --------------------------------------------------

def build_retrieval_context(documents):
    """
    Convert retrieved historical reports into a context
    block for the Safety Advisor.
    """

    context_blocks = []

    for number, document in enumerate(documents, start=1):

        metadata = document.metadata

        block = f"""
SOURCE {number}
Report ID: {metadata.get("report_id", "Unknown")}
Date: {metadata.get("event_date", "Unknown")}
Location: {metadata.get("city", "Unknown")}, {metadata.get("state", "Unknown")}
Incident Type: {metadata.get("event_type", "Unknown")}
Injury Nature: {metadata.get("injury_nature", "Unknown")}
Incident Source: {metadata.get("incident_source", "Unknown")}

Report Content:
{document.page_content}
""".strip()

        context_blocks.append(block)

    return "\n\n".join(context_blocks)


# --------------------------------------------------
# Retrieve evidence
# --------------------------------------------------

def retrieve_evidence(question, k=5):
    """
    Retrieve historical safety reports relevant to
    the safety officer's question.
    """

    documents = retrieve_similar_reports(
        question,
        k=k
    )

    context = build_retrieval_context(documents)

    return documents, context


# --------------------------------------------------
# Grounded RAG Safety Advisor
# --------------------------------------------------

def ask_safety_advisor(question, k=5):
    """
    Answer a safety question using retrieved historical
    reports as evidence.
    """

    documents, context = retrieve_evidence(
        question,
        k=k
    )

    prompt = f"""
You are the SafeSight AI Safety Advisor.

Your role is to help a safety officer understand patterns
in historical workplace safety reports.

USER QUESTION:
{question}

RETRIEVED HISTORICAL EVIDENCE:
{context}

INSTRUCTIONS:

1. Answer using only the retrieved evidence above.
2. Do not invent incidents, statistics, causes, or report IDs.
3. If the retrieved evidence is insufficient, clearly say so.
4. Identify recurring hazards or patterns when supported.
5. Distinguish observations from recommendations.
6. Recommendations must be practical and based on the
   retrieved safety evidence.
7. Cite supporting evidence using the exact report ID in
   square brackets, for example:
   [Report 2020010054]
8. Do not claim that the retrieved reports represent the
   entire dataset.
9. Keep the response concise and useful to a safety officer.

Structure the answer using:

FINDINGS
Briefly answer the question and describe supported patterns.

EVIDENCE
Mention the relevant historical reports and what they show.

RECOMMENDED ACTIONS
Provide practical actions justified by the evidence.
"""

    response = llm.invoke(prompt)

    # Gemini/LangChain may return text directly or content blocks
    if isinstance(response.content, str):
        answer = response.content
    else:
        text_parts = []

        for block in response.content:
            if isinstance(block, dict) and block.get("type") == "text":
                text_parts.append(block.get("text", ""))

        answer = "\n".join(text_parts)

    return {
        "question": question,
        "answer": answer,
        "sources": [
            {
                "report_id": doc.metadata.get("report_id"),
                "event_date": doc.metadata.get("event_date"),
                "city": doc.metadata.get("city"),
                "state": doc.metadata.get("state")
            }
            for doc in documents
        ]
    }


# --------------------------------------------------
# Retrieval-only test
# --------------------------------------------------

if __name__ == "__main__":

    question = (
        "What recurring safety risks appear during "
        "machinery maintenance and cleaning?"
    )

    print("\n====================================")
    print("    SafeSight AI Safety Advisor")
    print("====================================")

    print(f"\nQuestion:\n{question}")

    documents, context = retrieve_evidence(
        question,
        k=5
    )

    print("\n--- Retrieved Evidence ---")
    print(context)

    print("\n--- Retrieved Report IDs ---")

    for document in documents:
        print(
            document.metadata.get(
                "report_id",
                "Unknown"
            )
        )

    result = ask_safety_advisor(
        question,
        k=5
    )

    print("\n--- Advisor Answer ---")
    print(result["answer"])