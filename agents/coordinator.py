from pathlib import Path
import sys
from typing import TypedDict, Optional, Dict, Any

import pandas as pd

from langgraph.graph import StateGraph, START, END


# --------------------------------------------------
# Project imports
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_FILE = (
    PROJECT_ROOT
    / "data"
    / "safesight_incidents_clean.csv"
)

if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))


from agents.report_parser import (parse_safety_report, SafetyReportAnalysis)
from agents.risk_agent import assess_risk
from agents.pattern_agent import detect_recurring_patterns
from rag.vector_store import retrieve_similar_reports

# --------------------------------------------------
# 1. GRAPH STATE
# --------------------------------------------------

class SafetyState(TypedDict, total=False):
    """
    Shared state passed between SafeSight agents.
    """

    report_id: str
    report_text: str

    parsed_report: Dict[str, Any]
    risk_assessment: Dict[str, Any]
    pattern_analysis: Dict[str, Any]
    rag_evidence: list[Dict[str, Any]]

    status: str
    error: Optional[str]


# --------------------------------------------------
# 2. REPORT PARSER NODE
# --------------------------------------------------

def report_parser_node(state: SafetyState):
    """
    Extract structured safety information from
    the free-text report.
    """

    print("\n[Agent 1] Parsing safety report...")

    try:
        parsed = parse_safety_report(
            state["report_text"]
        )

        return {
            "parsed_report": parsed.model_dump(),
            "status": "REPORT_PARSED",
            "error": None
        }

    except Exception as error:

        return {
            "status": "PARSER_FAILED",
            "error": str(error)
        }


# --------------------------------------------------
# 3. ROUTING AFTER PARSER
# --------------------------------------------------

def route_after_parser(state: SafetyState):
    """
    Continue to risk scoring only when parsing
    completed successfully.
    """

    if state.get("error"):
        return "stop"

    return "risk"

def route_after_risk(state: SafetyState):
    """
    Continue only if risk scoring succeeded.
    """

    if state.get("error"):
        return "stop"

    return "patterns"


def route_after_patterns(state: SafetyState):
    """
    Continue only if pattern analysis succeeded.
    """

    if state.get("error"):
        return "stop"

    return "rag"


def route_after_rag(state: SafetyState):
    """
    Finish after successful evidence retrieval.
    """

    if state.get("error"):
        return "stop"

    return "complete"

# --------------------------------------------------
# 4. RISK SCORING NODE
# --------------------------------------------------

def risk_scoring_node(state: SafetyState):
    """
    Assess the risk level using the structured
    report produced by Agent 1.
    """

    print("\n[Agent 2] Assessing risk...")

    try:

        # Convert LangGraph dictionary state back into
        # the Pydantic model expected by Risk Agent.
        parsed_report = SafetyReportAnalysis(
            **state["parsed_report"]
        )

        assessment = assess_risk(
            parsed_report
        )

        return {
            "risk_assessment": assessment.model_dump(),
            "status": "RISK_ASSESSED",
            "error": None
        }

    except Exception as error:

        return {
            "status": "RISK_FAILED",
            "error": str(error)
        }

def pattern_analysis_node(state: SafetyState):
    """
    Analyse recurring safety patterns across
    the historical incident dataset.
    """

    print("\n[Agent 3] Analysing historical patterns...")

    try:

        # Load the full cleaned historical dataset
        df = pd.read_csv(
            DATA_FILE,
            dtype={"industry_code": str},
            low_memory=False
        )

        df["event_date"] = pd.to_datetime(
            df["event_date"],
            errors="coerce"
        )

        # Run the existing Pattern Agent
        patterns = detect_recurring_patterns(df)

        return {
            "pattern_analysis": patterns,
            "status": "PATTERNS_ANALYSED",
            "error": None
        }

    except Exception as error:

        return {
            "status": "PATTERN_FAILED",
            "error": str(error)
        }

def rag_evidence_node(state: SafetyState):
    """
    Retrieve historical incidents similar to the
    current safety report.

    Uses local embeddings and Chroma.
    No Gemini call is required.
    """

    print("\n[Agent 4] Retrieving historical evidence...")

    try:

        query = state["report_text"]

        documents = retrieve_similar_reports(
            query,
            k=5
        )

        evidence = []

        for document in documents:

            evidence.append(
                {
                    "report_id": document.metadata.get(
                        "report_id"
                    ),
                    "event_date": document.metadata.get(
                        "event_date"
                    ),
                    "city": document.metadata.get(
                        "city"
                    ),
                    "state": document.metadata.get(
                        "state"
                    ),
                    "event_type": document.metadata.get(
                        "event_type"
                    ),
                    "injury_nature": document.metadata.get(
                        "injury_nature"
                    ),
                    "content": document.page_content
                }
            )

        return {
            "rag_evidence": evidence,
            "status": "EVIDENCE_RETRIEVED",
            "error": None
        }

    except Exception as error:

        return {
            "status": "RAG_FAILED",
            "error": str(error)
        }

# --------------------------------------------------
# 5. BUILD LANGGRAPH
# --------------------------------------------------

def build_safety_graph():

    builder = StateGraph(SafetyState)

    builder.add_node(
        "report_parser",
        report_parser_node
    )

    builder.add_node(
        "risk_scoring",
        risk_scoring_node
    )

    builder.add_node(
        "pattern_analysis",
        pattern_analysis_node
    )

    builder.add_node(
        "rag_evidence",
        rag_evidence_node
    )

    builder.add_edge(
        START,
        "report_parser"
    )

    builder.add_conditional_edges(
        "report_parser",
        route_after_parser,
        {
            "risk": "risk_scoring",
            "stop": END
        }
    )

    builder.add_conditional_edges(
        "risk_scoring",
        route_after_risk,
        {
            "patterns": "pattern_analysis",
            "stop": END
        }
    )

    builder.add_conditional_edges(
        "pattern_analysis",
        route_after_patterns,
        {
            "rag": "rag_evidence",
            "stop": END
        }
    )

    builder.add_conditional_edges(
        "rag_evidence",
        route_after_rag,
        {
            "complete": END,
            "stop": END
        }
    )

    return builder.compile()


safety_graph = build_safety_graph()


# --------------------------------------------------
# 6. RUN COORDINATOR
# --------------------------------------------------

def analyze_with_coordinator(
    report_text,
    report_id="MANUAL"
):

    initial_state = {
        "report_id": str(report_id),
        "report_text": report_text,
        "status": "STARTED",
        "error": None
    }

    return safety_graph.invoke(initial_state)


# --------------------------------------------------
# TEST
# --------------------------------------------------

if __name__ == "__main__":

    print("\n====================================")
    print("   SafeSight Coordinator Ready")
    print("====================================")

    print("\nLangGraph compiled successfully.")

    print("\nWorkflow:")
    print(
        "START -> Report Parser -> Risk Scorer "
        "-> Pattern Analysis -> RAG Evidence -> END"
    )

    print("\nFailure routing:")
    print(
        "Any failed stage stops the workflow safely."
    )