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


from agents.report_parser import (
    parse_safety_report,
    SafetyReportAnalysis
)

from agents.risk_agent import assess_risk

from agents.pattern_agent import (
    detect_recurring_patterns
)

from rag.vector_store import (
    retrieve_similar_reports
)

from tools.tool_router import (
    run_safety_tool
)


# --------------------------------------------------
# 1. GRAPH STATE
# --------------------------------------------------

class SafetyState(TypedDict, total=False):
    """
    Shared state passed between SafeSight agents.
    """

    request_type: str

    report_id: str
    report_text: str

    officer_question: str
    tool_response: Dict[str, Any]

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
# 3. RISK SCORING NODE
# --------------------------------------------------

def risk_scoring_node(state: SafetyState):
    """
    Assess the risk level using the structured
    report produced by Agent 1.
    """

    print("\n[Agent 2] Assessing risk...")

    try:

        # Convert the dictionary stored in LangGraph
        # back into the Pydantic model expected
        # by the risk-scoring agent.

        parsed_report = SafetyReportAnalysis(
            **state["parsed_report"]
        )

        assessment = assess_risk(
            parsed_report
        )

        return {
            "risk_assessment":
                assessment.model_dump(),

            "status":
                "RISK_ASSESSED",

            "error":
                None
        }

    except Exception as error:

        return {
            "status": "RISK_FAILED",
            "error": str(error)
        }


# --------------------------------------------------
# 4. PATTERN ANALYSIS NODE
# --------------------------------------------------

def pattern_analysis_node(state: SafetyState):
    """
    Analyse recurring safety patterns across
    the complete historical incident dataset.
    """

    print(
        "\n[Agent 3] Analysing historical patterns..."
    )

    try:

        # Load the full cleaned dataset.

        df = pd.read_csv(
            DATA_FILE,
            dtype={"industry_code": str},
            low_memory=False
        )

        df["event_date"] = pd.to_datetime(
            df["event_date"],
            errors="coerce"
        )

        # Run the existing Pattern Agent.

        patterns = detect_recurring_patterns(
            df
        )

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


# --------------------------------------------------
# 5. RAG EVIDENCE NODE
# --------------------------------------------------

def rag_evidence_node(state: SafetyState):
    """
    Retrieve historical incidents similar to
    the current safety report.

    This uses local embeddings and Chroma.
    No Gemini call is required here.
    """

    print(
        "\n[Agent 4] Retrieving historical evidence..."
    )

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
                    "report_id":
                        document.metadata.get(
                            "report_id"
                        ),

                    "event_date":
                        document.metadata.get(
                            "event_date"
                        ),

                    "city":
                        document.metadata.get(
                            "city"
                        ),

                    "state":
                        document.metadata.get(
                            "state"
                        ),

                    "event_type":
                        document.metadata.get(
                            "event_type"
                        ),

                    "injury_nature":
                        document.metadata.get(
                            "injury_nature"
                        ),

                    "content":
                        document.page_content
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
# 6. SAFETY TOOL NODE
# --------------------------------------------------

def safety_tool_node(state: SafetyState):
    """
    Select and execute the appropriate SafeSight
    tool for a safety officer's natural-language
    question.
    """

    print(
        "\n[Tool Agent] Selecting safety tool..."
    )

    try:

        response = run_safety_tool(
            state["officer_question"]
        )

        return {
            "tool_response": response,
            "status": "TOOL_EXECUTED",
            "error": None
        }

    except Exception as error:

        return {
            "status": "TOOL_FAILED",
            "error": str(error)
        }


# --------------------------------------------------
# 7. REQUEST ROUTING
# --------------------------------------------------

def route_request(state: SafetyState):
    """
    Route a request to either:

    1. Incident analysis workflow
    2. Safety officer tool workflow
    """

    request_type = state.get(
        "request_type",
        ""
    ).lower()

    # Explicit request type.

    if request_type == "question":
        return "tool"

    if request_type == "incident":
        return "incident"

    # Automatically infer request type
    # when one was not explicitly provided.

    if state.get("officer_question"):
        return "tool"

    if state.get("report_text"):
        return "incident"

    return "stop"


# --------------------------------------------------
# 8. INCIDENT WORKFLOW ROUTING
# --------------------------------------------------

def route_after_parser(state: SafetyState):
    """
    Continue to risk scoring only when
    report parsing succeeds.
    """

    if state.get("error"):
        return "stop"

    return "risk"


def route_after_risk(state: SafetyState):
    """
    Continue to historical pattern analysis
    only when risk scoring succeeds.
    """

    if state.get("error"):
        return "stop"

    return "patterns"


def route_after_patterns(state: SafetyState):
    """
    Continue to RAG evidence retrieval only
    when pattern analysis succeeds.
    """

    if state.get("error"):
        return "stop"

    return "rag"


def route_after_rag(state: SafetyState):
    """
    Complete the incident workflow after
    successful evidence retrieval.
    """

    if state.get("error"):
        return "stop"

    return "complete"


# --------------------------------------------------
# 9. BUILD LANGGRAPH
# --------------------------------------------------

def build_safety_graph():
    """
    Build the SafeSight multi-path LangGraph
    coordinator.
    """

    builder = StateGraph(
        SafetyState
    )


    # -------------------------------
    # Register nodes
    # -------------------------------

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

    builder.add_node(
        "safety_tool",
        safety_tool_node
    )


    # -------------------------------
    # START routing
    # -------------------------------

    builder.add_conditional_edges(
        START,
        route_request,
        {
            "incident":
                "report_parser",

            "tool":
                "safety_tool",

            "stop":
                END
        }
    )


    # -------------------------------
    # Report Parser routing
    # -------------------------------

    builder.add_conditional_edges(
        "report_parser",
        route_after_parser,
        {
            "risk":
                "risk_scoring",

            "stop":
                END
        }
    )


    # -------------------------------
    # Risk Agent routing
    # -------------------------------

    builder.add_conditional_edges(
        "risk_scoring",
        route_after_risk,
        {
            "patterns":
                "pattern_analysis",

            "stop":
                END
        }
    )


    # -------------------------------
    # Pattern Agent routing
    # -------------------------------

    builder.add_conditional_edges(
        "pattern_analysis",
        route_after_patterns,
        {
            "rag":
                "rag_evidence",

            "stop":
                END
        }
    )


    # -------------------------------
    # RAG routing
    # -------------------------------

    builder.add_conditional_edges(
        "rag_evidence",
        route_after_rag,
        {
            "complete":
                END,

            "stop":
                END
        }
    )


    # -------------------------------
    # Tool workflow
    # -------------------------------

    builder.add_edge(
        "safety_tool",
        END
    )


    # Compile graph.

    return builder.compile()


# --------------------------------------------------
# 10. COMPILE GRAPH
# --------------------------------------------------

safety_graph = build_safety_graph()


# --------------------------------------------------
# 11. INCIDENT ANALYSIS ENTRY POINT
# --------------------------------------------------

def analyze_with_coordinator(
    report_text,
    report_id="MANUAL"
):
    """
    Run a free-text safety report through
    the complete incident-analysis workflow.
    """

    initial_state = {
        "request_type": "incident",
        "report_id": str(report_id),
        "report_text": report_text,
        "status": "STARTED",
        "error": None
    }

    return safety_graph.invoke(
        initial_state
    )


# --------------------------------------------------
# 12. SAFETY OFFICER QUESTION ENTRY POINT
# --------------------------------------------------

def ask_with_coordinator(
    question
):
    """
    Route a safety officer's question through
    the SafeSight tool workflow.
    """

    initial_state = {
        "request_type": "question",
        "officer_question": question,
        "status": "STARTED",
        "error": None
    }

    return safety_graph.invoke(
        initial_state
    )


# --------------------------------------------------
# 13. TEST
# --------------------------------------------------

if __name__ == "__main__":

    print(
        "\n===================================="
    )

    print(
        "   SafeSight LangGraph Router Test"
    )

    print(
        "===================================="
    )


    # Test only the tool workflow.
    #
    # This does NOT call Gemini.

    test_state = {
        "request_type":
            "question",

        "officer_question":
            "What are the main safety risks in Texas?"
    }


    result = safety_graph.invoke(
        test_state
    )


    print("\nFinal Status:")

    print(
        result.get("status")
    )


    print("\nSelected Tool:")

    print(
        result.get(
            "tool_response",
            {}
        ).get(
            "tool"
        )
    )


    print("\nTool Result:")

    print(
        result.get(
            "tool_response",
            {}
        ).get(
            "result"
        )
    )


    print("\nError:")

    print(
        result.get("error")
    )