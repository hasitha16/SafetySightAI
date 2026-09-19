from pathlib import Path
import sys
from typing import Any, Dict


PROJECT_ROOT = Path(__file__).resolve().parent.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))


from tools.safety_tools import (
    search_similar_incidents,
    analyze_safety_patterns,
    get_emerging_precursors,
    get_location_risk
)


def select_safety_tool(question: str) -> str:
    """
    Select the most appropriate SafeSight tool for a
    safety officer's question.
    """

    question_lower = question.lower()

    # Emerging / changing risks
    emerging_terms = [
        "emerging",
        "increasing",
        "growing",
        "rising",
        "recent trend",
        "precursor"
    ]

    if any(
        term in question_lower
        for term in emerging_terms
    ):
        return "get_emerging_precursors"

    # Location-specific analysis
    location_terms = [
        "texas",
        "florida",
        "new york",
        "illinois",
        "ohio",
        "pennsylvania",
        "georgia",
        "wisconsin",
        "alabama",
        "missouri",
        "houston",
        "orlando",
        "chicago",
        "dallas",
        "san antonio",
        "miami",
        "atlanta",
        "denver"
    ]

    if any(
        term in question_lower
        for term in location_terms
    ):
        return "get_location_risk"

    # Dataset-wide patterns
    pattern_terms = [
        "overall",
        "common",
        "most common",
        "dataset",
        "historical trend",
        "patterns",
        "statistics"
    ]

    if any(
        term in question_lower
        for term in pattern_terms
    ):
        return "analyze_safety_patterns"

    # Default:
    # retrieve relevant historical incidents
    return "search_similar_incidents"


def extract_location(question: str) -> str | None:
    """
    Extract a supported location from a question.
    """

    locations = [
        "Texas",
        "Florida",
        "New York",
        "Illinois",
        "Ohio",
        "Pennsylvania",
        "Georgia",
        "Wisconsin",
        "Alabama",
        "Missouri",
        "Houston",
        "Orlando",
        "Chicago",
        "Dallas",
        "San Antonio",
        "Miami",
        "Atlanta",
        "Denver"
    ]

    question_lower = question.lower()

    for location in locations:
        if location.lower() in question_lower:
            return location

    return None


def run_safety_tool(question: str) -> Dict[str, Any]:
    """
    Select and execute the appropriate SafeSight tool.
    """

    selected_tool = select_safety_tool(question)

    if selected_tool == "get_emerging_precursors":

        result = get_emerging_precursors.invoke(
            {"limit": 10}
        )

    elif selected_tool == "get_location_risk":

        location = extract_location(question)

        if location is None:
            return {
                "tool": selected_tool,
                "error": "A location could not be identified."
            }

        result = get_location_risk.invoke(
            {"location": location}
        )

    elif selected_tool == "analyze_safety_patterns":

        result = analyze_safety_patterns.invoke({})

    else:

        result = search_similar_incidents.invoke(
            {
                "query": question,
                "limit": 5
            }
        )

    return {
        "tool": selected_tool,
        "result": result
    }

if __name__ == "__main__":

    questions = [
        "What safety risks are increasing recently?",
        "What are the main safety risks in Texas?",
        "Show me incidents involving machinery during cleaning."
    ]

    print("\n====================================")
    print("   SafeSight Tool Calling Test")
    print("====================================")

    for question in questions:

        print("\n------------------------------------")
        print("Question:")
        print(question)

        response = run_safety_tool(question)

        print("\nSelected Tool:")
        print(response["tool"])

        print("\nTool Result:")

        result = response.get("result")

        if isinstance(result, list):
            for item in result[:3]:
                print(item)

        elif isinstance(result, dict):
            print(result)

        else:
            print(result)