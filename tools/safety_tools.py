from pathlib import Path
import sys

import pandas as pd
from langchain_core.tools import tool


# --------------------------------------------------
# Project setup
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))


from agents.pattern_agent import (
    detect_recurring_patterns,
    detect_emerging_incident_types
)

from rag.vector_store import retrieve_similar_reports


DATA_FILE = (
    PROJECT_ROOT
    / "data"
    / "safesight_incidents_clean.csv"
)


# --------------------------------------------------
# Shared dataset loader
# --------------------------------------------------

def load_safety_dataset():
    """
    Load the cleaned SafeSight historical incident dataset.
    """

    df = pd.read_csv(
        DATA_FILE,
        dtype={"industry_code": str},
        low_memory=False
    )

    df["event_date"] = pd.to_datetime(
        df["event_date"],
        errors="coerce"
    )

    return df


# --------------------------------------------------
# TOOL 1 — Historical Incident Search
# --------------------------------------------------

@tool
def search_similar_incidents(
    query: str,
    limit: int = 5
) -> list:
    """
    Search historical workplace safety reports for incidents
    semantically similar to a safety question or report.

    Use this tool when evidence from previous incidents is
    needed to support a safety assessment or recommendation.
    """

    limit = max(1, min(limit, 10))

    documents = retrieve_similar_reports(
        query,
        k=limit
    )

    results = []

    for document in documents:

        results.append(
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

    return results


# --------------------------------------------------
# TOOL 2 — Dataset-wide Pattern Analysis
# --------------------------------------------------

@tool
def analyze_safety_patterns() -> dict:
    """
    Analyse the complete historical safety dataset and return
    recurring incident types, locations, injury patterns,
    serious outcomes, yearly trends and emerging incidents.

    Use this tool for dataset-wide safety trend analysis rather
    than for one individual report.
    """

    df = load_safety_dataset()

    return detect_recurring_patterns(df)


# --------------------------------------------------
# TOOL 3 — Emerging Precursor Detection
# --------------------------------------------------

@tool
def get_emerging_precursors(
    limit: int = 10
) -> list:
    """
    Identify incident types that are increasing in the most
    recent period compared with the preceding period.

    Use this tool to detect emerging safety precursors and
    changing risk patterns.
    """

    df = load_safety_dataset()

    results = detect_emerging_incident_types(df)

    if hasattr(results, "head"):
        results = results.head(limit)

        return results.to_dict(
            orient="records"
        )

    if isinstance(results, list):
        return results[:limit]

    return results


# --------------------------------------------------
# TOOL 4 — Location Risk Analysis
# --------------------------------------------------

@tool
def get_location_risk(
    location: str
) -> dict:
    """
    Analyse historical safety incidents for a requested city
    or state.

    Use this tool when a safety officer asks about incident
    volume, serious outcomes or common incident types in a
    particular location.
    """

    df = load_safety_dataset()

    location = location.strip().lower()

    location_df = df[
        df["city"].fillna("").str.lower().eq(location)
        |
        df["state"].fillna("").str.lower().eq(location)
    ]

    if location_df.empty:
        return {
            "location": location,
            "total_reports": 0,
            "message": "No matching historical reports found."
        }

    top_incidents = (
        location_df["event_type"]
        .fillna("Unknown")
        .value_counts()
        .head(5)
    )

    hospitalized = int(
        location_df["hospitalized"].fillna(0).sum()
    )

    amputations = int(
        location_df["amputation"].fillna(0).sum()
    )

    return {
        "location": location,
        "total_reports": int(len(location_df)),
        "hospitalized": hospitalized,
        "amputations": amputations,
        "top_incident_types": top_incidents.to_dict()
    }


# --------------------------------------------------
# Tool registry
# --------------------------------------------------

SAFETY_TOOLS = [
    search_similar_incidents,
    analyze_safety_patterns,
    get_emerging_precursors,
    get_location_risk
]