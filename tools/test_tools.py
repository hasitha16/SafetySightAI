from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parent.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))


from tools.safety_tools import (
    search_similar_incidents,
    analyze_safety_patterns,
    get_emerging_precursors,
    get_location_risk
)


print("\n====================================")
print("       SafeSight Tool Test")
print("====================================")


# --------------------------------------------------
# Test Tool 1
# --------------------------------------------------

print("\n[Tool 1] Historical Incident Search")

incidents = search_similar_incidents.invoke(
    {
        "query": (
            "worker caught in machinery "
            "during cleaning"
        ),
        "limit": 3
    }
)

for number, incident in enumerate(
    incidents,
    start=1
):

    print(f"\nResult {number}")

    print(
        "Report ID:",
        incident.get("report_id")
    )

    print(
        "Incident Type:",
        incident.get("event_type")
    )

# --------------------------------------------------
# Test Tool 2
# --------------------------------------------------

print("\n====================================")
print("[Tool 2] Dataset-wide Pattern Analysis")

patterns = analyze_safety_patterns.invoke({})

print("\nSerious Outcomes:")
print(
    patterns.get("serious_outcomes")
)

print("\nTop Incident Types:")
print(
    patterns.get("incident_types")
)


# --------------------------------------------------
# Test Tool 3
# --------------------------------------------------

print("\n====================================")
print("[Tool 3] Emerging Precursor Detection")

precursors = get_emerging_precursors.invoke(
    {
        "limit": 5
    }
)

for number, precursor in enumerate(
    precursors,
    start=1
):
    print(f"\nPrecursor {number}")
    print(precursor)

# --------------------------------------------------
# Test Tool 4
# --------------------------------------------------

print("\n====================================")
print("[Tool 4] Location Risk Analysis")

location_result = get_location_risk.invoke(
    {
        "location": "Texas"
    }
)

print(location_result)