from report_parser import parse_safety_report
from risk_agent import assess_risk


def analyze_report(report_text: str):

    # Agent 1: Extract structured safety information
    parsed_report = parse_safety_report(report_text)

    # Agent 2: Assess the extracted risk
    risk_assessment = assess_risk(parsed_report)

    return {
        "parsed_report": parsed_report.model_dump(),
        "risk_assessment": risk_assessment.model_dump()
    }


if __name__ == "__main__":

    report = """
    Three correctional facility guards were escorting a restrained
    federal prison inmate when he became disruptive, requiring the use
    of force. Two guards and the inmate fell onto the Lieutenant's
    right leg, fracturing his fibula. He was transported to the
    hospital and released the following day.
    """

    result = analyze_report(report)

    print("\n========== SAFESIGHT AI ==========")

    print("\n--- Parsed Report ---")
    for key, value in result["parsed_report"].items():
        print(f"{key}: {value}")

    print("\n--- Risk Assessment ---")
    for key, value in result["risk_assessment"].items():
        print(f"{key}: {value}")