import os
from typing import List

from dotenv import load_dotenv
from pydantic import BaseModel, Field
from langchain_google_genai import ChatGoogleGenerativeAI


load_dotenv()


# --------------------------------------------------
# Structured output produced by the parsing agent
# --------------------------------------------------

class SafetyReportAnalysis(BaseModel):
    hazard_type: str = Field(
        description="Primary hazard identified in the safety report"
    )

    risk_factors: List[str] = Field(
        description="Specific factors that contributed to the incident"
    )

    location_context: str = Field(
        description="Workplace or location context mentioned in the report"
    )

    equipment_involved: List[str] = Field(
        description="Equipment, machinery, tools, vehicles, or objects involved"
    )

    people_at_risk: List[str] = Field(
        description="People or worker groups exposed to the hazard"
    )

    potential_consequence: str = Field(
        description="Potential or actual safety consequence"
    )


# --------------------------------------------------
# Gemini model
# --------------------------------------------------

llm = ChatGoogleGenerativeAI(
    model="gemini-3.6-flash",
    google_api_key=os.getenv("GEMINI_API_KEY")
)


# Force Gemini to return our Pydantic structure
structured_llm = llm.with_structured_output(SafetyReportAnalysis)


# --------------------------------------------------
# Report Parsing Agent
# --------------------------------------------------

def parse_safety_report(report_text: str) -> SafetyReportAnalysis:

    prompt = f"""
You are the Report Parsing Agent for SafeSight AI.

Your task is to analyse a workplace safety report and extract factual
safety information from the report.

Important rules:
- Use only information supported by the report.
- Do not invent missing details.
- Keep risk factors specific and concise.
- Identify the main hazard.
- Identify equipment only when it is actually mentioned.
- Identify people at risk when possible.
- Describe the actual or reasonably implied safety consequence.
- Do NOT assign a Low, Medium, or High risk level. Another agent
  will perform risk classification.

SAFETY REPORT:
{report_text}
"""

    return structured_llm.invoke(prompt)


# --------------------------------------------------
# Quick test
# --------------------------------------------------

if __name__ == "__main__":

    test_report = """
    Three correctional facility guards were escorting a restrained
    federal prison inmate when he became disruptive, requiring the use
    of force. Two guards and the inmate fell onto the Lieutenant's
    right leg, fracturing his fibula. He was transported to the
    hospital and released the following day.
    """

    result = parse_safety_report(test_report)

    print("\n--- SafeSight Report Parser ---")
    print(result.model_dump_json(indent=2))