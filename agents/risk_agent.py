import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from feedback.feedback_manager import format_overrides_for_prompt
from typing import Literal
from pydantic import BaseModel, Field
from langchain_google_genai import ChatGoogleGenerativeAI

from report_parser import SafetyReportAnalysis


# --------------------------------------------------
# Structured output from Risk Scoring Agent
# --------------------------------------------------

class RiskAssessment(BaseModel):
    risk_level: Literal["LOW", "MEDIUM", "HIGH"] = Field(
        description="Overall safety risk level"
    )

    confidence: float = Field(
        ge=0.0,
        le=1.0,
        description="Confidence in the classification from 0 to 1"
    )

    reasoning: str = Field(
        description="Short explanation for the assigned risk level"
    )

    recommended_action: str = Field(
        description="Practical action a safety officer should consider"
    )


# --------------------------------------------------
# Gemini model
# --------------------------------------------------

llm = ChatGoogleGenerativeAI(
    model="gemini-3.6-flash"
)

structured_llm = llm.with_structured_output(RiskAssessment)


# --------------------------------------------------
# Risk Scoring Agent
# --------------------------------------------------

def assess_risk(report: SafetyReportAnalysis) -> RiskAssessment:

    # Retrieve latest human corrections
    correction_examples = format_overrides_for_prompt(limit=5)

    prompt = f"""
You are the Risk Scoring Agent for SafeSight AI.

Your task is to classify the safety report as:
LOW, MEDIUM, or HIGH.

RISK FRAMEWORK

LOW:
- Minor unsafe condition or behaviour
- Limited injury potential
- No indication of serious harm
- Routine corrective action is appropriate

MEDIUM:
- Meaningful hazard that could cause injury
- Unsafe behaviour, equipment, environment, or process
- Could escalate if repeated or unresolved
- Timely corrective action is required

HIGH:
- Serious injury occurred, OR
- Clear potential for severe injury, permanent disability,
  hospitalization, fatality, or major equipment incident
- Urgent attention is required


HUMAN SAFETY OFFICER CORRECTIONS

The following are recent corrections made by safety officers.

Use them as examples of how human safety officers apply risk
judgement. Do not blindly copy their classifications. Apply the
same reasoning principles to the current report.

{correction_examples}


CURRENT PARSED SAFETY REPORT

Hazard:
{report.hazard_type}

Risk factors:
{report.risk_factors}

Location/context:
{report.location_context}

Equipment involved:
{report.equipment_involved}

People at risk:
{report.people_at_risk}

Potential consequence:
{report.potential_consequence}


INSTRUCTIONS

- Consider both actual and potential consequences.
- Consider the extracted risk factors and context.
- Use relevant human corrections as additional guidance.
- Do not classify based on keywords alone.
- Provide concise reasoning.
- Confidence must be between 0 and 1.
- Recommend a practical safety response.
"""

    return structured_llm.invoke(prompt)

# --------------------------------------------------
# Test
# --------------------------------------------------

if __name__ == "__main__":

    example = SafetyReportAnalysis(
        hazard_type="Physical altercation and fall during inmate escort",

        risk_factors=[
            "Disruptive inmate behavior",
            "Requirement for physical force",
            "Loss of balance and impact from multiple falling bodies"
        ],

        location_context="Correctional facility",

        equipment_involved=[
            "Inmate restraints"
        ],

        people_at_risk=[
            "Correctional facility guards",
            "Lieutenant"
        ],

        potential_consequence="Fractured fibula requiring hospitalization"
    )

    result = assess_risk(example)

    print("\n--- SafeSight Risk Assessment ---")
    print(result.model_dump_json(indent=2))