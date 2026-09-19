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

    prompt = f"""
You are the Risk Scoring Agent for SafeSight AI.

Evaluate the workplace safety information below.

Use this risk framework:

LOW:
- Minor unsafe condition or behaviour
- Limited potential for injury
- No indication of serious harm
- Routine corrective action is appropriate

MEDIUM:
- Meaningful hazard that could cause injury
- Unsafe behaviour, equipment, environment, or process
- Could escalate if repeated or left unresolved
- Requires timely corrective action

HIGH:
- Serious injury has occurred OR
- There is clear potential for severe injury, permanent disability,
  hospitalization, fatality, major equipment incident, or similarly
  severe consequence
- Requires urgent safety attention

Important:
- Consider both actual consequence and potential consequence.
- Do not classify based only on keywords.
- Use the extracted risk factors and context.
- Give concise reasoning.
- Confidence must be between 0 and 1.
- Recommend a practical safety response.

PARSED SAFETY REPORT:

Hazard:
{report.hazard_type}

Risk factors:
{", ".join(report.risk_factors)}

Location/context:
{report.location_context}

Equipment involved:
{", ".join(report.equipment_involved) if report.equipment_involved else "None identified"}

People at risk:
{", ".join(report.people_at_risk)}

Potential/actual consequence:
{report.potential_consequence}
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