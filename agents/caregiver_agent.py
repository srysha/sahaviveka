"""
agents/caregiver_agent.py

Creates a simplified caregiver-facing explanation from the
clinical decision-support case.

This does not provide a diagnosis or treatment plan.
"""

from pydantic import BaseModel

from utils.llm import call_structured_llm
from workflow.case_state import CaseState


class CaregiverExplanation(BaseModel):
    summary: str
    important_points: list[str] = []
    questions_for_clinician: list[str] = []
    safety_note: str


def run_caregiver_agent(
    state: CaseState,
) -> CaregiverExplanation:

    symptoms = "\n".join(
        f"- {symptom.name}: "
        f"duration={symptom.duration}, "
        f"severity={symptom.severity}"
        for symptom in state.symptom_extraction.symptoms
    )

    red_flags = "\n".join(
        f"- {flag.flag}: {flag.reason}"
        for flag in state.red_flag_detection.red_flags
    )

    possible_conditions = "\n".join(
        f"- {condition.get('condition', 'Unspecified')}: "
        f"{condition.get('rationale', '')}"
        for condition in state.possible_conditions
    )

    system_prompt = """
You are the caregiver communication component of a
clinical decision-support prototype.

Your job is to explain the information already produced by the
system in simple, understandable language for a caregiver.

STRICT RULES:

1. Do NOT diagnose the patient.
2. Do NOT say that the patient definitely has a condition.
3. Do NOT recommend medication or treatment.
4. Do NOT invent information.
5. Do not introduce medical claims that are not present in the
   provided information.
6. Clearly distinguish possible conditions from confirmed diagnoses.
7. Explain medical terminology in simple language where possible.
8. Encourage discussion with a qualified healthcare professional.
9. If red flags are present, clearly state that clinician review
   is important.
10. The caregiver explanation must not replace professional
    clinical assessment.

Return only the requested structured JSON.
"""

    user_prompt = f"""
SYMPTOMS:
{symptoms}

RED FLAGS:
{red_flags if red_flags else "No red flags were identified from the information provided."}

POSSIBLE CONDITIONS FOR CLINICIAN REVIEW:
{possible_conditions if possible_conditions else "None identified from the available evidence."}

Create a simple caregiver-facing explanation.
"""

    return call_structured_llm(
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        response_model=CaregiverExplanation,
        temperature=0.0,
    )