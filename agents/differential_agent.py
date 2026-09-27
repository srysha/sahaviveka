from typing import List

from pydantic import BaseModel, Field

from utils.llm import (
    call_structured_llm,
    LLMConfigError,
    LLMRequestError,
    LLMOutputError,
)
from workflow.case_state import CaseState


# ============================================================
# AGENT 5 - MOST EVIDENCE-SUPPORTED CONDITION
# ============================================================

class PossibleCondition(BaseModel):
    condition: str
    rationale: str
    evidence_source_ids: List[str] = Field(default_factory=list)
    confidence: str = "Limited evidence support"


class DifferentialAnalysis(BaseModel):
    most_supported_condition: PossibleCondition
    analysis_notes: str = ""


SYSTEM_PROMPT = """
You are Agent 5 in a clinical decision-support system.

Your task is to identify the ONE most evidence-supported
possible condition for the patient's clinical presentation.

Use ONLY:
1. The patient's extracted symptoms.
2. The medical evidence retrieved by Agent 4.

IMPORTANT RULES:

- Return EXACTLY ONE possible condition.
- Do NOT return a list of conditions.
- Do NOT make a definitive diagnosis.
- Do NOT recommend treatment.
- Do NOT invent a condition.
- Do NOT use outside medical knowledge that is not present
  in the retrieved evidence.
- Ignore evidence that is clearly unrelated.
- Select the condition with the strongest direct and
  comprehensive evidence match.
- Every selected condition MUST include at least one
  supporting evidence source ID.
- Confidence describes the STRENGTH OF THE RETRIEVED
  EVIDENCE, NOT the probability that the patient has
  the condition.
- The clinician remains responsible for the final
  assessment and decision.

IMPORTANT OUTPUT REQUIREMENT:

You MUST populate the "most_supported_condition" field.

Do not return only "analysis_notes".

If the evidence supports a reasonable condition, put it in
"most_supported_condition".
"""


def run_differential_agent(state: CaseState) -> CaseState:

    if not state.symptom_extraction:
        state.possible_conditions = []
        return state

    if not state.evidence:
        state.possible_conditions = []
        return state

    symptom_text = ", ".join(
        symptom.name
        for symptom in state.symptom_extraction.symptoms
    )

    evidence_text = ""

    for item in state.evidence:

        source_id = item.source_id or item.source_url

        evidence_text += f"""
SOURCE ID: {source_id}
ORGANIZATION: {item.source_organization}
TITLE: {item.title}
TOPIC: {item.topic}

EVIDENCE:
{item.evidence_text}

----------------------------------------
"""

    user_prompt = f"""
PATIENT SYMPTOMS:
{symptom_text}

RETRIEVED MEDICAL EVIDENCE:
{evidence_text}

Identify the ONE most evidence-supported possible condition.

Compare the patient's symptoms against ALL retrieved evidence.

Choose the single condition with the strongest direct
evidence match.

Your response MUST contain:

most_supported_condition:
    condition:
    rationale:
    evidence_source_ids:
    confidence:

analysis_notes:

Do not return multiple conditions.

Do not leave most_supported_condition empty.
"""

    try:

        result = call_structured_llm(
            system_prompt=SYSTEM_PROMPT,
            user_prompt=user_prompt,
            response_model=DifferentialAnalysis,
            temperature=0.0,
            max_tokens=1200,

            # Only one attempt.
            # If OpenRouter is unavailable, use the deterministic
            # evidence-based fallback instead of wasting quota
            # on repeated 429 requests.
            max_retries=1,
        )

        condition = result.most_supported_condition

        state.possible_conditions = [
            condition.model_dump()
        ]

        print(
            "Agent 5 selected one "
            "most evidence-supported condition."
        )

        return state

    except (
        LLMConfigError,
        LLMRequestError,
        LLMOutputError,
    ):

        print("Agent 5 LLM response unavailable.")
        print("Using safe evidence-based fallback.")

        state.possible_conditions = _fallback_differential(
            state
        )

        if state.possible_conditions:
            print(
                "Agent 5 fallback selected one "
                "evidence-supported condition."
            )
        else:
            print(
                "Agent 5 fallback found no condition "
                "directly supported by the retrieved evidence."
            )

        return state


# ============================================================
# SAFE FALLBACK
# ============================================================

def _fallback_differential(state: CaseState) -> list[dict]:

    """
    Conservative deterministic fallback used only when the LLM
    cannot provide a valid structured response.

    The fallback may select a condition ONLY when:
    1. The patient's symptoms match the condition, AND
    2. The retrieved Agent 4 evidence explicitly supports that
       condition.

    It never creates evidence that was not retrieved.
    """

    symptoms = {
        symptom.name.lower().strip()
        for symptom in state.symptom_extraction.symptoms
    }

    candidates = []

    for item in state.evidence:

        evidence = item.evidence_text.lower()
        title = item.title.lower()
        topic = (item.topic or "").lower()

        source_id = item.source_id or item.source_url

        # ----------------------------------------------------
        # Gastroenteritis
        # ----------------------------------------------------

        gastroenteritis_evidence = (
            "gastroenteritis" in evidence
            or "gastroenteritis" in title
            or "gastroenteritis" in topic
        )

        gi_symptoms = {
            "stomach ache",
            "abdominal pain",
            "abdominal cramps",
            "diarrhea",
            "loose motion",
            "vomiting",
            "nausea",
        }

        matching_symptoms = [
            symptom
            for symptom in symptoms
            if symptom in gi_symptoms
        ]

        if gastroenteritis_evidence and matching_symptoms:

            # Require more than one directly matching GI symptom
            # before selecting gastroenteritis.
            if len(set(matching_symptoms)) >= 2:

                candidates.append(
                    {
                        "condition": "Gastroenteritis",

                        "rationale":
                            "The retrieved medical evidence "
                            "explicitly discusses gastroenteritis "
                            "in association with symptoms matching "
                            "the patient's presentation, including "
                            + ", ".join(sorted(set(matching_symptoms)))
                            + ".",

                        "evidence_source_ids":
                            [source_id],

                        "confidence":
                            "Strong evidence support",

                        "_score":
                            len(set(matching_symptoms)) * 10,
                    }
                )

        # ----------------------------------------------------
        # Common cold / URI
        # ----------------------------------------------------

        if (
            "common cold" in evidence
            or "non-specific upper respiratory tract infection"
            in evidence
        ):

            matching_symptoms = sum(
                symptom in symptoms
                for symptom in [
                    "blocked nose",
                    "runny nose",
                    "stuffy nose",
                    "sore throat",
                    "dry cough",
                    "cough",
                    "headache",
                ]
            )

            if matching_symptoms > 0:

                candidates.append(
                    {
                        "condition":
                            "Common cold / non-specific "
                            "upper respiratory tract infection",

                        "rationale":
                            "The retrieved medical evidence "
                            "directly matches the patient's "
                            "respiratory symptoms.",

                        "evidence_source_ids":
                            [source_id],

                        "confidence":
                            "High",

                        "_score":
                            matching_symptoms * 10,
                    }
                )

        # ----------------------------------------------------
        # Influenza
        # ----------------------------------------------------

        if (
            "influenza" in evidence
            or "influenza" in title
        ):

            matching_symptoms = sum(
                symptom in symptoms
                for symptom in [
                    "sore throat",
                    "dry cough",
                    "cough",
                    "headache",
                    "blocked nose",
                    "runny nose",
                ]
            )

            if matching_symptoms > 0:

                candidates.append(
                    {
                        "condition": "Influenza",

                        "rationale":
                            "The retrieved medical evidence "
                            "describes influenza symptoms that "
                            "overlap with the patient's presentation.",

                        "evidence_source_ids":
                            [source_id],

                        "confidence":
                            "Moderate",

                        "_score":
                            matching_symptoms * 8,
                    }
                )

        # ----------------------------------------------------
        # COVID-19
        # ----------------------------------------------------

        if (
            "covid-19" in evidence
            or "covid-19" in title
        ):

            matching_symptoms = sum(
                symptom in symptoms
                for symptom in [
                    "sore throat",
                    "dry cough",
                    "cough",
                    "headache",
                    "blocked nose",
                    "runny nose",
                ]
            )

            if matching_symptoms > 0:

                candidates.append(
                    {
                        "condition": "COVID-19",

                        "rationale":
                            "The retrieved evidence describes "
                            "symptoms that overlap with the "
                            "patient's presentation.",

                        "evidence_source_ids":
                            [source_id],

                        "confidence":
                            "Limited evidence support",

                        "_score":
                            matching_symptoms * 6,
                    }
                )

    if not candidates:
        return []

    candidates.sort(
        key=lambda item: item["_score"],
        reverse=True
    )

    best = candidates[0]

    best.pop("_score", None)

    return [best]