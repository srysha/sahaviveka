"""
agents/missing_info_agent.py

Agent 3 — Missing Information

Identifies clinically useful information that was not provided.

This agent does not diagnose and does not assume that missing
information is normal.
"""

from typing import List

from pydantic import BaseModel, Field

from workflow.case_state import CaseState


class MissingInformationResult(BaseModel):
    missing_information: List[str] = Field(
        default_factory=list
    )
    reasoning: str = ""


def _is_missing(value) -> bool:
    if value is None:
        return True

    text = str(value).strip().lower()

    return text in {
        "",
        "not provided",
        "unknown",
        "unspecified",
        "none provided",
    }


def _explicitly_none(value) -> bool:
    if value is None:
        return False

    text = str(value).strip().lower()

    return text in {
        "none",
        "no",
        "nil",
        "none reported",
        "no history",
        "no medical history",
        "no medications",
        "no known allergies",
        "none known",
    }


def _contains_any(
    text: str,
    terms: list[str],
) -> bool:

    text = text.lower()

    return any(
        term in text
        for term in terms
    )


def _deterministic_missing_information(
    state: CaseState,
) -> list[str]:

    extraction = state.symptom_extraction

    if extraction is None:
        return [
            "Structured symptom information"
        ]

    symptoms = extraction.symptoms

    if not symptoms:
        return [
            "Clear description of the main symptoms"
        ]

    symptom_text = " ".join(
        symptom.name.lower()
        for symptom in symptoms
    )

    missing = []

    # -------------------------------------------------------------
    # General information useful across presentations
    # -------------------------------------------------------------

    missing.append(
        "Whether the symptoms are improving, worsening, "
        "or staying the same"
    )

    if all(
        _is_missing(symptom.duration)
        for symptom in symptoms
    ):
        missing.append(
            "Duration of each symptom"
        )

    if all(
        _is_missing(symptom.frequency)
        for symptom in symptoms
    ):
        missing.append(
            "Frequency or number of episodes of the symptoms"
        )

    # -------------------------------------------------------------
    # Fever / systemic symptoms
    # -------------------------------------------------------------

    if not _contains_any(
        symptom_text,
        [
            "fever",
            "temperature",
            "chills",
        ],
    ):
        missing.append(
            "Whether fever, chills, or a measured temperature "
            "is present"
        )

    # -------------------------------------------------------------
    # General red-flag screening
    # -------------------------------------------------------------

    missing.append(
        "Whether any severe or rapidly worsening symptoms "
        "are present"
    )

    missing.append(
        "Whether there is any fainting, confusion, or "
        "significant weakness"
    )

    # -------------------------------------------------------------
    # Gastrointestinal presentations
    # -------------------------------------------------------------

    gastrointestinal = _contains_any(
        symptom_text,
        [
            "stomach",
            "abdominal",
            "abdomen",
            "belly",
            "vomit",
            "vomiting",
            "nausea",
            "diarrhea",
            "diarrhoea",
            "loose motion",
            "loose stool",
            "constipation",
            "stool",
            "bowel",
        ],
    )

    if gastrointestinal:

        missing.append(
            "Location of abdominal discomfort or pain and "
            "whether it is constant or intermittent"
        )

        missing.append(
            "Frequency and characteristics of vomiting or "
            "loose stools, if present"
        )

        missing.append(
            "Whether there is blood in vomit or stool, "
            "if applicable"
        )

        missing.append(
            "Whether the patient is able to maintain normal "
            "food and fluid intake"
        )

        missing.append(
            "Whether there are signs of reduced hydration "
            "such as markedly reduced urination or dizziness"
        )

    # -------------------------------------------------------------
    # Respiratory presentations
    # -------------------------------------------------------------

    respiratory = _contains_any(
        symptom_text,
        [
            "cough",
            "sore throat",
            "runny nose",
            "blocked nose",
            "stuffy nose",
            "breathlessness",
            "shortness of breath",
            "wheezing",
            "chest",
        ],
    )

    if respiratory:

        missing.append(
            "Whether shortness of breath, wheezing, chest pain, "
            "or difficulty breathing is present"
        )

        missing.append(
            "Recent exposure to someone with similar "
            "respiratory symptoms"
        )

    # -------------------------------------------------------------
    # Neurological presentations
    # -------------------------------------------------------------

    neurological = _contains_any(
        symptom_text,
        [
            "headache",
            "dizziness",
            "vertigo",
            "weakness",
            "numbness",
            "tingling",
            "confusion",
            "seizure",
            "fainting",
        ],
    )

    if neurological:

        missing.append(
            "Whether there are changes in consciousness, "
            "new weakness, numbness, or difficulty speaking"
        )

    # -------------------------------------------------------------
    # Urinary presentations
    # -------------------------------------------------------------

    urinary = _contains_any(
        symptom_text,
        [
            "urine",
            "urination",
            "urinary",
            "burning while urinating",
            "painful urination",
            "flank pain",
        ],
    )

    if urinary:

        missing.append(
            "Urinary frequency, urgency, pain, or changes "
            "in urine appearance"
        )

    # -------------------------------------------------------------
    # Skin / allergic presentations
    # -------------------------------------------------------------

    skin_or_allergy = _contains_any(
        symptom_text,
        [
            "rash",
            "itching",
            "hives",
            "swelling",
            "skin",
            "allergic",
            "allergy",
        ],
    )

    if skin_or_allergy:

        missing.append(
            "Timing and progression of the skin or allergic "
            "symptoms and any known trigger"
        )

    # -------------------------------------------------------------
    # Relevant history
    # -------------------------------------------------------------

    if not extraction.medical_history:
        missing.append(
            "Relevant medical history"
        )

    if not extraction.medications:
        missing.append(
            "Current medications and recent medication use"
        )

    if not extraction.allergies:
        missing.append(
            "Known allergies"
        )

    # -------------------------------------------------------------
    # Remove duplicates while preserving order
    # -------------------------------------------------------------

    return list(
        dict.fromkeys(missing)
    )


def run_missing_information_agent(
    state: CaseState,
) -> CaseState:

    missing = _deterministic_missing_information(
        state
    )

    state.missing_information = missing

    print(
        "Agent 3 completed missing-information assessment "
        "using deterministic clinical checks."
    )

    return state