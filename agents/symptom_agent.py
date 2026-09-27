"""
agents/symptom_agent.py

Agent 1 — Symptom Extraction

LLM-first semantic extraction for arbitrary patient presentations.
A conservative generic fallback is used only if the LLM is unavailable
or returns an empty symptom list.
"""

import re

from models.schemas import (
    PatientInput,
    SymptomExtraction,
    ExtractedSymptom,
)

from utils.llm import call_structured_llm


SYSTEM_PROMPT = """
You are Agent 1 of a clinical decision-support system.

TASK:
Extract the patient's DISTINCT symptoms from natural-language
patient information.

This is semantic extraction only. Do NOT diagnose.

RULES:

1. Recognize symptoms from ANY body system.
2. Do not use a fixed symptom dictionary.
3. Extract only information explicitly provided.
4. Do not invent symptoms.
5. Combine repeated descriptions of the same symptom.
6. Never treat an entire sentence as a symptom.
7. Descriptive details belong in the symptom's notes.
8. Preserve:
   - anatomical location
   - side
   - duration
   - severity
   - frequency
   - onset
   - progression
   - triggers
   - other relevant characteristics
9. Convert colloquial wording into a concise clinical description.
10. Correct obvious spelling mistakes when the intended symptom is clear.
11. Extract medical history, medications and allergies only when
    explicitly provided.
12. Do not infer diagnoses.
13. Do not infer missing information.
14. Each distinct symptom should appear exactly once.

If the main complaint contains multiple clearly separated symptoms,
such as "stomach ache, loose motion, vomiting", extract them as
three separate symptoms.

For example:

Input:
"stomach ache, loose motion, vomiting for 3 days"

Output should contain:
- stomach ache
- loose motion
- vomiting

Do NOT combine them into one symptom.

If one symptom is described across several sentences, combine those
sentences into ONE symptom with the additional details stored in
that symptom's fields or notes.

Return ONLY the SymptomExtraction structure.
"""


def _clean_text(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip()


def _normalize_symptom_name(name: str) -> str:
    """
    Conservative normalization of obvious spelling variations.

    This does NOT use a disease/symptom dictionary.
    """

    cleaned = _clean_text(name)

    corrections = {
        "vomitting": "vomiting",
        "vomitting sensation": "vomiting sensation",
        "diarhea": "diarrhea",
        "diarrhoea": "diarrhea",
        "loose motions": "loose motion",
        "stomache ache": "stomach ache",
    }

    return corrections.get(
        cleaned.lower(),
        cleaned,
    )


def _normalize_extraction(
    result: SymptomExtraction,
) -> SymptomExtraction:

    for symptom in result.symptoms:
        symptom.name = _normalize_symptom_name(
            symptom.name
        )

    return result


def _generic_fallback(
    patient: PatientInput,
) -> SymptomExtraction:

    complaint = _clean_text(patient.main_complaint)
    description = _clean_text(patient.symptom_description)

    symptoms = []

    if complaint:

        complaint_parts = [
            item.strip()
            for item in complaint.split(",")
            if item.strip()
        ]

        for item in complaint_parts:
            symptoms.append(
                ExtractedSymptom(
                    name=_normalize_symptom_name(item),
                    duration=(
                        patient.duration
                        if patient.duration
                        else "Not provided"
                    ),
                    severity=(
                        patient.severity
                        if patient.severity
                        else "Not provided"
                    ),
                    notes=(
                        description
                        if description
                        else "Not provided"
                    ),
                )
            )

    medical_history = []
    medications = []
    allergies = []

    history = _clean_text(patient.medical_history)

    if history and history.lower() not in {
        "none",
        "none provided",
        "not provided",
        "no medical history",
    }:
        medical_history = [history]

    medication_text = _clean_text(patient.medications)

    if medication_text and medication_text.lower() not in {
        "none",
        "none provided",
        "not provided",
        "no medications",
    }:
        medications = [
            item.strip()
            for item in medication_text.split(",")
            if item.strip()
        ]

    allergy_text = _clean_text(patient.allergies)

    if allergy_text and allergy_text.lower() not in {
        "none",
        "none provided",
        "not provided",
        "no known allergies",
    }:
        allergies = [
            item.strip()
            for item in allergy_text.split(",")
            if item.strip()
        ]

    return SymptomExtraction(
        symptoms=symptoms,
        medical_history=medical_history,
        medications=medications,
        allergies=allergies,
        extraction_notes=(
            "LLM extraction was unavailable. The fallback preserves "
            "explicitly supplied complaint items and separates only "
            "clearly comma-separated symptoms."
        ),
    )


def run_symptom_agent(
    patient: PatientInput,
) -> SymptomExtraction:

    user_prompt = f"""
Extract the patient's distinct symptoms.

MAIN COMPLAINT:
{patient.main_complaint}

SYMPTOM DESCRIPTION:
{patient.symptom_description}

DURATION:
{patient.duration}

SEVERITY:
{patient.severity}

ONSET:
{patient.onset}

PROGRESSION:
{patient.progression}

MEDICAL HISTORY:
{patient.medical_history}

MEDICATIONS:
{patient.medications}

ALLERGIES:
{patient.allergies}

OTHER INFORMATION:
{patient.other_info}

Remember:

- Handle ANY type of symptom.
- Do not use a predefined symptom list.
- If the main complaint explicitly lists multiple symptoms,
  extract each distinct symptom separately.
- For example, "stomach ache, loose motion, vomiting" means
  three separate symptoms.
- Correct obvious spelling mistakes when the intended symptom
  is clear.
- Combine repeated descriptions of the same symptom.
- Do not turn descriptive sentences into separate symptoms.
- Preserve useful characteristics in the symptom fields/notes.
- Do not diagnose.
- Do not invent information.
"""

    try:
        result = call_structured_llm(
            system_prompt=SYSTEM_PROMPT,
            user_prompt=user_prompt,
            response_model=SymptomExtraction,
            max_tokens=1500,
            temperature=0.0,
            max_retries=1,
        )

        result = _normalize_extraction(result)

        if result.symptoms:
            print(
                "Agent 1 completed symptom extraction using LLM."
            )
            return result

        print(
            "Agent 1 LLM returned an empty symptom list. "
            "Using conservative fallback."
        )

    except Exception as error:
        print("Agent 1 LLM unavailable.")
        print(f"Agent 1 error: {error}")

    result = _generic_fallback(patient)

    print(
        "Agent 1 completed symptom extraction using fallback."
    )

    return result