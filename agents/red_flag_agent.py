"""
agents/red_flag_agent.py

Agent 2 — Red-Flag Detection

Reviews structured symptoms from Agent 1 and identifies
documented warning signs that may require clinician attention.

This agent does NOT diagnose or make autonomous medical decisions.

If the LLM is unavailable, a deterministic fallback is used
so that the clinical workflow can continue safely.
"""

from models.schemas import SymptomExtraction, RedFlagDetection, RedFlag

from utils.llm import call_structured_llm


SYSTEM_PROMPT = """
You are Agent 2 of a clinical decision-support prototype.

Your task is RED-FLAG DETECTION only.

Review the structured information provided by Agent 1 and
identify warning signs that are explicitly documented.

STRICT RULES:

1. Do NOT diagnose the patient.
2. Do NOT suggest a disease or medical condition.
3. Do NOT invent symptoms or measurements.
4. Use ONLY information explicitly present in the input.
5. Do NOT assume that missing information is normal.
6. If no documented red flag is supported, return an empty
   red_flags list and set no_red_flags_identified to true.
7. Keep reasoning concise.
8. "Urgent" means the documented finding may warrant prompt
   clinical assessment. It is not an emergency diagnosis.
9. The final clinical decision remains with a qualified
   healthcare professional.

Return only the requested RedFlagDetection structure.
"""


def _fallback_red_flag_detection(
    extraction: SymptomExtraction,
) -> RedFlagDetection:
    """
    Deterministic fallback.

    Only flags explicitly documented concerning symptoms.
    It does NOT infer that missing information is absent.
    """

    red_flags = []

    # ---------------------------------------------------------
    # Collect explicitly reported symptom names
    # ---------------------------------------------------------

    symptom_names = [
        symptom.name.lower().strip()
        for symptom in extraction.symptoms
    ]

    # ---------------------------------------------------------
    # Explicit warning signs
    # ---------------------------------------------------------

    emergency_phrases = {
        "chest pain": (
            "Chest pain was explicitly reported."
        ),
        "shortness of breath": (
            "Shortness of breath was explicitly reported."
        ),
        "breathlessness": (
            "Breathlessness was explicitly reported."
        ),
        "difficulty breathing": (
            "Difficulty breathing was explicitly reported."
        ),
        "fainting": (
            "Fainting was explicitly reported."
        ),
        "loss of consciousness": (
            "Loss of consciousness was explicitly reported."
        ),
        "confusion": (
            "Confusion was explicitly reported."
        ),
    }

    for symptom_name in symptom_names:

        if symptom_name in emergency_phrases:

            red_flags.append(
                RedFlag(
                    flag=symptom_name,
                    reason=emergency_phrases[symptom_name],
                    urgency="Prompt clinical assessment",
                )
            )

    # ---------------------------------------------------------
    # Explicit severe symptoms
    # ---------------------------------------------------------

    for symptom in extraction.symptoms:

        if (
            symptom.severity
            and symptom.severity.lower().strip()
            in {
                "severe",
                "very severe",
                "extreme",
            }
        ):

            # Only report the severity that the patient
            # explicitly provided.
            red_flags.append(
                RedFlag(
                    flag=f"Severe {symptom.name}",
                    reason=(
                        f"The symptom '{symptom.name}' "
                        "was explicitly reported as severe."
                    ),
                    urgency="Prompt clinical assessment",
                )
            )

    # ---------------------------------------------------------
    # Remove duplicate flags
    # ---------------------------------------------------------

    unique_flags = []
    seen = set()

    for flag in red_flags:

        key = flag.flag.lower()

        if key not in seen:
            seen.add(key)
            unique_flags.append(flag)

    # ---------------------------------------------------------
    # Return result
    # ---------------------------------------------------------

    if unique_flags:

        return RedFlagDetection(
            red_flags=unique_flags,
            no_red_flags_identified=False,
            assessment_notes=(
                "Red flags were identified only from "
                "explicitly documented information."
            ),
        )

    return RedFlagDetection(
        red_flags=[],
        no_red_flags_identified=True,
        assessment_notes=(
            "No documented red flags were identified "
            "from the information provided. Missing "
            "information was not assumed to be normal."
        ),
    )


def run_red_flag_agent(
    extraction: SymptomExtraction,
) -> RedFlagDetection:
    """
    Run Agent 2 using structured output from Agent 1.

    If the LLM is unavailable or returns unusable output,
    use the deterministic fallback.
    """

    user_prompt = f"""
Review the following structured symptom extraction from Agent 1.

SYMPTOMS:
{extraction.symptoms}

MEDICAL HISTORY:
{extraction.medical_history}

MEDICATIONS:
{extraction.medications}

ALLERGIES:
{extraction.allergies}

EXTRACTION NOTES:
{extraction.extraction_notes}

Identify ONLY documented red flags.

If no documented red flags are present, return:

red_flags: []

no_red_flags_identified: true

Do not infer that missing information is normal.
Do not diagnose.
Do not name diseases.

Return the required RedFlagDetection structure.
"""

    # =========================================================
    # ONE LLM ATTEMPT
    # =========================================================

    try:

        result = call_structured_llm(
            system_prompt=SYSTEM_PROMPT,
            user_prompt=user_prompt,
            response_model=RedFlagDetection,
            max_tokens=1024,
            temperature=0.0,
            max_retries=1,
        )

        print(
            "Agent 2 completed red-flag detection using LLM."
        )

        return result

    except Exception as error:

        print(
            "Agent 2 LLM unavailable."
        )

        print(
            f"Agent 2 fallback reason: {error}"
        )

        fallback_result = _fallback_red_flag_detection(
            extraction
        )

        print(
            "Agent 2 completed red-flag detection "
            "using deterministic fallback."
        )

        return fallback_result