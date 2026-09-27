"""
agents/clinical_note_agent.py

Agent 6: Clinical Note Generation

Creates a concise clinician-facing note from information
already produced by the workflow.

The LLM is used when available.
A deterministic fallback is used if the LLM is unavailable.
"""

from pydantic import BaseModel, Field

from utils.llm import call_structured_llm
from workflow.case_state import CaseState


class ClinicalNote(BaseModel):

    presenting_complaint: str

    symptom_summary: str

    relevant_history: str

    red_flag_summary: str

    missing_information: list[str] = Field(
        default_factory=list
    )

    possible_conditions: list[str] = Field(
        default_factory=list
    )

    evidence_summary: str

    clinician_review_note: str


def _build_fallback_note(state: CaseState) -> ClinicalNote:
    """
    Deterministic fallback.

    Uses ONLY information already present in CaseState.

    The fallback deliberately keeps the evidence section
    concise and does not include URLs or full article text.
    """

    # ---------------------------------------------------------
    # Symptoms
    # ---------------------------------------------------------

    if (
        state.symptom_extraction
        and state.symptom_extraction.symptoms
    ):

        symptom_parts = []

        for symptom in state.symptom_extraction.symptoms:

            symptom_parts.append(
                f"{symptom.name} "
                f"(duration: {symptom.duration}, "
                f"severity: {symptom.severity})"
            )

        symptom_summary = ", ".join(symptom_parts)

    else:

        symptom_summary = "No symptoms were extracted."

    # ---------------------------------------------------------
    # History
    # ---------------------------------------------------------

    history_parts = []

    if state.patient.age != "Not provided":
        history_parts.append(
            f"Age: {state.patient.age}"
        )

    if state.patient.sex != "Not provided":
        history_parts.append(
            f"Sex: {state.patient.sex}"
        )

    if state.patient.medical_history != "Not provided":
        history_parts.append(
            f"Medical history: "
            f"{state.patient.medical_history}"
        )

    if state.patient.medications != "Not provided":
        history_parts.append(
            f"Medications: "
            f"{state.patient.medications}"
        )

    if state.patient.allergies != "Not provided":
        history_parts.append(
            f"Allergies: "
            f"{state.patient.allergies}"
        )

    if history_parts:
        relevant_history = "; ".join(history_parts)
    else:
        relevant_history = "No additional history provided."

    # ---------------------------------------------------------
    # Red flags
    # ---------------------------------------------------------

    if (
        state.red_flag_detection
        and state.red_flag_detection.red_flags
    ):

        red_flag_parts = []

        for flag in state.red_flag_detection.red_flags:

            red_flag_parts.append(
                f"{flag.flag}: "
                f"{flag.reason} "
                f"(Urgency: {flag.urgency})"
            )

        red_flag_summary = "; ".join(red_flag_parts)

    else:

        red_flag_summary = (
            "No red flags were identified from "
            "the information provided."
        )

    # ---------------------------------------------------------
    # Missing information
    # ---------------------------------------------------------

    missing_information = list(
        state.missing_information
    )

    # ---------------------------------------------------------
    # Possible condition
    # ---------------------------------------------------------

    possible_conditions = []

    if state.possible_conditions:

        condition = state.possible_conditions[0]

        condition_name = condition.get(
            "condition",
            "Unspecified",
        )

        possible_conditions.append(
            condition_name
        )

    # ---------------------------------------------------------
    # Evidence summary
    # ---------------------------------------------------------

    evidence_parts = []

    for item in state.evidence:

        evidence_parts.append(
            f"{item.source_organization} — "
            f"{item.title}"
        )

    if evidence_parts:

        evidence_summary = (
            "The retrieved medical evidence was reviewed "
            "for relevance to the patient's documented "
            "presentation. Sources include: "
            + "; ".join(evidence_parts)
            + ". Detailed evidence and source links are "
              "available in Agent 4."
        )

    else:

        evidence_summary = (
            "No approved medical evidence was retrieved."
        )

    # ---------------------------------------------------------
    # Review note
    # ---------------------------------------------------------

    if possible_conditions:

        review_note = (
            "The listed condition is an "
            "evidence-supported possibility only "
            "and is NOT a confirmed diagnosis. "
            "Clinician review is required."
        )

    else:

        review_note = (
            "No evidence-supported condition was "
            "identified. Clinician review is required."
        )

    return ClinicalNote(
        presenting_complaint=(
            state.patient.main_complaint
        ),

        symptom_summary=symptom_summary,

        relevant_history=relevant_history,

        red_flag_summary=red_flag_summary,

        missing_information=missing_information,

        possible_conditions=possible_conditions,

        evidence_summary=evidence_summary,

        clinician_review_note=review_note,
    )


def run_clinical_note_agent(
    state: CaseState,
) -> CaseState:

    # =========================================================
    # Prepare symptoms
    # =========================================================

    if (
        state.symptom_extraction
        and state.symptom_extraction.symptoms
    ):

        symptom_text = "\n".join(
            f"- {symptom.name}: "
            f"duration={symptom.duration}, "
            f"severity={symptom.severity}, "
            f"frequency={symptom.frequency}"
            for symptom in state.symptom_extraction.symptoms
        )

    else:

        symptom_text = "No symptoms were extracted."

    # =========================================================
    # Prepare red flags
    # =========================================================

    if (
        state.red_flag_detection
        and state.red_flag_detection.red_flags
    ):

        red_flag_text = "\n".join(
            f"- {flag.flag}: "
            f"{flag.reason} "
            f"(Urgency: {flag.urgency})"
            for flag in state.red_flag_detection.red_flags
        )

    else:

        red_flag_text = (
            "No red flags were identified."
        )

    # =========================================================
    # Prepare possible condition
    # =========================================================

    if state.possible_conditions:

        condition = state.possible_conditions[0]

        possible_conditions_text = (
            f"Condition: "
            f"{condition.get('condition', 'Unspecified')}\n"
            f"Rationale: "
            f"{condition.get('rationale', '')}\n"
            f"Evidence support: "
            f"{condition.get('confidence', 'Not specified')}\n"
            f"Evidence source IDs: "
            f"{', '.join(condition.get('evidence_source_ids', []))}"
        )

    else:

        possible_conditions_text = (
            "No evidence-supported possible condition "
            "was identified."
        )

    # =========================================================
    # Prepare evidence
    # =========================================================

    if state.evidence:

        evidence_text = "\n\n".join(
            f"Source ID: {item.source_id or 'Unknown'}\n"
            f"Organization: {item.source_organization}\n"
            f"Title: {item.title}\n"
            f"Topic: {item.topic}\n"
            f"Evidence: {item.evidence_text}"
            for item in state.evidence
        )

    else:

        evidence_text = (
            "No approved medical evidence was retrieved."
        )

    # =========================================================
    # LLM prompts
    # =========================================================

    system_prompt = """
You are Agent 6 of a clinical decision-support prototype.

Create a concise clinician-facing note using ONLY information
provided by the workflow.

STRICT RULES:

- Do NOT make a diagnosis.
- Do NOT confirm a condition.
- Do NOT recommend treatment or medication.
- Do NOT invent information.
- Do NOT invent symptoms, tests, findings, sources, or citations.
- Preserve uncertainty.
- The clinician remains the final decision-maker.

IMPORTANT EVIDENCE DISPLAY RULES:

The evidence_summary must be SHORT and clinician-friendly.

Do NOT include:
- URLs
- hyperlinks
- markdown links
- full article text
- long evidence quotations
- long source identifiers
- repeated source content

Instead, provide:
1. A brief synthesis of what the retrieved evidence supports
   in relation to the documented presentation.
2. A short source list using only:
   Organization — Article Title

For example:

"The retrieved evidence supports an acute upper-respiratory
presentation with overlap between the documented symptoms
and common respiratory conditions.

Sources:
CDC — Sore Throat Basics
NIH — The Diagnosis and Treatment of Acute Cough in Adults
NHS — Respiratory Tract Infections"

Detailed evidence and clickable source links are available
separately in Agent 4.

If Agent 5 supplied a most evidence-supported possibility,
place its EXACT condition name into possible_conditions.

If Agent 5 supplied one condition, possible_conditions MUST
contain exactly one item.

Do not leave possible_conditions empty when Agent 5 supplied
a condition.

Return only the requested structured JSON.
"""

    user_prompt = f"""
PATIENT INFORMATION

Main complaint:
{state.patient.main_complaint}

Age:
{state.patient.age}

Sex:
{state.patient.sex}

Duration:
{state.patient.duration}

Onset:
{state.patient.onset}

Progression:
{state.patient.progression}

Severity:
{state.patient.severity}

Temperature:
{state.patient.temperature}

Heart rate:
{state.patient.heart_rate}

Blood pressure:
{state.patient.blood_pressure}

Oxygen saturation:
{state.patient.oxygen_saturation}

Medical history:
{state.patient.medical_history}

Medications:
{state.patient.medications}

Allergies:
{state.patient.allergies}


EXTRACTED SYMPTOMS

{symptom_text}


RED FLAGS

{red_flag_text}


MISSING INFORMATION

{
    state.missing_information
    if state.missing_information
    else "None identified."
}


MOST EVIDENCE-SUPPORTED POSSIBILITY

{possible_conditions_text}


APPROVED MEDICAL EVIDENCE

{evidence_text}


Create the clinician-facing note.

The JSON must contain:

presenting_complaint
symptom_summary
relevant_history
red_flag_summary
missing_information
possible_conditions
evidence_summary
clinician_review_note

IMPORTANT:

The evidence_summary must be concise.

Do NOT include URLs or hyperlinks.

Do NOT copy full evidence passages.

Do NOT include article links.

Mention sources only as:
Organization — Article Title

The possible condition is NOT a confirmed diagnosis.
Clinician review is required.
"""

    # =========================================================
    # ONE LLM REQUEST
    # =========================================================

    try:

        result = call_structured_llm(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            response_model=ClinicalNote,
            temperature=0.0,
            max_retries=1,
        )

        state.clinical_note = result.model_dump_json(
            indent=2
        )

        print(
            "Agent 6 generated clinician note using LLM."
        )

    except Exception as error:

        print(
            "Agent 6 LLM unavailable."
        )

        print(
            f"Agent 6 fallback reason: {error}"
        )

        fallback = _build_fallback_note(state)

        state.clinical_note = fallback.model_dump_json(
            indent=2
        )

        print(
            "Agent 6 generated clinician note "
            "using deterministic fallback."
        )

    state.next_action = "CLINICIAN_REVIEW"

    return state