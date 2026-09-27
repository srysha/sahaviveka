"""
workflow/clinician_review.py

Creates a structured clinician-review package from the current
clinical decision-support case.

The AI does not make the final clinical decision.
"""

from workflow.case_state import CaseState


def build_clinician_review(state: CaseState) -> dict:
    """
    Build a structured summary for clinician review.
    """

    review = {
        "patient": state.patient.model_dump(),

        "symptoms": (
            state.symptom_extraction.model_dump()
            if state.symptom_extraction
            else None
        ),

        "red_flags": (
            state.red_flag_detection.model_dump()
            if state.red_flag_detection
            else None
        ),

        "missing_information": state.missing_information,

        "possible_conditions": state.possible_conditions,

        "evidence": [
            {
                "source_id": item.source_id,
                "organization": item.source_organization,
                "source": item.source_name,
                "title": item.title,
                "url": item.source_url,
                "publication_date": item.publication_date,
                "evidence": item.evidence_text,
            }
            for item in state.evidence
        ],

        "next_action": state.next_action,

        "clinical_decision_required": True,
    }

    return review