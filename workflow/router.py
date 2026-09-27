"""
workflow/router.py

Adaptive Router for the clinical decision-support prototype.

The router decides what the system should do next based on the
information currently available in CaseState.

It does NOT diagnose the patient.
"""

from models.schemas import RedFlagDetection
from workflow.case_state import CaseState


# Possible actions the router can choose.
CHECK_FOR_MISSING_INFORMATION = "CHECK_FOR_MISSING_INFORMATION"
REVIEW_RED_FLAGS = "REVIEW_RED_FLAGS"
RETRIEVE_EVIDENCE = "RETRIEVE_EVIDENCE"
CLINICIAN_REVIEW = "CLINICIAN_REVIEW"


def choose_next_action(state: CaseState) -> str:
    """
    Decide the next workflow step from the current case state.

    This is intentionally rule-based for now. Later, we can make the
    routing more sophisticated while keeping the final decision under
    clinician oversight.
    """

    # If Agent 1 has not run yet, we need symptom extraction first.
    if state.symptom_extraction is None:
        return "RUN_SYMPTOM_EXTRACTION"

    # If Agent 2 has not run yet, check for red flags.
    if state.red_flag_detection is None:
        return "RUN_RED_FLAG_DETECTION"

    # If red flags were identified, prioritize clinician review.
    if state.red_flag_detection.red_flags:
        return CLINICIAN_REVIEW

    # If important information is missing, gather it before
    # attempting evidence-based analysis.
    if state.missing_information:
        return CHECK_FOR_MISSING_INFORMATION

    # Otherwise, retrieve evidence from the approved knowledge base.
    if not state.evidence:
        return RETRIEVE_EVIDENCE

    # If evidence exists, the next stage can perform analysis.
    return "ANALYZE_EVIDENCE"