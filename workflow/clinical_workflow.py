"""
workflow/clinical_workflow.py

Main workflow connecting the clinical decision-support agents
through the shared CaseState and adaptive router.
"""

from models.schemas import PatientInput

from agents.symptom_agent import run_symptom_agent
from agents.red_flag_agent import run_red_flag_agent
from agents.missing_info_agent import run_missing_information_agent
from agents.differential_agent import run_differential_agent
from agents.clinical_note_agent import run_clinical_note_agent

from knowledge_base.evidence_agent import run_evidence_agent

from workflow.case_state import CaseState
from workflow.router import choose_next_action


def run_initial_workflow(patient: PatientInput) -> CaseState:
    """
    Run the adaptive clinical decision-support workflow.

    Flow:

    Patient Input
        ↓
    Agent 1 — Symptom Extraction
        ↓
    Agent 2 — Red-Flag Detection
        ↓
    Agent 3 — Missing Information
        ↓
    Agent 4 — Evidence Retrieval
        ↓
    Agent 5 — Differential Analysis
        ↓
    Agent 6 — Clinical Note
        ↓
    Clinician Review
    """

    # ---------------------------------------------------------
    # Create shared case state
    # ---------------------------------------------------------

    state = CaseState(patient=patient)

    # ---------------------------------------------------------
    # Agent 1 — Symptom Extraction
    # ---------------------------------------------------------

    state.symptom_extraction = run_symptom_agent(patient)

    # ---------------------------------------------------------
    # Agent 2 — Red-Flag Detection
    # ---------------------------------------------------------

    state.red_flag_detection = run_red_flag_agent(
        state.symptom_extraction
    )

    # ---------------------------------------------------------
    # Agent 3 — Missing Information
    # ---------------------------------------------------------

    state = run_missing_information_agent(state)

    # ---------------------------------------------------------
    # Adaptive routing
    # ---------------------------------------------------------

    state.next_action = choose_next_action(state)

    # ---------------------------------------------------------
    # Red-flag branch
    # ---------------------------------------------------------

    if state.next_action == "CLINICIAN_REVIEW":
        return state

    # ---------------------------------------------------------
    # Agent 4 — Evidence Retrieval
    # ---------------------------------------------------------

    # Missing information is advisory and does not block
    # evidence retrieval.
    state.next_action = "RETRIEVE_EVIDENCE"

    state = run_evidence_agent(state)

    # ---------------------------------------------------------
    # Agent 5 — Differential Analysis
    # ---------------------------------------------------------

    if state.evidence:
        state = run_differential_agent(state)
    else:
        state.possible_conditions = []
        state.next_action = "CLINICIAN_REVIEW"

    # ---------------------------------------------------------
    # Agent 6 — Clinical Note Generation
    # ---------------------------------------------------------

    state = run_clinical_note_agent(state)

    return state