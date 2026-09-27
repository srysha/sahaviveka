"""
knowledge_base/evidence_agent.py

Agent 4 — Medical Evidence Retrieval

Retrieves evidence relevant to the patient's current
clinical presentation.

This agent retrieves evidence only.
It does not diagnose the patient.
"""

from workflow.case_state import CaseState
from knowledge_base.evidence_store import retrieve_evidence


def run_evidence_agent(
    state: CaseState,
) -> CaseState:
    """
    Agent 4: Retrieve relevant medical evidence from
    approved medical sources.
    """

    if state.symptom_extraction is None:
        state.evidence = []
        state.next_action = "CLINICIAN_REVIEW"
        return state

    symptoms = [
        symptom.name
        for symptom in state.symptom_extraction.symptoms
        if symptom.name
    ]

    # Remove duplicate symptom names while preserving order.
    symptoms = list(
        dict.fromkeys(symptoms)
    )

    if not symptoms:
        state.evidence = []
        state.next_action = "CLINICIAN_REVIEW"
        return state

    print("\n==============================")
    print("AGENT 4: EVIDENCE RETRIEVAL")
    print("==============================")

    print("SYMPTOMS USED FOR SEARCH:")

    for symptom in symptoms:
        print(f"- {symptom}")

    evidence = retrieve_evidence(
        symptoms
    )

    state.evidence = evidence

    if evidence:
        state.next_action = "ANALYZE_EVIDENCE"
    else:
        state.next_action = "CLINICIAN_REVIEW"

    print(
        f"\nHIGH-RELEVANCE EVIDENCE ITEMS: "
        f"{len(evidence)}"
    )

    for item in evidence:
        print(
            f"\nSOURCE: "
            f"{item.source_organization}"
        )

        print(
            f"TITLE: {item.title}"
        )

        print(
            f"URL: {item.source_url}"
        )

    return state