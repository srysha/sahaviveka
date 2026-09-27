"""
workflow/case_state.py

Shared case state for the adaptive clinical decision-support workflow.

This stores what the system knows about a case and what still needs to
happen next. It does not make a diagnosis.
"""

from typing import List, Optional
from pydantic import BaseModel, Field

from models.schemas import (
    PatientInput,
    SymptomExtraction,
    RedFlagDetection,
    EvidenceItem,

)


class CaseState(BaseModel):
    """
    The shared state passed between agents.

    Agents add information to this state rather than working as
    completely separate, disconnected steps.
    """

    # Original patient information
    patient: PatientInput

    # Information extracted by Agent 1
    symptom_extraction: Optional[SymptomExtraction] = None

    # Red flags identified by Agent 2
    red_flag_detection: Optional[RedFlagDetection] = None

    # Information that may still be needed
    missing_information: List[str] = Field(default_factory=list)

    # What the router decides should happen next
    next_action: Optional[str] = None

    # Evidence retrieved later from approved medical sources
    evidence: List[EvidenceItem] = Field(default_factory=list)

    # Possible conditions considered later
    possible_conditions: List[dict] = Field(default_factory=list)

    # Final clinician-facing note, generated later
    clinical_note: Optional[str] = None