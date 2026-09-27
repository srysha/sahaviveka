"""
models/schemas.py

These classes define the exact "shape" of data moving through the system.
Think of each one as a form with fixed fields - the AI has to fill in
that exact form, it can't invent new fields or skip the structure.

Today we only need the schema for Agent 1 (Symptom Extraction).
We will add one schema per agent as we build each one, and a
PatientCase schema later to hold the whole shared state.
"""

from typing import List, Optional
from pydantic import BaseModel, Field


# ---------------------------------------------------------------------
# Raw patient input - what the clinician/user types into the form
# ---------------------------------------------------------------------

class PatientInput(BaseModel):
    age: Optional[str] = "Not provided"
    sex: Optional[str] = "Not provided"
    main_complaint: str = ""
    symptom_description: str = ""  # free-text narrative, e.g. "fever for 3 days..."
    duration: Optional[str] = "Not provided"
    severity: Optional[str] = "Not provided"
    onset: Optional[str] = "Not provided"
    progression: Optional[str] = "Not provided"
    medical_history: Optional[str] = "Not provided"
    medications: Optional[str] = "Not provided"
    allergies: Optional[str] = "Not provided"
    temperature: Optional[str] = "Not provided"
    heart_rate: Optional[str] = "Not provided"
    blood_pressure: Optional[str] = "Not provided"
    oxygen_saturation: Optional[str] = "Not provided"
    other_info: Optional[str] = "Not provided"


# ---------------------------------------------------------------------
# Agent 1 output - Symptom Extraction
# ---------------------------------------------------------------------

class ExtractedSymptom(BaseModel):
    name: str
    duration: Optional[str] = "Not provided"
    severity: Optional[str] = "Not provided"
    frequency: Optional[str] = "Not provided"
    notes: Optional[str] = "Not provided"


class SymptomExtraction(BaseModel):
    symptoms: List[ExtractedSymptom] = Field(default_factory=list)
    medical_history: List[str] = Field(default_factory=list)
    medications: List[str] = Field(default_factory=list)
    allergies: List[str] = Field(default_factory=list)
    extraction_notes: Optional[str] = Field(
        default="",
        description="Anything the agent explicitly could NOT determine from the input."
    )

    # ---------------------------------------------------------------------
# Agent 2 output - Red-Flag Detection
# ---------------------------------------------------------------------

class RedFlag(BaseModel):
    flag: str
    reason: str
    urgency: str = "Needs review"


class RedFlagDetection(BaseModel):
    red_flags: List[RedFlag] = Field(default_factory=list)
    no_red_flags_identified: bool = False
    assessment_notes: Optional[str] = ""

class EvidenceItem(BaseModel):
    """
    A piece of evidence retrieved from an approved medical source.
    """

    source_organization: str
    source_name: str
    source_url: str

    title: str

    publication_date: Optional[str] = None

    topic: str

    evidence_text: str

    relevance: str = "Relevant"

    source_id: Optional[str] = None
