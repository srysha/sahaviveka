from models.schemas import (
    PatientInput,
    RedFlagDetection,
    EvidenceItem,
)

from workflow.case_state import CaseState
from agents.clinical_note_agent import run_clinical_note_agent
from agents.symptom_agent import run_symptom_agent


print("========================================")
print("TESTING AGENT 6 - CLINICAL NOTE")
print("========================================")


# ---------------------------------------------------------
# SAMPLE PATIENT
# ---------------------------------------------------------

patient = PatientInput(
    age="20",
    sex="Female",
    main_complaint="Cold-like respiratory symptoms",
    symptom_description=(
        "The patient has a blocked nose, sore throat, dry cough, "
        "and headache."
    ),
    duration="3 days",
    severity="moderate",
)


# ---------------------------------------------------------
# RUN AGENT 1
# ---------------------------------------------------------

print("\n===== PREPARING AGENT 1 OUTPUT =====")

symptom_extraction = run_symptom_agent(patient)

print("\nSymptoms prepared:")

for symptom in symptom_extraction.symptoms:
    print(
        f"- {symptom.name} | "
        f"duration: {symptom.duration} | "
        f"severity: {symptom.severity}"
    )


# ---------------------------------------------------------
# USE ALREADY-TESTED AGENT 5 RESULT
# ---------------------------------------------------------
# We do NOT run Agents 2-5 again.
# This saves OpenRouter requests.

possible_conditions = [
    {
        "condition": (
            "Common cold (non-specific upper respiratory "
            "tract infection)"
        ),
        "rationale": (
            "The patient's symptoms of blocked nose, sore throat, "
            "dry cough, and headache are all listed as prominent "
            "cold symptoms in the CDC evidence."
        ),
        "evidence_source_ids": [
            "TAVILY_1"
        ],
        "confidence": "high",
    }
]


# ---------------------------------------------------------
# USE ALREADY-TESTED AGENT 4 EVIDENCE
# ---------------------------------------------------------

evidence = [
    EvidenceItem(
        source_id="TAVILY_1",
        source_organization=(
            "Centers for Disease Control and Prevention"
        ),
        source_name="CDC",
        title=(
            "Outpatient Clinical Care for Adults | "
            "Antibiotic Prescribing"
        ),
        source_url=(
            "https://www.cdc.gov/antibiotic-use/hcp/"
            "clinical-care/adult-outpatient.html"
        ),
        publication_date=None,
        topic=(
            "Common cold / upper respiratory tract infection"
        ),
        evidence_text=(
            "Prominent cold symptoms include fever, cough, "
            "rhinorrhea, nasal congestion, postnasal drip, "
            "sore throat, headache, and myalgias."
        ),
        relevance="Relevant",
    )
]


# ---------------------------------------------------------
# CREATE CASE STATE
# ---------------------------------------------------------

state = CaseState(
    patient=patient,

    symptom_extraction=symptom_extraction,

    red_flag_detection=RedFlagDetection(
        red_flags=[],
        no_red_flags_identified=True,
        assessment_notes=(
            "No red flags identified from the supplied "
            "information."
        ),
    ),

    missing_information=[
        "Temperature",
        "Onset",
        "Progression",
    ],

    possible_conditions=possible_conditions,

    evidence=evidence,
)


# ---------------------------------------------------------
# RUN AGENT 6
# ---------------------------------------------------------

print("\n========================================")
print("RUNNING AGENT 6")
print("========================================")

state = run_clinical_note_agent(state)


# ---------------------------------------------------------
# DISPLAY RESULT
# ---------------------------------------------------------

print("\n========================================")
print("AGENT 6 OUTPUT")
print("========================================")

print(state.clinical_note)


# ---------------------------------------------------------
# FINAL STATUS
# ---------------------------------------------------------

print("\n========================================")
print("NEXT ACTION")
print("========================================")

print(state.next_action)

print("\n========================================")
print("TEST COMPLETE")
print("========================================")