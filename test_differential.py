from models.schemas import PatientInput
from agents.symptom_agent import run_symptom_agent
from knowledge_base.evidence_store import retrieve_evidence
from workflow.case_state import CaseState
from agents.differential_agent import run_differential_agent


print("========================================")
print("TESTING AGENT 4 + AGENT 5")
print("========================================")


# ------------------------------------------------------------
# 1. Create test patient
# ------------------------------------------------------------

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


# ------------------------------------------------------------
# 2. Run Agent 1
# ------------------------------------------------------------

print("\n===== RUNNING AGENT 1 =====")

symptom_extraction = run_symptom_agent(patient)

print("\nExtracted symptoms:")

for symptom in symptom_extraction.symptoms:
    print(
        f"- {symptom.name} | "
        f"duration: {symptom.duration} | "
        f"severity: {symptom.severity}"
    )


# ------------------------------------------------------------
# 3. Get symptom names for evidence retrieval
# ------------------------------------------------------------

symptom_names = [
    symptom.name
    for symptom in symptom_extraction.symptoms
]


# ------------------------------------------------------------
# 4. Run Agent 4
# ------------------------------------------------------------

print("\n========================================")
print("RUNNING AGENT 4 - EVIDENCE RETRIEVAL")
print("========================================")

evidence = retrieve_evidence(symptom_names)

print(f"\nEvidence items found: {len(evidence)}")

for i, item in enumerate(evidence, start=1):

    print(f"\n--- SOURCE {i} ---")
    print("Organization:", item.source_organization)
    print("Title:", item.title)
    print("URL:", item.source_url)
    print("Evidence:", item.evidence_text[:500])


# ------------------------------------------------------------
# 5. Create workflow state
# ------------------------------------------------------------

state = CaseState(
    patient=patient,
    symptom_extraction=symptom_extraction,
    evidence=evidence,
)


# ------------------------------------------------------------
# 6. Run Agent 5
# ------------------------------------------------------------

print("\n========================================")
print("RUNNING AGENT 5 - DIFFERENTIAL ANALYSIS")
print("========================================")

state = run_differential_agent(state)


# ------------------------------------------------------------
# 7. Display possible conditions
# ------------------------------------------------------------

print("\nPossible conditions:")

if not state.possible_conditions:

    print("No evidence-supported conditions returned.")

else:

    for i, condition in enumerate(
        state.possible_conditions,
        start=1
    ):

        print(f"\n--- CONDITION {i} ---")
        print("Condition:", condition["condition"])
        print("Rationale:", condition["rationale"])
        print(
            "Evidence source IDs:",
            condition["evidence_source_ids"]
        )
        print(
            "Evidence support:",
            condition["confidence"]
        )


print("\n========================================")
print("TEST COMPLETE")
print("========================================")