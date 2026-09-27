from models.schemas import PatientInput
from agents.symptom_agent import run_symptom_agent


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


print("========================================")
print("TESTING AGENT 1 - SYMPTOM EXTRACTION")
print("========================================")

result = run_symptom_agent(patient)

print("\nExtracted symptoms:")

for symptom in result.symptoms:
    print(
        f"- {symptom.name} | "
        f"duration: {symptom.duration} | "
        f"severity: {symptom.severity}"
    )

print("\nMedical history:", result.medical_history)
print("Medications:", result.medications)
print("Allergies:", result.allergies)

print("\n========================================")
print("TEST COMPLETE")
print("========================================")