"""
app.py

Streamlit interface for the full multi-agent clinical
decision-support prototype.
"""

import json

import streamlit as st

from models.schemas import PatientInput
from utils.llm import (
    LLMConfigError,
    LLMRequestError,
    LLMOutputError,
)
from utils.logging_utils import setup_logging

from workflow.clinical_workflow import run_initial_workflow
from workflow.clinician_review import build_clinician_review

from agents.caregiver_agent import run_caregiver_agent


setup_logging()

st.set_page_config(
    page_title="AI Clinical Decision-Support Prototype",
    page_icon="🩺",
    layout="centered",
)


# ---------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------

st.title("🩺 AI Clinical Decision-Support Prototype")

st.caption(
    "Multi-agent research prototype — built for a college hackathon."
)

st.warning(
    "**AI-generated decision support for research/educational use only.** "
    "This system does not provide a definitive diagnosis or replace a "
    "qualified healthcare professional. All outputs require clinician "
    "review. Use simulated / de-identified patient data only."
)


# ---------------------------------------------------------------------
# Workflow
# ---------------------------------------------------------------------

with st.expander(
    "🔎 Agent Workflow",
    expanded=False,
):
    st.markdown(
        """
1. **Patient Input** ✅
2. **Agent 1 — Symptom Extraction** ✅
3. **Agent 2 — Red-Flag Detection** ✅
4. **Agent 3 — Missing Information** ✅
5. **Agent 4 — Medical Evidence Retrieval** ✅
6. **Agent 5 — Differential Analysis** ✅
7. **Agent 6 — Clinical Note Generation** ✅
8. **Caregiver Mode** ✅
9. **Clinician Review** — final human decision
        """
    )


# ---------------------------------------------------------------------
# Patient input form
# ---------------------------------------------------------------------

st.subheader("Patient Information")

st.caption(
    "Leave anything blank if unknown — the system will record it as "
    "'Not provided', never guess."
)


with st.form("patient_form"):

    col1, col2 = st.columns(2)

    with col1:

        age = st.text_input(
            "Age",
            placeholder="e.g. 34",
        )

        sex = st.selectbox(
            "Sex",
            [
                "Not provided",
                "Female",
                "Male",
                "Intersex",
                "Other",
            ],
        )

        duration = st.text_input(
            "Duration of main issue",
            placeholder="e.g. 3 days",
        )

        severity = st.select_slider(
            "Symptom severity (self-reported)",
            options=[
                "Not provided",
                "Mild",
                "Moderate",
                "Severe",
            ],
            value="Not provided",
        )

    with col2:

        onset = st.selectbox(
            "Onset",
            [
                "Not provided",
                "Sudden",
                "Gradual",
                "Comes and goes",
            ],
        )

        progression = st.selectbox(
            "Progression",
            [
                "Not provided",
                "Improving",
                "Worsening",
                "Stable",
            ],
        )

        temperature = st.text_input(
            "Temperature (°C/°F)",
            placeholder="e.g. 38.5°C",
        )

        heart_rate = st.text_input(
            "Heart rate (bpm)",
            placeholder="e.g. 98",
        )

    col3, col4 = st.columns(2)

    with col3:

        blood_pressure = st.text_input(
            "Blood pressure",
            placeholder="e.g. 120/80",
        )

    with col4:

        oxygen_saturation = st.text_input(
            "Oxygen saturation (%)",
            placeholder="e.g. 97",
        )

    main_complaint = st.text_input(
        "Main complaint (short)",
        placeholder="e.g. Fever and headache",
    )

    symptom_description = st.text_area(
        "Describe symptoms in your own words",
        placeholder=(
            "e.g. I have had fever for three days and a severe "
            "headache since yesterday. I also vomited twice."
        ),
        height=110,
    )

    medical_history = st.text_area(
        "Relevant medical history",
        placeholder="e.g. Type 2 diabetes, or 'none'",
    )

    medications = st.text_area(
        "Current medications",
        placeholder="e.g. Metformin 500mg, or 'none'",
    )

    allergies = st.text_area(
        "Allergies",
        placeholder="e.g. Penicillin, or 'none known'",
    )

    other_info = st.text_area(
        "Other relevant information (optional)"
    )

    submitted = st.form_submit_button(
        "▶ ANALYZE CASE",
        use_container_width=True,
    )


# ---------------------------------------------------------------------
# Run complete workflow
# ---------------------------------------------------------------------

if submitted:

    if (
        not main_complaint.strip()
        and not symptom_description.strip()
    ):

        st.error(
            "Please enter at least a main complaint or a "
            "symptom description before analyzing."
        )

    else:

        patient_input = PatientInput(

            age=age or "Not provided",

            sex=sex,

            main_complaint=main_complaint,

            symptom_description=symptom_description,

            duration=duration or "Not provided",

            severity=severity,

            onset=onset,

            progression=progression,

            medical_history=(
                medical_history or "Not provided"
            ),

            medications=(
                medications or "Not provided"
            ),

            allergies=(
                allergies or "Not provided"
            ),

            temperature=(
                temperature or "Not provided"
            ),

            heart_rate=(
                heart_rate or "Not provided"
            ),

            blood_pressure=(
                blood_pressure or "Not provided"
            ),

            oxygen_saturation=(
                oxygen_saturation or "Not provided"
            ),

            other_info=(
                other_info or "Not provided"
            ),
        )

        st.divider()

        st.subheader("🧠 Multi-Agent Analysis")

        try:

            with st.spinner(
                "Running the clinical decision-support workflow..."
            ):

                case_state = run_initial_workflow(
                    patient_input
                )

            # Save the case so caregiver mode can use it
            # after a Streamlit rerun.
            st.session_state["case_state"] = case_state

        except LLMConfigError as e:

            st.error(
                f"⚙️ Setup issue: {e}"
            )

            st.stop()

        except LLMRequestError as e:

            st.error(
                f"📡 Could not reach the AI service: {e}"
            )

            st.stop()

        except LLMOutputError as e:

            st.error(
                f"🤖 The AI returned unusable structured data: {e}"
            )

            st.stop()

        except Exception as e:

            st.error(
                f"❌ Unexpected workflow error: "
                f"{type(e).__name__}: {e}"
            )

            st.stop()


# ---------------------------------------------------------------------
# Display saved case
# ---------------------------------------------------------------------

if "case_state" in st.session_state:

    state = st.session_state["case_state"]

    st.divider()

    st.subheader("📊 Analysis Results")


    # -------------------------------------------------------------
    # Agent 1 — Symptoms
    # -------------------------------------------------------------

    with st.container(border=True):

        st.markdown(
            "### 🧾 Agent 1 — Extracted Symptoms"
        )

        if (
            state.symptom_extraction
            and state.symptom_extraction.symptoms
        ):

            for symptom in (
                state.symptom_extraction.symptoms
            ):

                st.markdown(
                    f"- **{symptom.name}** — "
                    f"duration: {symptom.duration}, "
                    f"severity: {symptom.severity}, "
                    f"frequency: {symptom.frequency}"
                )

        else:

            st.markdown(
                "_No symptoms could be extracted._"
            )


    # -------------------------------------------------------------
    # Agent 2 — Red Flags
    # -------------------------------------------------------------

    with st.container(border=True):

        st.markdown(
            "### 🚨 Agent 2 — Red-Flag Detection"
        )

        if (
            state.red_flag_detection
            and state.red_flag_detection.red_flags
        ):

            for flag in (
                state.red_flag_detection.red_flags
            ):

                st.markdown(
                    f"**{flag.flag}**  \n"
                    f"Reason: {flag.reason}  \n"
                    f"Urgency: **{flag.urgency}**"
                )

        else:

            st.success(
                "No red flags were identified from "
                "the information provided."
            )


    # -------------------------------------------------------------
    # Agent 3 — Missing Information
    # -------------------------------------------------------------

    with st.container(border=True):

        st.markdown(
            "### 🔎 Agent 3 — Missing Information"
        )

        if state.missing_information:

            st.info(
                "The following information may be useful "
                "for further clinical assessment:"
            )

            for item in state.missing_information:

                st.markdown(
                    f"- {item}"
                )

        else:

            st.success(
                "No additional missing information "
                "was identified."
            )


    # -------------------------------------------------------------
    # Agent 4 — Evidence
    # -------------------------------------------------------------

    with st.container(border=True):

        st.markdown(
            "### 📚 Agent 4 — Approved Medical Evidence"
        )

        if state.evidence:

            for item in state.evidence:

                st.markdown(
                    f"**{item.title}**"
                )

                st.caption(
                    f"{item.source_organization} "
                    f"({item.source_name})"
                )

                if item.publication_date:

                    st.caption(
                        f"Publication date: "
                        f"{item.publication_date}"
                    )

                # -------------------------------------------------
                # Display-only evidence preview.
                #
                # IMPORTANT:
                # This does NOT modify the evidence retrieved by
                # Agent 4. It only prevents very long source text
                # from filling the Streamlit interface.
                # -------------------------------------------------

                evidence_preview = (
                    item.evidence_text.strip()
                )

                if len(evidence_preview) > 350:

                    evidence_preview = (
                        evidence_preview[:350]
                        .rsplit(" ", 1)[0]
                        + "..."
                    )

                st.markdown(
                    f"**Evidence:** {evidence_preview}"
                )

                if item.source_url:

                    st.markdown(
                        f"[Open source]({item.source_url})"
                    )

                st.divider()

        else:

            st.info(
                "No approved evidence was retrieved."
            )


    # -------------------------------------------------------------
    # Agent 5 — Differential
    # -------------------------------------------------------------

    with st.container(border=True):

        st.markdown(
            "### 🩺 Agent 5 — Possible Conditions"
        )

        if state.possible_conditions:

            st.caption(
                "These are possibilities for clinician "
                "review, not confirmed diagnoses."
            )

            for condition in (
                state.possible_conditions
            ):

                st.markdown(
                    f"**{condition.get('condition', 'Unspecified')}**"
                )

                st.markdown(
                    f"Rationale: "
                    f"{condition.get('rationale', '')}"
                )

                source_ids = condition.get(
                    "evidence_source_ids",
                    [],
                )

                if source_ids:

                    st.caption(
                        "Evidence source ID(s): "
                        + ", ".join(source_ids)
                    )

                st.caption(
                    "Confidence: "
                    + condition.get(
                        "confidence",
                        "Uncertain",
                    )
                )

                st.divider()

        else:

            st.info(
                "No evidence-supported possible "
                "conditions were identified."
            )


    # -------------------------------------------------------------
    # Agent 6 — Clinical Note
    # -------------------------------------------------------------

    with st.container(border=True):

        st.markdown(
            "### 📋 Agent 6 — Clinician-Facing Note"
        )

        if state.clinical_note:

            try:

                note = json.loads(
                    state.clinical_note
                )

                presenting_complaint = (
                    note.get(
                        "presenting_complaint",
                        "Not provided",
                    )
                )

                # Display-only formatting fix.
                # Does not modify the actual stored note.
                presenting_complaint = (
                    presenting_complaint
                    .replace(",headache", ", headache")
                )

                st.markdown(
                    f"**Presenting complaint**  \n"
                    f"{presenting_complaint}"
                )

                st.markdown(
                    f"**Symptom summary**  \n"
                    f"{note.get('symptom_summary', 'Not provided')}"
                )

                st.markdown(
                    f"**Relevant history**  \n"
                    f"{note.get('relevant_history', 'Not provided')}"
                )

                st.markdown(
                    f"**Red-flag summary**  \n"
                    f"{note.get('red_flag_summary', 'Not provided')}"
                )

                # -------------------------------------------------
                # Display-only evidence summary.
                #
                # The original generated note remains unchanged.
                # We only shorten what is shown in the UI.
                # -------------------------------------------------

                evidence_summary = note.get(
                    "evidence_summary",
                    "Not provided",
                )

                if len(evidence_summary) > 700:

                    evidence_summary = (
                        evidence_summary[:700]
                        .rsplit(" ", 1)[0]
                        + "..."
                    )

                st.markdown(
                    f"**Evidence summary**  \n"
                    f"{evidence_summary}"
                )

                st.markdown(
                    f"**Clinician review note**  \n"
                    f"{note.get('clinician_review_note', 'Not provided')}"
                )

            except json.JSONDecodeError:

                st.markdown(
                    state.clinical_note
                )


    # -------------------------------------------------------------
    # Caregiver Mode
    # -------------------------------------------------------------

    st.divider()

    st.subheader("👨‍👩‍👧 Caregiver Mode")

    st.caption(
        "Creates a simpler explanation of the information "
        "already produced by the system."
    )

    if st.button(
        "Generate Caregiver Explanation",
        use_container_width=True,
    ):

        try:

            with st.spinner(
                "Generating caregiver explanation..."
            ):

                caregiver_result = (
                    run_caregiver_agent(state)
                )

            st.markdown(
                f"**Summary**  \n"
                f"{caregiver_result.summary}"
            )

            if caregiver_result.important_points:

                st.markdown(
                    "**Important points:**"
                )

                for point in (
                    caregiver_result.important_points
                ):

                    st.markdown(
                        f"- {point}"
                    )

            if caregiver_result.questions_for_clinician:

                st.markdown(
                    "**Questions for the clinician:**"
                )

                for question in (
                    caregiver_result.questions_for_clinician
                ):

                    st.markdown(
                        f"- {question}"
                    )

            st.info(
                caregiver_result.safety_note
            )

        except LLMConfigError as e:

            st.error(
                f"⚙️ Setup issue: {e}"
            )

        except LLMRequestError as e:

            st.error(
                f"📡 Could not reach the AI service: {e}"
            )

        except LLMOutputError as e:

            st.error(
                f"🤖 Caregiver explanation was unusable: {e}"
            )


    # -------------------------------------------------------------
    # Clinician Review
    # -------------------------------------------------------------

    st.divider()

    st.subheader("👨‍⚕️ Clinician Review")

    st.warning(
        "**Human review required.** "
        "The AI output is decision support only. "
        "A qualified healthcare professional must make "
        "the final clinical assessment and decision."
    )

    review = build_clinician_review(state)

    with st.expander(
        "View structured clinician-review package"
    ):

        st.json(review)