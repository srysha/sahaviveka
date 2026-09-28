"""
app.py

Sahaviveka — Multi-agent clinical decision-support prototype.
Supports: English | हिंदी | मराठी
Modes: Light | Dark | Colorblind-safe
"""

import json
import streamlit as st

from models.schemas import PatientInput
from utils.llm import LLMConfigError, LLMRequestError, LLMOutputError
from utils.logging_utils import setup_logging
from workflow.clinical_workflow import run_initial_workflow
from workflow.clinician_review import build_clinician_review
from agents.caregiver_agent import run_caregiver_agent

setup_logging()

# ─────────────────────────────────────────────────────────────────────
# Page config  (must be first Streamlit call)
# ─────────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Sahaviveka — Clinical AI",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ─────────────────────────────────────────────────────────────────────
# Translations
# ─────────────────────────────────────────────────────────────────────

TRANSLATIONS = {
    "English": {
        "app_name":          "Sahaviveka",
        "app_subtitle":      "Multi-Agent Clinical Decision Support · Research Prototype",
        "disclaimer":        "⚠️ **AI-generated decision support for research/educational use only.** Not a substitute for a qualified healthcare professional. All outputs require clinician review. Use simulated / de-identified data only.",
        "sidebar_agents":    "🤖 Agent Pipeline",
        "sidebar_disclaimer":"Research & educational use only. Not a substitute for professional clinical judgment.",
        "sidebar_case":      "📊 Case Summary",
        "sidebar_symptoms":  "Symptoms",
        "sidebar_flags":     "Red Flags",
        "sidebar_conditions":"Conditions",
        "sidebar_evidence":  "Evidence",
        "section_vitals":    "🫀 Vitals",
        "section_demo":      "👤 Demographics & Symptom Profile",
        "section_narrative": "📋 Clinical Narrative",
        "label_temp":        "🌡️ Temperature (°C/°F)",
        "label_hr":          "💓 Heart Rate (bpm)",
        "label_bp":          "🩸 Blood Pressure",
        "label_o2":          "💨 O₂ Saturation (%)",
        "label_age":         "Age",
        "label_sex":         "Sex",
        "label_duration":    "Duration of issue",
        "label_severity":    "Severity (self-reported)",
        "label_onset":       "Onset",
        "label_progression": "Progression",
        "label_complaint":   "Main complaint (short)",
        "label_symptoms":    "Describe symptoms in your own words",
        "label_history":     "Relevant medical history",
        "label_medications": "Current medications",
        "label_allergies":   "Allergies",
        "label_other":       "Other relevant information (optional)",
        "btn_analyze":       "🔬  ANALYZE CASE",
        "sex_options":       ["Not provided", "Female", "Male", "Intersex", "Other"],
        "severity_options":  ["Not provided", "Mild", "Moderate", "Severe"],
        "onset_options":     ["Not provided", "Sudden", "Gradual", "Comes and goes"],
        "progression_options":["Not provided", "Improving", "Worsening", "Stable"],
        "results_title":     "🧠 Multi-Agent Analysis Results",
        "agent1_title":      "🧾 Agent 1 — Extracted Symptoms",
        "agent2_title":      "🚨 Agent 2 — Red-Flag Detection",
        "agent3_title":      "🔎 Agent 3 — Missing Information",
        "agent4_title":      "📚 Agent 4 — Medical Evidence",
        "agent5_title":      "🩺 Agent 5 — Possible Conditions",
        "agent6_title":      "📋 Agent 6 — Clinician-Facing Note",
        "caregiver_title":   "👨‍👩‍👧 Caregiver Mode",
        "caregiver_caption": "Generates a plain-language summary of findings for caregivers.",
        "btn_caregiver":     "💬  Generate Caregiver Explanation",
        "clinician_title":   "👨‍⚕️ Clinician Review",
        "clinician_warning": "**Human review required.** AI output is decision support only. A qualified healthcare professional must make the final clinical assessment.",
        "expander_review":   "📄 View structured clinician-review package",
        "no_symptoms":       "No symptoms could be extracted.",
        "no_flags":          "✅ No red flags were identified from the information provided.",
        "no_missing":        "✅ No additional missing information was identified.",
        "no_evidence":       "No approved evidence was retrieved.",
        "no_conditions":     "No evidence-supported possible conditions were identified.",
        "conditions_caption":"These are possibilities for clinician review, not confirmed diagnoses.",
        "form_hint":         "Leave anything blank if unknown — recorded as 'Not provided', never guessed.",
        "spinner_main":      "🧠 Running multi-agent clinical workflow…",
        "spinner_caregiver": "Generating caregiver explanation…",
        "success_analysis":  "✅ Analysis complete — results shown below.",
        "err_no_complaint":  "❌ Please enter at least a main complaint or a symptom description.",
        "caregiver_summary": "📝 Summary",
        "caregiver_points":  "⚠️ Important points:",
        "caregiver_questions":"❓ Questions for the clinician:",
        "open_source":       "🔗 Open source",
        "pub_date":          "📅 ",
        "evidence_sources":  "📎 Sources: ",
        "confidence":        "Confidence: ",
        "rationale":         "Rationale: ",
        "duration_lbl":      "Duration: ",
        "severity_lbl":      "Severity: ",
        "frequency_lbl":     "Frequency: ",
        "note_presenting":   "Presenting Complaint",
        "note_symptom":      "Symptom Summary",
        "note_history":      "Relevant History",
        "note_redflag":      "Red-Flag Summary",
        "note_evidence":     "Evidence Summary",
        "note_clinician":    "Clinician Review Note",
        "mode_label":        "🎨 Display Mode",
        "lang_label":        "🌐 Language",
    },
    "हिंदी": {
        "app_name":          "सहविवेक",
        "app_subtitle":      "बहु-एजेंट नैदानिक निर्णय सहायता · शोध प्रोटोटाइप",
        "disclaimer":        "⚠️ **यह केवल शोध/शैक्षिक उपयोग के लिए AI-जनित निर्णय समर्थन है।** यह किसी योग्य स्वास्थ्य पेशेवर का विकल्प नहीं है। सभी आउटपुट के लिए चिकित्सक की समीक्षा आवश्यक है।",
        "sidebar_agents":    "🤖 एजेंट पाइपलाइन",
        "sidebar_disclaimer":"केवल शोध एवं शैक्षिक उपयोग। पेशेवर नैदानिक निर्णय का विकल्प नहीं।",
        "sidebar_case":      "📊 केस सारांश",
        "sidebar_symptoms":  "लक्षण",
        "sidebar_flags":     "रेड फ्लैग",
        "sidebar_conditions":"स्थितियाँ",
        "sidebar_evidence":  "साक्ष्य",
        "section_vitals":    "🫀 जीवन-संकेत",
        "section_demo":      "👤 जनसांख्यिकी और लक्षण प्रोफ़ाइल",
        "section_narrative": "📋 नैदानिक विवरण",
        "label_temp":        "🌡️ तापमान (°C/°F)",
        "label_hr":          "💓 हृदय गति (bpm)",
        "label_bp":          "🩸 रक्तचाप",
        "label_o2":          "💨 ऑक्सीजन संतृप्ति (%)",
        "label_age":         "आयु",
        "label_sex":         "लिंग",
        "label_duration":    "समस्या की अवधि",
        "label_severity":    "गंभीरता (स्व-रिपोर्ट)",
        "label_onset":       "शुरुआत",
        "label_progression": "प्रगति",
        "label_complaint":   "मुख्य शिकायत (संक्षेप)",
        "label_symptoms":    "अपने शब्दों में लक्षण बताएं",
        "label_history":     "प्रासंगिक चिकित्सा इतिहास",
        "label_medications": "वर्तमान दवाएं",
        "label_allergies":   "एलर्जी",
        "label_other":       "अन्य प्रासंगिक जानकारी (वैकल्पिक)",
        "btn_analyze":       "🔬  केस विश्लेषण करें",
        "sex_options":       ["जानकारी नहीं", "महिला", "पुरुष", "इंटरसेक्स", "अन्य"],
        "severity_options":  ["जानकारी नहीं", "हल्का", "मध्यम", "गंभीर"],
        "onset_options":     ["जानकारी नहीं", "अचानक", "धीरे-धीरे", "आता-जाता"],
        "progression_options":["जानकारी नहीं", "सुधर रहा है", "बिगड़ रहा है", "स्थिर"],
        "results_title":     "🧠 बहु-एजेंट विश्लेषण परिणाम",
        "agent1_title":      "🧾 एजेंट 1 — निकाले गए लक्षण",
        "agent2_title":      "🚨 एजेंट 2 — रेड-फ्लैग पहचान",
        "agent3_title":      "🔎 एजेंट 3 — अनुपलब्ध जानकारी",
        "agent4_title":      "📚 एजेंट 4 — चिकित्सा साक्ष्य",
        "agent5_title":      "🩺 एजेंट 5 — संभावित स्थितियाँ",
        "agent6_title":      "📋 एजेंट 6 — चिकित्सक-केंद्रित नोट",
        "caregiver_title":   "👨‍👩‍👧 देखभालकर्ता मोड",
        "caregiver_caption": "निष्कर्षों का सरल भाषा में सारांश देखभालकर्ताओं के लिए।",
        "btn_caregiver":     "💬  देखभालकर्ता व्याख्या उत्पन्न करें",
        "clinician_title":   "👨‍⚕️ चिकित्सक समीक्षा",
        "clinician_warning": "**मानव समीक्षा आवश्यक है।** AI आउटपुट केवल निर्णय समर्थन है। अंतिम नैदानिक मूल्यांकन योग्य स्वास्थ्य पेशेवर द्वारा किया जाना चाहिए।",
        "expander_review":   "📄 संरचित चिकित्सक-समीक्षा पैकेज देखें",
        "no_symptoms":       "कोई लक्षण नहीं निकाला जा सका।",
        "no_flags":          "✅ दी गई जानकारी से कोई रेड फ्लैग नहीं मिला।",
        "no_missing":        "✅ कोई अतिरिक्त जानकारी अनुपलब्ध नहीं है।",
        "no_evidence":       "कोई अनुमोदित साक्ष्य प्राप्त नहीं हुआ।",
        "no_conditions":     "कोई साक्ष्य-समर्थित संभावित स्थिति नहीं मिली।",
        "conditions_caption":"ये चिकित्सक समीक्षा के लिए संभावनाएं हैं, निश्चित निदान नहीं।",
        "form_hint":         "कुछ भी खाली छोड़ें अगर पता न हो — 'जानकारी नहीं' दर्ज होगा, अनुमान नहीं।",
        "spinner_main":      "🧠 बहु-एजेंट नैदानिक वर्कफ्लो चल रहा है…",
        "spinner_caregiver": "देखभालकर्ता व्याख्या उत्पन्न हो रही है…",
        "success_analysis":  "✅ विश्लेषण पूर्ण — परिणाम नीचे दिखाए गए हैं।",
        "err_no_complaint":  "❌ कृपया कम से कम मुख्य शिकायत या लक्षण विवरण दर्ज करें।",
        "caregiver_summary": "📝 सारांश",
        "caregiver_points":  "⚠️ महत्वपूर्ण बिंदु:",
        "caregiver_questions":"❓ चिकित्सक से प्रश्न:",
        "open_source":       "🔗 स्रोत खोलें",
        "pub_date":          "📅 ",
        "evidence_sources":  "📎 स्रोत: ",
        "confidence":        "विश्वास: ",
        "rationale":         "कारण: ",
        "duration_lbl":      "अवधि: ",
        "severity_lbl":      "गंभीरता: ",
        "frequency_lbl":     "आवृत्ति: ",
        "note_presenting":   "प्रस्तुत शिकायत",
        "note_symptom":      "लक्षण सारांश",
        "note_history":      "प्रासंगिक इतिहास",
        "note_redflag":      "रेड-फ्लैग सारांश",
        "note_evidence":     "साक्ष्य सारांश",
        "note_clinician":    "चिकित्सक समीक्षा नोट",
        "mode_label":        "🎨 प्रदर्शन मोड",
        "lang_label":        "🌐 भाषा",
    },
    "मराठी": {
        "app_name":          "सहविवेक",
        "app_subtitle":      "बहु-एजंट क्लिनिकल निर्णय सहाय्य · संशोधन प्रोटोटाइप",
        "disclaimer":        "⚠️ **हे केवळ संशोधन/शैक्षणिक उपयोगासाठी AI-निर्मित निर्णय सहाय्य आहे।** हे पात्र आरोग्य व्यावसायिकाचा पर्याय नाही. सर्व आउटपुटसाठी चिकित्सकांची समीक्षा आवश्यक आहे।",
        "sidebar_agents":    "🤖 एजंट पाइपलाइन",
        "sidebar_disclaimer":"केवळ संशोधन व शैक्षणिक उपयोग. व्यावसायिक क्लिनिकल निर्णयाचा पर्याय नाही.",
        "sidebar_case":      "📊 केस सारांश",
        "sidebar_symptoms":  "लक्षणे",
        "sidebar_flags":     "रेड फ्लॅग",
        "sidebar_conditions":"अवस्था",
        "sidebar_evidence":  "पुरावे",
        "section_vitals":    "🫀 जीवन-संकेत",
        "section_demo":      "👤 जनसांख्यिकी आणि लक्षण प्रोफाइल",
        "section_narrative": "📋 क्लिनिकल वर्णन",
        "label_temp":        "🌡️ तापमान (°C/°F)",
        "label_hr":          "💓 हृदय गती (bpm)",
        "label_bp":          "🩸 रक्तदाब",
        "label_o2":          "💨 ऑक्सिजन संपृक्तता (%)",
        "label_age":         "वय",
        "label_sex":         "लिंग",
        "label_duration":    "समस्येचा कालावधी",
        "label_severity":    "तीव्रता (स्व-नोंदणी)",
        "label_onset":       "सुरुवात",
        "label_progression": "प्रगती",
        "label_complaint":   "मुख्य तक्रार (थोडक्यात)",
        "label_symptoms":    "स्वतःच्या शब्दांत लक्षणे सांगा",
        "label_history":     "संबंधित वैद्यकीय इतिहास",
        "label_medications": "सध्याची औषधे",
        "label_allergies":   "ऍलर्जी",
        "label_other":       "इतर संबंधित माहिती (पर्यायी)",
        "btn_analyze":       "🔬  केस विश्लेषण करा",
        "sex_options":       ["माहीत नाही", "महिला", "पुरुष", "इंटरसेक्स", "इतर"],
        "severity_options":  ["माहीत नाही", "सौम्य", "मध्यम", "तीव्र"],
        "onset_options":     ["माहीत नाही", "अचानक", "हळूहळू", "येते-जाते"],
        "progression_options":["माहीत नाही", "सुधारत आहे", "बिघडत आहे", "स्थिर"],
        "results_title":     "🧠 बहु-एजंट विश्लेषण परिणाम",
        "agent1_title":      "🧾 एजंट 1 — काढलेली लक्षणे",
        "agent2_title":      "🚨 एजंट 2 — रेड-फ्लॅग शोध",
        "agent3_title":      "🔎 एजंट 3 — अनुपलब्ध माहिती",
        "agent4_title":      "📚 एजंट 4 — वैद्यकीय पुरावे",
        "agent5_title":      "🩺 एजंट 5 — संभाव्य अवस्था",
        "agent6_title":      "📋 एजंट 6 — चिकित्सक-केंद्रित नोट",
        "caregiver_title":   "👨‍👩‍👧 काळजीवाहक मोड",
        "caregiver_caption": "काळजीवाहकांसाठी निष्कर्षांचे सोप्या भाषेत सारांश.",
        "btn_caregiver":     "💬  काळजीवाहक स्पष्टीकरण तयार करा",
        "clinician_title":   "👨‍⚕️ चिकित्सक समीक्षा",
        "clinician_warning": "**मानवी समीक्षा आवश्यक आहे।** AI आउटपुट केवळ निर्णय सहाय्य आहे. अंतिम क्लिनिकल मूल्यांकन पात्र आरोग्य व्यावसायिकाने करणे आवश्यक आहे.",
        "expander_review":   "📄 संरचित चिकित्सक-समीक्षा पॅकेज पहा",
        "no_symptoms":       "कोणतीही लक्षणे काढता आली नाहीत.",
        "no_flags":          "✅ दिलेल्या माहितीमधून कोणतेही रेड फ्लॅग आढळले नाहीत.",
        "no_missing":        "✅ कोणतीही अतिरिक्त माहिती अनुपलब्ध नाही.",
        "no_evidence":       "कोणतेही मंजूर पुरावे मिळाले नाहीत.",
        "no_conditions":     "कोणत्याही पुरावा-समर्थित संभाव्य अवस्था आढळल्या नाहीत.",
        "conditions_caption":"या चिकित्सक समीक्षेसाठी शक्यता आहेत, निश्चित निदान नाही.",
        "form_hint":         "काहीही माहीत नसल्यास रिकामे सोडा — 'माहीत नाही' नोंदवले जाईल, अंदाज नाही.",
        "spinner_main":      "🧠 बहु-एजंट क्लिनिकल वर्कफ्लो चालू आहे…",
        "spinner_caregiver": "काळजीवाहक स्पष्टीकरण तयार होत आहे…",
        "success_analysis":  "✅ विश्लेषण पूर्ण — खाली परिणाम दाखवले आहेत.",
        "err_no_complaint":  "❌ कृपया किमान मुख्य तक्रार किंवा लक्षणांचे वर्णन प्रविष्ट करा.",
        "caregiver_summary": "📝 सारांश",
        "caregiver_points":  "⚠️ महत्त्वाचे मुद्दे:",
        "caregiver_questions":"❓ चिकित्सकांसाठी प्रश्न:",
        "open_source":       "🔗 स्रोत उघडा",
        "pub_date":          "📅 ",
        "evidence_sources":  "📎 स्रोत: ",
        "confidence":        "विश्वास: ",
        "rationale":         "कारण: ",
        "duration_lbl":      "कालावधी: ",
        "severity_lbl":      "तीव्रता: ",
        "frequency_lbl":     "वारंवारता: ",
        "note_presenting":   "सादर तक्रार",
        "note_symptom":      "लक्षण सारांश",
        "note_history":      "संबंधित इतिहास",
        "note_redflag":      "रेड-फ्लॅग सारांश",
        "note_evidence":     "पुरावा सारांश",
        "note_clinician":    "चिकित्सक समीक्षा नोट",
        "mode_label":        "🎨 प्रदर्शन मोड",
        "lang_label":        "🌐 भाषा",
    },
}

AGENTS_BY_LANG = {
    "English": [
        ("1", "Symptom Extraction",    "🧾"),
        ("2", "Red-Flag Detection",    "🚨"),
        ("3", "Missing Information",   "🔎"),
        ("4", "Evidence Retrieval",    "📚"),
        ("5", "Differential Analysis", "🩺"),
        ("6", "Clinical Note",         "📋"),
        ("7", "Caregiver Mode",        "👨‍👩‍👧"),
        ("8", "Clinician Review",      "👨‍⚕️"),
    ],
    "हिंदी": [
        ("1", "लक्षण निष्कर्षण",       "🧾"),
        ("2", "रेड-फ्लैग पहचान",       "🚨"),
        ("3", "अनुपलब्ध जानकारी",      "🔎"),
        ("4", "साक्ष्य पुनर्प्राप्ति",  "📚"),
        ("5", "विभेदक विश्लेषण",        "🩺"),
        ("6", "नैदानिक नोट",           "📋"),
        ("7", "देखभालकर्ता मोड",        "👨‍👩‍👧"),
        ("8", "चिकित्सक समीक्षा",       "👨‍⚕️"),
    ],
    "मराठी": [
        ("1", "लक्षण निष्कर्षण",       "🧾"),
        ("2", "रेड-फ्लॅग शोध",         "🚨"),
        ("3", "अनुपलब्ध माहिती",        "🔎"),
        ("4", "पुरावा पुनर्प्राप्ती",   "📚"),
        ("5", "विभेदक विश्लेषण",        "🩺"),
        ("6", "क्लिनिकल नोट",          "📋"),
        ("7", "काळजीवाहक मोड",          "👨‍👩‍👧"),
        ("8", "चिकित्सक समीक्षा",       "👨‍⚕️"),
    ],
}


# ─────────────────────────────────────────────────────────────────────
# Session-state defaults
# ─────────────────────────────────────────────────────────────────────

if "lang"     not in st.session_state: st.session_state["lang"]     = "English"
if "mode"     not in st.session_state: st.session_state["mode"]     = "Light"

# ─────────────────────────────────────────────────────────────────────
# Sidebar — settings + pipeline
# ─────────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("## 🩺 Sahaviveka")
    st.markdown("सहविवेक")
    st.markdown("---")

    lang = st.selectbox(
        "🌐 Language / भाषा / भाषा",
        ["English", "हिंदी", "मराठी"],
        index=["English", "हिंदी", "मराठी"].index(st.session_state["lang"]),
        key="lang_select",
    )
    st.session_state["lang"] = lang
    T = TRANSLATIONS[lang]

    mode = st.radio(
        T["mode_label"],
        ["Light", "Dark", "Colorblind-Safe"],
        index=["Light", "Dark", "Colorblind-Safe"].index(st.session_state["mode"]),
        horizontal=False,
        key="mode_radio",
    )
    st.session_state["mode"] = mode

    st.markdown("---")
    st.markdown(f"### {T['sidebar_agents']}")
    for num, name, icon in AGENTS_BY_LANG[lang]:
        st.markdown(f"{icon} **Agent {num}** — {name}")

    st.markdown("---")
    st.markdown(f"_{T['sidebar_disclaimer']}_")

    if "case_state" in st.session_state:
        state = st.session_state["case_state"]
        st.markdown("---")
        st.markdown(f"### {T['sidebar_case']}")
        n_symptoms   = len(state.symptom_extraction.symptoms) if (state.symptom_extraction and state.symptom_extraction.symptoms) else 0
        n_flags      = len(state.red_flag_detection.red_flags) if (state.red_flag_detection and state.red_flag_detection.red_flags) else 0
        n_conditions = len(state.possible_conditions) if state.possible_conditions else 0
        n_evidence   = len(state.evidence) if state.evidence else 0
        ca, cb = st.columns(2)
        with ca:
            st.metric(T["sidebar_symptoms"],  n_symptoms)
            st.metric(T["sidebar_flags"],     n_flags)
        with cb:
            st.metric(T["sidebar_conditions"],n_conditions)
            st.metric(T["sidebar_evidence"],  n_evidence)


# ─────────────────────────────────────────────────────────────────────
# Theme palettes
# ─────────────────────────────────────────────────────────────────────

THEMES = {
    "Light": {
        "bg":           "#f0f4f8",
        "card_bg":      "#ffffff",
        "card_border":  "#e2e8f0",
        "text":         "#1e293b",
        "text_muted":   "#64748b",
        "banner_grad":  "linear-gradient(135deg,#0f2942 0%,#1565c0 50%,#0288d1 100%)",
        "banner_txt":   "#ffffff",
        "accent":       "#0288d1",
        "accent_light": "#e8f4fd",
        "chip_bg":      "#e8f4fd",
        "chip_border":  "#b3d9f0",
        "chip_txt":     "#0277bd",
        "flag_bg":      "#fff8f8",
        "flag_border":  "#e53935",
        "note_bg":      "#f8fafc",
        "evidence_accent":"#0288d1",
        "cond_bg":      "#f8fafc",
        "cg_bg":        "linear-gradient(135deg,#e8f5e9,#f1f8e9)",
        "cg_border":    "#a5d6a7",
        "btn_grad":     "linear-gradient(135deg,#0f2942,#1565c0)",
        "critical_bg":  "#fde8e8", "critical_col":"#c62828", "critical_brd":"#ef9a9a",
        "high_bg":      "#fff3e0", "high_col":    "#e65100", "high_brd":    "#ffcc80",
        "moderate_bg":  "#fff8e1", "moderate_col":"#f57f17", "moderate_brd":"#ffe082",
        "low_bg":       "#e8f5e9", "low_col":     "#2e7d32", "low_brd":     "#a5d6a7",
        "sidebar_bg":   "linear-gradient(180deg,#0f2942 0%,#1a4a72 100%)",
    },
    "Dark": {
        "bg":           "#0e1117",
        "card_bg":      "#1a1d27",
        "card_border":  "#2d3145",
        "text":         "#e2e8f0",
        "text_muted":   "#94a3b8",
        "banner_grad":  "linear-gradient(135deg,#0d1b2a 0%,#0e3a6e 50%,#0277bd 100%)",
        "banner_txt":   "#e2e8f0",
        "accent":       "#38bdf8",
        "accent_light": "#0c2d48",
        "chip_bg":      "#0c2d48",
        "chip_border":  "#0369a1",
        "chip_txt":     "#38bdf8",
        "flag_bg":      "#2a1010",
        "flag_border":  "#ef4444",
        "note_bg":      "#12151f",
        "evidence_accent":"#38bdf8",
        "cond_bg":      "#12151f",
        "cg_bg":        "linear-gradient(135deg,#0f2a10,#182a10)",
        "cg_border":    "#4ade80",
        "btn_grad":     "linear-gradient(135deg,#0369a1,#0ea5e9)",
        "critical_bg":  "#2a0d0d", "critical_col":"#f87171", "critical_brd":"#7f1d1d",
        "high_bg":      "#2a1500", "high_col":    "#fb923c", "high_brd":    "#7c2d12",
        "moderate_bg":  "#2a1e00", "moderate_col":"#fbbf24", "moderate_brd":"#78350f",
        "low_bg":       "#0d2a0d", "low_col":     "#4ade80", "low_brd":     "#14532d",
        "sidebar_bg":   "linear-gradient(180deg,#090d14 0%,#0e1c30 100%)",
    },
    "Colorblind-Safe": {
        # Uses IBM colorblind-safe palette
        "bg":           "#f7f7f7",
        "card_bg":      "#ffffff",
        "card_border":  "#d4d4d4",
        "text":         "#1a1a1a",
        "text_muted":   "#555555",
        "banner_grad":  "linear-gradient(135deg,#1a1a1a 0%,#0072b2 100%)",
        "banner_txt":   "#ffffff",
        "accent":       "#0072b2",
        "accent_light": "#e8f2fa",
        "chip_bg":      "#e8f2fa",
        "chip_border":  "#0072b2",
        "chip_txt":     "#004d7a",
        "flag_bg":      "#fff3e0",
        "flag_border":  "#d55e00",
        "note_bg":      "#f7f7f7",
        "evidence_accent":"#0072b2",
        "cond_bg":      "#f7f7f7",
        "cg_bg":        "linear-gradient(135deg,#e8f5e8,#f0f8e8)",
        "cg_border":    "#009e73",
        "btn_grad":     "linear-gradient(135deg,#1a1a1a,#0072b2)",
        "critical_bg":  "#fff0e6", "critical_col":"#d55e00", "critical_brd":"#d55e00",
        "high_bg":      "#fff9e6", "high_col":    "#e69f00", "high_brd":    "#e69f00",
        "moderate_bg":  "#f7f7e6", "moderate_col":"#56b4e9", "moderate_brd":"#56b4e9",
        "low_bg":       "#e6f7e6", "low_col":     "#009e73", "low_brd":     "#009e73",
        "sidebar_bg":   "linear-gradient(180deg,#1a1a1a 0%,#0072b2 100%)",
    },
}

M = THEMES[mode]


# ─────────────────────────────────────────────────────────────────────
# CSS injection
# ─────────────────────────────────────────────────────────────────────

st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

html, body, [class*="css"] {{ font-family: 'Inter', sans-serif; }}
#MainMenu {{ visibility: hidden; }}
footer {{ visibility: hidden; }}

.stApp {{ background: {M["bg"]}; color: {M["text"]}; }}

[data-testid="stSidebar"] {{ background: {M["sidebar_bg"]}; color: white; }}
[data-testid="stSidebar"] * {{ color: white !important; }}
[data-testid="stSidebar"] h1,
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3 {{ color: #7ecff0 !important; }}

.top-banner {{
    background: {M["banner_grad"]};
    border-radius: 16px; padding: 2rem 2.5rem;
    margin-bottom: 1.5rem; color: {M["banner_txt"]};
    display: flex; align-items: center; gap: 1.5rem;
    box-shadow: 0 4px 20px rgba(0,0,0,0.25);
}}
.top-banner .banner-icon {{ font-size: 3.5rem; line-height: 1; }}
.top-banner h1 {{ margin: 0; font-size: 1.9rem; font-weight: 700; color: {M["banner_txt"]} !important; }}
.top-banner .subtitle {{ margin: 0.25rem 0 0 0; font-size: 0.88rem; opacity: 0.8; }}
.top-banner .devanagari {{ font-size: 1rem; opacity: 0.7; letter-spacing: 1px; }}

.section-card {{
    background: {M["card_bg"]}; border-radius: 14px;
    padding: 1.5rem; margin-bottom: 1.25rem;
    box-shadow: 0 2px 12px rgba(0,0,0,0.08);
    border: 1px solid {M["card_border"]}; color: {M["text"]};
}}
.section-card h3 {{
    margin-top: 0; font-size: 1.1rem; font-weight: 600;
    color: {M["text"]}; padding-bottom: 0.6rem;
    border-bottom: 2px solid {M["card_border"]}; margin-bottom: 1rem;
}}

.symptom-chip {{
    display: inline-block; background: {M["chip_bg"]};
    border: 1px solid {M["chip_border"]}; color: {M["chip_txt"]};
    padding: 4px 12px; border-radius: 20px;
    font-size: 0.82rem; font-weight: 500; margin: 3px;
}}

.badge-critical {{
    background: {M["critical_bg"]}; color: {M["critical_col"]};
    border: 1px solid {M["critical_brd"]}; padding: 3px 10px;
    border-radius: 20px; font-size: 0.75rem; font-weight: 700;
    text-transform: uppercase;
}}
.badge-high {{
    background: {M["high_bg"]}; color: {M["high_col"]};
    border: 1px solid {M["high_brd"]}; padding: 3px 10px;
    border-radius: 20px; font-size: 0.75rem; font-weight: 700;
    text-transform: uppercase;
}}
.badge-moderate {{
    background: {M["moderate_bg"]}; color: {M["moderate_col"]};
    border: 1px solid {M["moderate_brd"]}; padding: 3px 10px;
    border-radius: 20px; font-size: 0.75rem; font-weight: 700;
    text-transform: uppercase;
}}
.badge-low {{
    background: {M["low_bg"]}; color: {M["low_col"]};
    border: 1px solid {M["low_brd"]}; padding: 3px 10px;
    border-radius: 20px; font-size: 0.75rem; font-weight: 700;
    text-transform: uppercase;
}}

.red-flag-row {{
    background: {M["flag_bg"]}; border-left: 4px solid {M["flag_border"]};
    padding: 0.8rem 1rem; border-radius: 0 8px 8px 0; margin-bottom: 0.75rem;
}}
.red-flag-row p {{ margin: 0.15rem 0; font-size: 0.88rem; color: {M["text"]}; }}

.condition-card {{
    background: {M["cond_bg"]}; border: 1px solid {M["card_border"]};
    border-radius: 10px; padding: 1rem; margin-bottom: 0.75rem;
}}
.condition-card h4 {{ margin: 0 0 0.4rem 0; font-size: 1rem; color: {M["text"]}; }}
.condition-card p {{ margin: 0.2rem 0; font-size: 0.85rem; color: {M["text_muted"]}; }}

.evidence-card {{
    background: {M["note_bg"]}; border: 1px solid {M["card_border"]};
    border-left: 4px solid {M["evidence_accent"]}; border-radius: 0 10px 10px 0;
    padding: 0.9rem 1.1rem; margin-bottom: 0.75rem;
}}
.evidence-card h5 {{ margin: 0 0 0.3rem 0; font-size: 0.92rem; color: {M["text"]}; }}
.evidence-card p {{ margin: 0.15rem 0; font-size: 0.82rem; color: {M["text_muted"]}; }}

.note-field {{ margin-bottom: 1rem; }}
.note-field label {{
    font-size: 0.78rem; font-weight: 600; color: {M["text_muted"]};
    text-transform: uppercase; letter-spacing: 0.6px; display: block; margin-bottom: 0.2rem;
}}
.note-field p {{
    font-size: 0.92rem; color: {M["text"]}; line-height: 1.6;
    margin: 0; padding: 0.6rem 0.8rem; background: {M["note_bg"]};
    border-radius: 8px; border: 1px solid {M["card_border"]};
}}

.caregiver-card {{
    background: {M["cg_bg"]}; border: 1px solid {M["cg_border"]};
    border-radius: 12px; padding: 1.2rem 1.5rem; margin-bottom: 0.75rem;
}}

div[data-testid="stFormSubmitButton"] > button {{
    background: {M["btn_grad"]} !important;
    color: white !important; border: none !important;
    border-radius: 10px !important; font-size: 1rem !important;
    font-weight: 600 !important; padding: 0.75rem !important;
    letter-spacing: 0.5px; transition: all 0.2s ease;
}}
div[data-testid="stFormSubmitButton"] > button:hover {{
    transform: translateY(-1px);
    box-shadow: 0 6px 20px rgba(0,0,0,0.3) !important;
}}

[data-testid="stForm"] {{
    border: none !important; background: {M["card_bg"]};
    border-radius: 14px; padding: 1.5rem;
    box-shadow: 0 2px 12px rgba(0,0,0,0.08);
}}

hr {{ border-color: {M["card_border"]}; }}
p, li, label, .stMarkdown {{ color: {M["text"]} !important; }}
h1, h2, h3, h4, h5, h6 {{ color: {M["text"]} !important; }}
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────
# Top banner
# ─────────────────────────────────────────────────────────────────────

st.markdown(f"""
<div class="top-banner">
    <div class="banner-icon">🩺</div>
    <div>
        <h1>{T["app_name"]} &nbsp;·&nbsp; <span class="devanagari">सहविवेक</span></h1>
        <p class="subtitle">{T["app_subtitle"]}</p>
    </div>
</div>
""", unsafe_allow_html=True)

st.error(T["disclaimer"])


# ─────────────────────────────────────────────────────────────────────
# Patient input form
# ─────────────────────────────────────────────────────────────────────

st.markdown(f"### 📝 {T['section_narrative']}")
st.caption(T["form_hint"])

with st.form("patient_form"):

    st.markdown(f"##### {T['section_vitals']}")
    v1, v2, v3, v4 = st.columns(4)
    with v1: temperature       = st.text_input(T["label_temp"],  placeholder="e.g. 38.5°C")
    with v2: heart_rate        = st.text_input(T["label_hr"],    placeholder="e.g. 98")
    with v3: blood_pressure    = st.text_input(T["label_bp"],    placeholder="e.g. 120/80")
    with v4: oxygen_saturation = st.text_input(T["label_o2"],    placeholder="e.g. 97")

    st.markdown("---")
    st.markdown(f"##### {T['section_demo']}")

    d1, d2, d3, d4 = st.columns(4)
    with d1: age       = st.text_input(T["label_age"],      placeholder="e.g. 34")
    with d2: sex       = st.selectbox(T["label_sex"],       T["sex_options"])
    with d3: duration  = st.text_input(T["label_duration"], placeholder="e.g. 3 days")
    with d4: severity  = st.select_slider(T["label_severity"], options=T["severity_options"], value=T["severity_options"][0])

    e1, e2 = st.columns(2)
    with e1: onset       = st.selectbox(T["label_onset"],       T["onset_options"])
    with e2: progression = st.selectbox(T["label_progression"], T["progression_options"])

    st.markdown("---")
    st.markdown(f"##### {T['section_narrative']}")

    main_complaint = st.text_input(T["label_complaint"], placeholder="e.g. Fever and headache")

    ta1, ta2 = st.columns(2)
    with ta1:
        symptom_description = st.text_area(T["label_symptoms"],   height=110, placeholder="e.g. Fever for 3 days, severe headache since yesterday...")
        medical_history     = st.text_area(T["label_history"],    height=90,  placeholder="e.g. Type 2 diabetes, or 'none'")
    with ta2:
        medications = st.text_area(T["label_medications"], height=90, placeholder="e.g. Metformin 500mg, or 'none'")
        allergies   = st.text_area(T["label_allergies"],   height=90, placeholder="e.g. Penicillin, or 'none known'")

    other_info = st.text_area(T["label_other"], height=70)

    submitted = st.form_submit_button(T["btn_analyze"], use_container_width=True)


# ─────────────────────────────────────────────────────────────────────
# Run workflow
# ─────────────────────────────────────────────────────────────────────

if submitted:
    if not main_complaint.strip() and not symptom_description.strip():
        st.error(T["err_no_complaint"])
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
            medical_history=medical_history or "Not provided",
            medications=medications or "Not provided",
            allergies=allergies or "Not provided",
            temperature=temperature or "Not provided",
            heart_rate=heart_rate or "Not provided",
            blood_pressure=blood_pressure or "Not provided",
            oxygen_saturation=oxygen_saturation or "Not provided",
            other_info=other_info or "Not provided",
        )
        try:
            with st.spinner(T["spinner_main"]):
                case_state = run_initial_workflow(patient_input)
            st.session_state["case_state"] = case_state
            st.success(T["success_analysis"])
        except LLMConfigError as e:
            st.error(f"⚙️ {e}"); st.stop()
        except LLMRequestError as e:
            st.error(f"📡 {e}"); st.stop()
        except LLMOutputError as e:
            st.error(f"🤖 {e}"); st.stop()
        except Exception as e:
            st.error(f"❌ {type(e).__name__}: {e}"); st.stop()


# ─────────────────────────────────────────────────────────────────────
# Display results
# ─────────────────────────────────────────────────────────────────────

if "case_state" in st.session_state:
    state = st.session_state["case_state"]

    st.markdown("---")
    st.markdown(f"## {T['results_title']}")

    left_col, right_col = st.columns(2, gap="medium")

    # ── Agent 1 ─────────────────────────────────────────────────────
    with left_col:
        st.markdown(f'<div class="section-card"><h3>{T["agent1_title"]}</h3>', unsafe_allow_html=True)
        if state.symptom_extraction and state.symptom_extraction.symptoms:
            chips = "".join(
                f'<span class="symptom-chip">💊 {s.name}</span>'
                for s in state.symptom_extraction.symptoms
            )
            st.markdown(chips + "<br>", unsafe_allow_html=True)
            for s in state.symptom_extraction.symptoms:
                with st.expander(f"📌 {s.name}"):
                    st.markdown(f"- **{T['duration_lbl']}** {s.duration}")
                    st.markdown(f"- **{T['severity_lbl']}** {s.severity}")
                    st.markdown(f"- **{T['frequency_lbl']}** {s.frequency}")
        else:
            st.info(T["no_symptoms"])
        st.markdown("</div>", unsafe_allow_html=True)

    # ── Agent 2 ─────────────────────────────────────────────────────
    with right_col:
        st.markdown(f'<div class="section-card"><h3>{T["agent2_title"]}</h3>', unsafe_allow_html=True)
        if state.red_flag_detection and state.red_flag_detection.red_flags:
            for flag in state.red_flag_detection.red_flags:
                u = (flag.urgency or "").lower()
                badge = ("badge-critical" if "critical" in u else
                         "badge-high"     if "high"     in u else
                         "badge-moderate" if "moderate" in u else "badge-low")
                st.markdown(f"""
                    <div class="red-flag-row">
                      <p><strong>🚩 {flag.flag}</strong>
                         &nbsp;<span class="{badge}">{flag.urgency}</span></p>
                      <p style="color:{M['text_muted']};">{flag.reason}</p>
                    </div>""", unsafe_allow_html=True)
        else:
            st.success(T["no_flags"])
        st.markdown("</div>", unsafe_allow_html=True)

    left_col2, right_col2 = st.columns(2, gap="medium")

    # ── Agent 3 ─────────────────────────────────────────────────────
    with left_col2:
        st.markdown(f'<div class="section-card"><h3>{T["agent3_title"]}</h3>', unsafe_allow_html=True)
        if state.missing_information:
            for item in state.missing_information:
                st.markdown(f"- 📌 {item}")
        else:
            st.success(T["no_missing"])
        st.markdown("</div>", unsafe_allow_html=True)

    # ── Agent 4 ─────────────────────────────────────────────────────
    with right_col2:
        st.markdown(f'<div class="section-card"><h3>{T["agent4_title"]}</h3>', unsafe_allow_html=True)
        if state.evidence:
            for item in state.evidence:
                preview = item.evidence_text.strip()
                if len(preview) > 350:
                    preview = preview[:350].rsplit(" ", 1)[0] + "…"
                src_link = (
                    f'<a href="{item.source_url}" target="_blank" '
                    f'style="font-size:0.8rem;color:{M["accent"]};">{T["open_source"]}</a>'
                    if item.source_url else ""
                )
                pub = f'<p>{T["pub_date"]}{item.publication_date}</p>' if item.publication_date else ""
                st.markdown(f"""
                    <div class="evidence-card">
                      <h5>{item.title}</h5>
                      <p style="color:{M["accent"]};font-weight:500;">{item.source_organization} ({item.source_name})</p>
                      {pub}
                      <p>{preview}</p>
                      {src_link}
                    </div>""", unsafe_allow_html=True)
        else:
            st.info(T["no_evidence"])
        st.markdown("</div>", unsafe_allow_html=True)

    # ── Agent 5 ─────────────────────────────────────────────────────
    st.markdown(f'<div class="section-card"><h3>{T["agent5_title"]}</h3>', unsafe_allow_html=True)
    st.caption(T["conditions_caption"])
    if state.possible_conditions:
        cond_cols = st.columns(min(len(state.possible_conditions), 3))
        for idx, cond in enumerate(state.possible_conditions):
            confidence = cond.get("confidence", "Uncertain")
            conf_l = confidence.lower()
            conf_color = (M["critical_col"] if "high" in conf_l else
                          M["high_col"]     if "moderate" in conf_l or "medium" in conf_l else
                          M["low_col"])
            source_ids = cond.get("evidence_source_ids", [])
            src_html = (f'<p style="color:{M["accent"]};font-size:0.8rem;">'
                        f'{T["evidence_sources"]}{", ".join(source_ids)}</p>'
                        if source_ids else "")
            with cond_cols[idx % 3]:
                st.markdown(f"""
                    <div class="condition-card">
                      <h4>🏥 {cond.get('condition', 'Unspecified')}</h4>
                      <p>{T['rationale']}{cond.get('rationale', '')}</p>
                      {src_html}
                      <p style="color:{conf_color};font-weight:600;font-size:0.82rem;">
                          {T['confidence']}{confidence}
                      </p>
                    </div>""", unsafe_allow_html=True)
    else:
        st.info(T["no_conditions"])
    st.markdown("</div>", unsafe_allow_html=True)

    # ── Agent 6 ─────────────────────────────────────────────────────
    st.markdown(f'<div class="section-card"><h3>{T["agent6_title"]}</h3>', unsafe_allow_html=True)
    if state.clinical_note:
        try:
            note = json.loads(state.clinical_note)
            pc = note.get("presenting_complaint", "Not provided").replace(",headache", ", headache")
            ev = note.get("evidence_summary", "Not provided")
            if len(ev) > 700:
                ev = ev[:700].rsplit(" ", 1)[0] + "…"
            note_fields = [
                (T["note_presenting"], pc),
                (T["note_symptom"],    note.get("symptom_summary",    "Not provided")),
                (T["note_history"],    note.get("relevant_history",   "Not provided")),
                (T["note_redflag"],    note.get("red_flag_summary",   "Not provided")),
                (T["note_evidence"],   ev),
                (T["note_clinician"],  note.get("clinician_review_note", "Not provided")),
            ]
            note_cols = st.columns(2)
            for i, (label, value) in enumerate(note_fields):
                with note_cols[i % 2]:
                    st.markdown(f"""
                        <div class="note-field">
                          <label>{label}</label>
                          <p>{value}</p>
                        </div>""", unsafe_allow_html=True)
        except json.JSONDecodeError:
            st.markdown(state.clinical_note)
    st.markdown("</div>", unsafe_allow_html=True)

    # ── Caregiver Mode ───────────────────────────────────────────────
    st.markdown("---")
    st.markdown(f"### {T['caregiver_title']}")
    st.caption(T["caregiver_caption"])

    if st.button(T["btn_caregiver"], use_container_width=True):
        try:
            with st.spinner(T["spinner_caregiver"]):
                cg = run_caregiver_agent(state)
            st.markdown(f"""
                <div class="caregiver-card">
                  <h4 style="margin:0 0 0.5rem 0;color:{M['cg_border']};">{T['caregiver_summary']}</h4>
                  <p style="margin:0;color:{M['text']};">{cg.summary}</p>
                </div>""", unsafe_allow_html=True)
            if cg.important_points:
                st.markdown(f"**{T['caregiver_points']}**")
                for pt in cg.important_points:
                    st.markdown(f"- {pt}")
            if cg.questions_for_clinician:
                st.markdown(f"**{T['caregiver_questions']}**")
                for q in cg.questions_for_clinician:
                    st.markdown(f"- {q}")
            st.info(f"🛡️ {cg.safety_note}")
        except LLMConfigError  as e: st.error(f"⚙️ {e}")
        except LLMRequestError as e: st.error(f"📡 {e}")
        except LLMOutputError  as e: st.error(f"🤖 {e}")

    # ── Clinician Review ─────────────────────────────────────────────
    st.markdown("---")
    st.markdown(f"### {T['clinician_title']}")
    st.warning(T["clinician_warning"])
    review = build_clinician_review(state)
    with st.expander(T["expander_review"]):
        st.json(review)
