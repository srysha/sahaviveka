"""
tests/test_symptom_agent.py

These tests check Agent 1's LOGIC without calling the real Claude API -
we "mock" (fake) the LLM call so the tests run instantly, for free, and
without needing an API key. This is what lets you test your code before
you even have billing set up.

Run with:  python -m pytest tests/test_symptom_agent.py -v
"""

from unittest.mock import patch

from models.schemas import PatientInput, SymptomExtraction, ExtractedSymptom
from agents.symptom_agent import run_symptom_agent
from utils.llm import LLMConfigError


def _sample_patient() -> PatientInput:
    return PatientInput(
        main_complaint="Fever and headache",
        symptom_description=(
            "I have had fever for three days and severe headache since yesterday. "
            "I also vomited twice."
        ),
    )


@patch("agents.symptom_agent.call_structured_llm")
def test_symptom_agent_returns_valid_structure(mock_llm):
    """Agent 1 should return a SymptomExtraction object built from the LLM's structured reply."""
    mock_llm.return_value = SymptomExtraction(
        symptoms=[
            ExtractedSymptom(name="fever", duration="3 days"),
            ExtractedSymptom(name="severe headache", duration="1 day"),
            ExtractedSymptom(name="vomiting", frequency="twice"),
        ],
        medical_history=[],
        medications=[],
        allergies=[],
    )

    result = run_symptom_agent(_sample_patient())

    assert isinstance(result, SymptomExtraction)
    assert len(result.symptoms) == 3
    assert result.symptoms[0].name == "fever"
    assert result.symptoms[0].duration == "3 days"
    mock_llm.assert_called_once()


@patch("agents.symptom_agent.call_structured_llm")
def test_symptom_agent_handles_empty_input_gracefully(mock_llm):
    """If nothing meaningful was said, the agent should return empty lists, not invented data."""
    mock_llm.return_value = SymptomExtraction(symptoms=[], medical_history=[], medications=[], allergies=[])

    empty_patient = PatientInput(main_complaint="", symptom_description="")
    result = run_symptom_agent(empty_patient)

    assert result.symptoms == []


@patch("agents.symptom_agent.call_structured_llm")
def test_symptom_agent_propagates_config_errors(mock_llm):
    """If the API key is missing, the agent must raise a clear error, not crash silently or fake data."""
    mock_llm.side_effect = LLMConfigError("No API key found.")

    try:
        run_symptom_agent(_sample_patient())
        assert False, "Expected LLMConfigError to be raised"
    except LLMConfigError as e:
        assert "API key" in str(e)
