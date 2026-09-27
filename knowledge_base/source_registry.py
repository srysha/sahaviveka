"""
knowledge_base/source_registry.py

Approved medical-source registry for the clinical AI prototype.

Only sources included here are allowed to become evidence
for the clinical reasoning stage.
"""

APPROVED_SOURCES = [
    {
        "organization": "World Health Organization",
        "short_name": "WHO",
        "base_url": "https://www.who.int/",
        "trust_level": "primary_health_authority",
    },
    {
        "organization": "Centers for Disease Control and Prevention",
        "short_name": "CDC",
        "base_url": "https://www.cdc.gov/",
        "trust_level": "primary_health_authority",
    },
    {
        "organization": "National Institute for Health and Care Excellence",
        "short_name": "NICE",
        "base_url": "https://www.nice.org.uk/",
        "trust_level": "clinical_guideline_authority",
    },
]