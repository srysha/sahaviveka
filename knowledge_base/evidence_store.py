"""
knowledge_base/evidence_store.py

Agent 4 — Medical Evidence Retrieval

Retrieves relevant medical evidence from approved medical sources.

This module retrieves evidence only.
It does not diagnose the patient.
"""

from typing import List
from urllib.parse import urlparse
import os

from models.schemas import EvidenceItem


# -------------------------------------------------------------------
# APPROVED MEDICAL SOURCES
# -------------------------------------------------------------------

APPROVED_DOMAINS = {
    "who.int": "WHO",
    "cdc.gov": "CDC",
    "nice.org.uk": "NICE",
    "nih.gov": "NIH",
    "medlineplus.gov": "MedlinePlus",
    "nhs.uk": "NHS",
}


# -------------------------------------------------------------------
# DOMAIN CHECK
# -------------------------------------------------------------------

def is_approved_url(url: str) -> bool:

    try:
        hostname = urlparse(url).hostname

        if not hostname:
            return False

        hostname = hostname.lower()

        for domain in APPROVED_DOMAINS:

            if (
                hostname == domain
                or hostname.endswith("." + domain)
            ):
                return True

        return False

    except Exception:
        return False


def _get_organization(url: str) -> str:

    try:
        hostname = (
            urlparse(url).hostname or ""
        ).lower()

        for domain, organization in APPROVED_DOMAINS.items():

            if (
                hostname == domain
                or hostname.endswith("." + domain)
            ):
                return organization

    except Exception:
        pass

    return "Medical source"


# -------------------------------------------------------------------
# QUERY
# -------------------------------------------------------------------

def build_search_query(
    symptoms: List[str],
) -> str:

    cleaned = [
        str(symptom).strip()
        for symptom in symptoms
        if str(symptom).strip()
    ]

    symptom_text = ", ".join(
        cleaned
    )

    return (
        f"clinical guidance for {symptom_text}; "
        "symptoms; causes; assessment; "
        "warning signs; management; "
        "differential diagnosis"
    )


# -------------------------------------------------------------------
# RELEVANCE
# -------------------------------------------------------------------

def _normalise(text: str) -> str:

    return " ".join(
        str(text or "").lower().split()
    )


def _relevance_score(
    result: dict,
    symptoms: List[str],
) -> float:

    title = _normalise(
        result.get("title", "")
    )

    content = _normalise(
        result.get("content", "")
    )

    url = _normalise(
        result.get("url", "")
    )

    combined = (
        title
        + " "
        + content
        + " "
        + url
    )

    score = float(
        result.get("score", 0) or 0
    )

    # Direct symptom phrase matches
    for symptom in symptoms:

        symptom_text = _normalise(
            symptom
        )

        if not symptom_text:
            continue

        if symptom_text in title:
            score += 5

        elif symptom_text in combined:
            score += 2

        # Also reward individual meaningful words
        for word in symptom_text.split():

            if len(word) >= 5 and word in combined:
                score += 0.5

    # Prefer clinical guidance
    guidance_terms = [
        "guideline",
        "clinical guidance",
        "clinical assessment",
        "symptoms",
        "management",
        "treatment",
        "diagnosis",
        "signs",
    ]

    for term in guidance_terms:

        if term in combined:
            score += 0.5

    # Penalize obvious isolated case reports rather than
    # completely rejecting them.
    case_terms = [
        "case report",
        "case study",
        "case presentation",
    ]

    for term in case_terms:

        if term in title:
            score -= 3

    return score


# -------------------------------------------------------------------
# RETRIEVE EVIDENCE
# -------------------------------------------------------------------

def retrieve_evidence(
    symptoms: List[str],
) -> List[EvidenceItem]:

    if not symptoms:
        print(
            "Agent 4: No symptoms available for evidence retrieval."
        )
        return []

    api_key = os.getenv(
        "TAVILY_API_KEY"
    )

    if not api_key:
        print(
            "Agent 4: TAVILY_API_KEY is not configured."
        )
        return []

    try:

        from tavily import TavilyClient

    except ImportError:

        print(
            "Agent 4: Tavily package is not installed."
        )

        return []

    query = build_search_query(
        symptoms
    )

    print(
        "\n=============================="
    )
    print(
        "AGENT 4 SEARCH QUERY"
    )
    print(
        "=============================="
    )

    print(query)

    try:

        client = TavilyClient(
            api_key=api_key
        )

        response = client.search(
            query=query,
            search_depth="advanced",
            max_results=10,
            include_domains=list(
                APPROVED_DOMAINS.keys()
            ),
        )

    except Exception as error:

        print(
            f"Agent 4 Tavily retrieval failed: {error}"
        )

        return []

    raw_results = response.get(
        "results",
        []
    )

    print(
        f"Agent 4 raw search results: "
        f"{len(raw_results)}"
    )

    candidates = []

    for result in raw_results:

        url = str(
            result.get("url", "")
        ).strip()

        title = str(
            result.get("title", "")
        ).strip()

        content = str(
            result.get("content", "")
        ).strip()

        if not url:
            continue

        if not is_approved_url(url):
            continue

        if not title:
            title = "Medical evidence source"

        if not content:

            content = str(
                result.get(
                    "raw_content",
                    ""
                )
            ).strip()

        if not content:
            continue

        candidates.append(
            {
                "result": result,
                "url": url,
                "title": title,
                "content": content,
                "score": _relevance_score(
                    result,
                    symptoms,
                ),
            }
        )

    candidates.sort(
        key=lambda item: item["score"],
        reverse=True,
    )

    selected = candidates[:5]

    evidence = []

    for item in selected:

        result = item["result"]

        organization = _get_organization(
            item["url"]
        )

        publication_date = (
            result.get("published_date")
            or result.get("date")
        )

        evidence.append(
            EvidenceItem(
                source_organization=organization,
                source_name=organization,
                source_url=item["url"],
                title=item["title"],
                publication_date=publication_date,
                topic=", ".join(symptoms),
                evidence_text=item["content"],
                relevance="High"
                if item["score"] >= 3
                else "Relevant",
                source_id=item["url"],
            )
        )

    print(
        f"APPROVED RELEVANT EVIDENCE FOUND: "
        f"{len(evidence)}"
    )

    for index, item in enumerate(
        evidence,
        start=1,
    ):

        print(
            f"\nSOURCE {index}: "
            f"{item.source_organization}"
        )

        print(
            f"TITLE: {item.title}"
        )

        print(
            f"URL: {item.source_url}"
        )

    return evidence