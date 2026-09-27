"""
utils/llm.py

LLM utility layer using OpenRouter.

Provides:
- structured JSON output
- Pydantic validation
- retry handling
- OpenRouter model fallback
"""

import json
import os
import time
from typing import Type, TypeVar

import requests
from dotenv import load_dotenv
from pydantic import BaseModel, ValidationError


# ============================================================
# CONFIGURATION
# ============================================================

load_dotenv()

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"

DEFAULT_MODEL = "openrouter/free"

# OpenRouter tries these models in order when the primary
# request fails or is rate-limited.
FALLBACK_MODELS = [
    "openrouter/free",
    "openai/gpt-oss-20b:free",
]


# ============================================================
# EXCEPTIONS
# ============================================================

class LLMConfigError(Exception):
    pass


class LLMRequestError(Exception):
    pass


class LLMOutputError(Exception):
    pass


# ============================================================
# TYPE VARIABLE
# ============================================================

T = TypeVar("T", bound=BaseModel)


# ============================================================
# STRUCTURED LLM CALL
# ============================================================

def call_structured_llm(
    system_prompt: str,
    user_prompt: str,
    response_model: Type[T],
    temperature: float = 0.0,
    max_tokens: int = 2000,
    max_retries: int = 3,
) -> T:

    # --------------------------------------------------------
    # Check API key
    # --------------------------------------------------------

    if not OPENROUTER_API_KEY:
        raise LLMConfigError(
            "OPENROUTER_API_KEY is missing from the .env file."
        )

    # --------------------------------------------------------
    # Convert Pydantic model into JSON schema
    # --------------------------------------------------------

    schema = response_model.model_json_schema()

    # --------------------------------------------------------
    # Headers
    # --------------------------------------------------------

    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
        "HTTP-Referer": "http://localhost:8501",
        "X-Title": "Clinical AI Prototype",
    }

    # --------------------------------------------------------
    # Request payload
    # --------------------------------------------------------

    payload = {
        "model": DEFAULT_MODEL,

        # OpenRouter uses this list as model-level fallback.
        # If the primary model fails or is rate-limited,
        # OpenRouter can try the next model.
        "models": FALLBACK_MODELS,

        "messages": [
            {
                "role": "system",
                "content": system_prompt,
            },
            {
                "role": "user",
                "content": user_prompt,
            },
        ],

        "temperature": temperature,

        "max_tokens": max_tokens,

        "response_format": {
            "type": "json_schema",
            "json_schema": {
                "name": response_model.__name__,
                "strict": True,
                "schema": schema,
            },
        },
    }

    # --------------------------------------------------------
    # Retry loop
    # --------------------------------------------------------

    last_error = None

    for attempt in range(1, max_retries + 1):

        print(
            f"\n===== OPENROUTER ATTEMPT "
            f"{attempt}/{max_retries} ====="
        )

        try:

            response = requests.post(
                OPENROUTER_URL,
                headers=headers,
                json=payload,
                timeout=90,
            )

            # ------------------------------------------------
            # HTTP errors
            # ------------------------------------------------

            if response.status_code != 200:

                error_text = response.text

                # Rate limit / temporary server errors
                if response.status_code in {429, 500, 502, 503, 504}:

                    print(
                        f"OpenRouter temporary error "
                        f"({response.status_code}). "
                        f"Retrying..."
                    )

                    last_error = LLMRequestError(
                        f"OpenRouter API error "
                        f"{response.status_code}: "
                        f"{error_text}"
                    )

                    if attempt < max_retries:
                        wait_time = 2 ** (attempt - 1)
                        time.sleep(wait_time)
                        continue

                    raise last_error

                # Other HTTP errors
                raise LLMRequestError(
                    f"OpenRouter API error "
                    f"{response.status_code}: "
                    f"{error_text}"
                )

            # ------------------------------------------------
            # Parse response JSON
            # ------------------------------------------------

            try:
                data = response.json()

            except ValueError as exc:

                last_error = LLMOutputError(
                    "OpenRouter returned invalid JSON."
                )

                if attempt < max_retries:
                    time.sleep(2 ** (attempt - 1))
                    continue

                raise last_error from exc

            # ------------------------------------------------
            # Extract assistant message
            # ------------------------------------------------

            try:
                content = data["choices"][0]["message"]["content"]

            except (KeyError, IndexError, TypeError) as exc:

                last_error = LLMOutputError(
                    f"OpenRouter response did not contain "
                    f"expected message content.\n"
                    f"Response: {data}"
                )

                if attempt < max_retries:
                    time.sleep(2 ** (attempt - 1))
                    continue

                raise last_error from exc

            # ------------------------------------------------
            # Check empty output
            # ------------------------------------------------

            if not content or not content.strip():

                last_error = LLMOutputError(
                    "OpenRouter returned an empty response."
                )

                print("OpenRouter returned empty output.")

                if attempt < max_retries:
                    time.sleep(2 ** (attempt - 1))
                    continue

                raise last_error

            # ------------------------------------------------
            # Print raw response for debugging
            # ------------------------------------------------

            print("\nRAW OPENROUTER RESPONSE")
            print(content)

            # ------------------------------------------------
            # Parse JSON
            # ------------------------------------------------

            try:

                parsed = json.loads(content)

            except json.JSONDecodeError as exc:

                last_error = LLMOutputError(
                    "OpenRouter returned text that was "
                    "not valid JSON.\n"
                    f"Raw output:\n{content}"
                )

                if attempt < max_retries:
                    time.sleep(2 ** (attempt - 1))
                    continue

                raise last_error from exc

            # ------------------------------------------------
            # Validate against Pydantic model
            # ------------------------------------------------

            try:

                result = response_model.model_validate(parsed)

            except ValidationError as exc:

                last_error = LLMOutputError(
                    "OpenRouter returned JSON that does not "
                    "match the required schema.\n"
                    f"Validation error:\n{exc}\n"
                    f"Raw output:\n{content}"
                )

                if attempt < max_retries:
                    time.sleep(2 ** (attempt - 1))
                    continue

                raise last_error from exc

            # ------------------------------------------------
            # Success
            # ------------------------------------------------

            print(
                f"\nStructured output validated successfully "
                f"using {response_model.__name__}."
            )

            return result

        except requests.RequestException as exc:

            last_error = LLMRequestError(
                f"Network error while contacting OpenRouter: "
                f"{exc}"
            )

            print(f"Network error: {exc}")

            if attempt < max_retries:
                time.sleep(2 ** (attempt - 1))
                continue

            raise last_error from exc

    # --------------------------------------------------------
    # Should never normally reach here
    # --------------------------------------------------------

    raise LLMRequestError(
        f"OpenRouter request failed after "
        f"{max_retries} attempts.\n"
        f"Last error: {last_error}"
    )