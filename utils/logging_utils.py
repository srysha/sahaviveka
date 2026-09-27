"""
utils/logging_utils.py

Sets up simple logging so we can see, in a file called app.log, what each
agent did and when. On purpose, we log STATUS and ERRORS, not the actual
patient text - so we don't build a habit of writing identifying health
details into a plain text file.
"""

import logging
import os

LOG_FILE = os.path.join(os.path.dirname(os.path.dirname(__file__)), "app.log")


def setup_logging():
    logger = logging.getLogger("clinical_ai")
    if logger.handlers:  # avoid duplicate handlers if called more than once
        return logger

    logger.setLevel(logging.INFO)

    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
    )

    file_handler = logging.FileHandler(LOG_FILE)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    return logger


def log_agent_event(agent_name: str, status: str, detail: str = ""):
    """
    Log one line about an agent's run.
    detail should be a short, non-identifying note (e.g. '3 symptoms extracted'),
    never raw patient text.
    """
    logger = logging.getLogger("clinical_ai")
    logger.info(f"[{agent_name}] status={status} {detail}")
