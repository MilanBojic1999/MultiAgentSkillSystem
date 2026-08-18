import json
import logging
import os
import time

_TRUTHY = ("1", "true", "yes", "on")


def _build_logger() -> logging.Logger:
    """Configure the pipeline's dedicated logger from env vars.

    LOG_LEVEL   — standard logging level name (default INFO).
    LOG_FILE    — log file path (default langgraph_smart_reasoning.log);
                  set to an empty string to disable file logging.
    LOG_CONSOLE — "true" to also emit events to stderr (default false).
    """
    logger = logging.getLogger("agent_pipeline")
    if logger.handlers:
        return logger

    level = getattr(logging, os.getenv("LOG_LEVEL", "INFO").upper(), logging.INFO)
    logger.setLevel(level)
    logger.propagate = False  # keep events out of the root logger (uvicorn etc.)

    formatter = logging.Formatter("%(message)s")  # log_event lines are already JSON

    log_file = os.getenv("LOG_FILE", "langgraph_smart_reasoning.log")
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    if os.getenv("LOG_CONSOLE", "false").strip().lower() in _TRUTHY:
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

    if not logger.handlers:  # LOG_FILE="" and console off — swallow silently
        logger.addHandler(logging.NullHandler())

    return logger


logger = _build_logger()


def log_event(event: str, **kwargs):
    logger.info(json.dumps({"event": event, "ts": time.time(), **kwargs}, default=str))
