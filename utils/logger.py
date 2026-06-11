import logging
import json
import time

logging.basicConfig(filename='langgraph_smart_reasoning.log', level=logging.INFO)
logger = logging.getLogger(__name__)

def log_event(event: str, **kwargs):
    logger.info(json.dumps({"event": event, "ts": time.time(), **kwargs}))