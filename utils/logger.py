import logging
import json
import time

logger = logging.getLogger(__name__)
logging.basicConfig(filename='langgraph_smart_reasoning.log', level=logging.INFO)

def log_event(event: str, **kwargs):
    logger.info(json.dumps({"event": event, "ts": time.time(), **kwargs}))