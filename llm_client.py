import json
import logging
import time

import requests

import config

logger = logging.getLogger(__name__)


def generate(prompt: str) -> str:
    payload = {
        "model": config.OLLAMA_MODEL,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": config.LLM_TEMPERATURE,
            "num_predict": config.LLM_MAX_TOKENS,
        },
    }
    resp = requests.post(
        f"{config.OLLAMA_BASE_URL}/api/generate",
        json=payload,
        timeout=(10, config.LLM_TIMEOUT),
    )
    resp.raise_for_status()
    return resp.json()["response"].strip()


def generate_stream(prompt: str):
    start = time.time()
    payload = {
        "model": config.OLLAMA_MODEL,
        "prompt": prompt,
        "stream": True,
        "options": {
            "temperature": config.LLM_TEMPERATURE,
            "num_predict": config.LLM_MAX_TOKENS,
        },
    }
    try:
        resp = requests.post(
            f"{config.OLLAMA_BASE_URL}/api/generate",
            json=payload,
            stream=True,
            timeout=(10, config.LLM_TIMEOUT),
        )
        resp.raise_for_status()
    except requests.exceptions.Timeout:
        logger.warning("Ollama timeout after %ds", time.time() - start)
        yield "The model took too long. Try asking again."
        return
    except requests.exceptions.ConnectionError:
        logger.error("Cannot reach Ollama at %s", config.OLLAMA_BASE_URL)
        yield f"Cannot reach Ollama at {config.OLLAMA_BASE_URL}. Make sure `ollama serve` is running."
        return

    for line in resp.iter_lines():
        if not line:
            continue
        try:
            chunk = json.loads(line.decode("utf-8"))
            token = chunk.get("response", "")
            if token:
                yield token
        except json.JSONDecodeError:
            continue

    elapsed = time.time() - start
    logger.info("Generated response in %.2fs", elapsed)
