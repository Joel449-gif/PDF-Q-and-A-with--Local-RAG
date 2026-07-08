import json
import logging
import time

import requests

import config

logger = logging.getLogger(__name__)

_use_gemini = bool(config.GEMINI_API_KEY)
_gemini_client = None


def _get_gemini_client():
    global _gemini_client
    if _gemini_client is None and _use_gemini:
        from google import genai
        _gemini_client = genai.Client(api_key=config.GEMINI_API_KEY)
    return _gemini_client


def generate(prompt: str) -> str:
    if _use_gemini:
        client = _get_gemini_client()
        response = client.models.generate_content(
            model=config.GEMINI_MODEL,
            contents=prompt,
            config={
                "temperature": config.LLM_TEMPERATURE,
                "max_output_tokens": config.LLM_MAX_TOKENS,
            },
        )
        return response.text.strip()
    else:
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
    """Generator yielding text tokens from Gemini (if key set) or Ollama."""
    start = time.time()

    if _use_gemini:
        client = _get_gemini_client()
        for chunk in client.models.generate_content_stream(
            model=config.GEMINI_MODEL,
            contents=prompt,
            config={
                "temperature": config.LLM_TEMPERATURE,
                "max_output_tokens": config.LLM_MAX_TOKENS,
            },
        ):
            if chunk.text:
                yield chunk.text
    else:
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
