import json
import logging
import time

import requests

import config

logger = logging.getLogger(__name__)


def _use_gemini():
    return bool(config.GEMINI_API_KEY)


def _gemini_request(prompt: str, stream: bool = False):
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{config.GEMINI_MODEL}:{'streamGenerateContent' if stream else 'generateContent'}?key={config.GEMINI_API_KEY}"
    payload = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {
            "temperature": config.LLM_TEMPERATURE,
            "maxOutputTokens": config.LLM_MAX_TOKENS,
        },
    }
    resp = requests.post(url, json=payload, timeout=(10, config.LLM_TIMEOUT))
    resp.raise_for_status()
    return resp


def generate(prompt: str) -> str:
    if _use_gemini():
        try:
            resp = _gemini_request(prompt, stream=False)
            data = resp.json()
            return data["candidates"][0]["content"]["parts"][0]["text"].strip()
        except Exception as e:
            logger.error("Gemini generate error: %s", e)
            return f"Gemini API error: {e}"
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
    start = time.time()

    if _use_gemini():
        try:
            resp = _gemini_request(prompt, stream=True)
            for line in resp.iter_lines():
                if not line:
                    continue
                line = line.decode("utf-8")
                if line.startswith("data: "):
                    data = json.loads(line[6:])
                    text = data.get("candidates", [{}])[0].get("content", {}).get("parts", [{}])[0].get("text", "")
                    if text:
                        yield text
        except Exception as e:
            logger.error("Gemini stream error: %s", e)
            yield f"Gemini API error: {e}"
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
