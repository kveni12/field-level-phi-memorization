"""OpenAI GPT client for Luna / GPT surface realization."""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence

from dotenv import load_dotenv


class OpenAIError(RuntimeError):
    """Raised when OpenAI generation fails after retries."""


@dataclass(frozen=True)
class OpenAIGenerateResult:
    text: str
    model: str
    raw: dict


def _load_api_key(explicit: Optional[str] = None) -> str:
    load_dotenv(dotenv_path=Path.cwd() / ".env", override=False)
    repo_root = Path(__file__).resolve().parents[3]
    load_dotenv(dotenv_path=repo_root / ".env", override=False)
    load_dotenv(override=False)
    key = explicit or os.environ.get("OPENAI_API_KEY")
    if not key:
        raise OpenAIError(
            "OPENAI_API_KEY not set. Export it or put it in the repo-root .env file."
        )
    return key.strip()


class OpenAIClient:
    """Thin wrapper around the official OpenAI Python SDK."""

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = _load_api_key(api_key)
        try:
            from openai import OpenAI
        except ImportError as exc:  # pragma: no cover
            raise ImportError(
                "openai is required. Install with `pip install openai`."
            ) from exc
        self._client = OpenAI(api_key=self.api_key)

    def generate(
        self,
        *,
        model: str,
        prompt: str,
        system: str,
        temperature: float = 0.3,
        seed: Optional[int] = None,
        num_predict: int = 256,
        format_json: bool = True,
        max_retries: int = 8,
        top_p: Optional[float] = None,
        frequency_penalty: Optional[float] = None,
        presence_penalty: Optional[float] = None,
        reasoning_effort: Optional[str] = None,
    ) -> OpenAIGenerateResult:
        kwargs: dict = {
            "model": model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": prompt},
            ],
            "temperature": float(temperature),
            "max_completion_tokens": int(num_predict),
        }
        if format_json:
            kwargs["response_format"] = {"type": "json_object"}
        if seed is not None:
            kwargs["seed"] = int(seed)
        if top_p is not None:
            kwargs["top_p"] = float(top_p)
        if frequency_penalty is not None:
            kwargs["frequency_penalty"] = float(frequency_penalty)
        if presence_penalty is not None:
            kwargs["presence_penalty"] = float(presence_penalty)
        if reasoning_effort is not None:
            kwargs["reasoning_effort"] = str(reasoning_effort)

        last_err: Optional[Exception] = None
        for attempt in range(max_retries):
            try:
                response = self._client.chat.completions.create(**kwargs)
                text = (response.choices[0].message.content or "").strip()
                if not text:
                    raise OpenAIError("Empty OpenAI response")
                raw = response.model_dump() if hasattr(response, "model_dump") else {"text": text}
                return OpenAIGenerateResult(text=text, model=model, raw=raw)
            except Exception as exc:  # noqa: BLE001
                last_err = exc
                msg = str(exc).lower()
                if "max_completion_tokens" in msg:
                    kwargs.pop("max_completion_tokens", None)
                    kwargs["max_tokens"] = int(num_predict)
                    continue
                if "reasoning_effort" in msg and (
                    "unsupported" in msg or "unknown" in msg or "invalid" in msg
                ):
                    kwargs.pop("reasoning_effort", None)
                    continue
                if "frequency_penalty" in msg or "presence_penalty" in msg:
                    kwargs.pop("frequency_penalty", None)
                    kwargs.pop("presence_penalty", None)
                    continue
                if "temperature" in msg and "unsupported" in msg:
                    kwargs.pop("temperature", None)
                    continue
                if "seed" in msg and "unsupported" in msg:
                    kwargs.pop("seed", None)
                    continue
                if any(tok in msg for tok in ("429", "rate", "quota", "timeout", "500", "503")):
                    time.sleep(min(30.0, max(2.0, 1.75 ** attempt)))
                    continue
                if attempt + 1 < max_retries:
                    time.sleep(0.75 * (attempt + 1))
                    continue
                break
        raise OpenAIError(f"OpenAI generate failed for model={model}: {last_err}")
