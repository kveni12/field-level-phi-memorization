"""JSON parsing helpers for model responses."""

from __future__ import annotations

import json


def parse_model_json(raw_text: str) -> dict:
    text = raw_text.strip()
    if not text:
        raise ValueError("Empty model response")
    try:
        data = json.loads(text)
        if isinstance(data, dict):
            return data
    except json.JSONDecodeError:
        pass
    start = text.find("{")
    end = text.rfind("}")
    if start < 0 or end <= start:
        raise ValueError(f"No JSON object found in model response: {text[:160]}")
    data = json.loads(text[start : end + 1])
    if not isinstance(data, dict):
        raise ValueError("Model JSON root must be an object")
    return data
