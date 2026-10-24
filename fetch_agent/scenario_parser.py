"""
Parses natural-language disaster scenario descriptions into simulation parameters.

Primary: ASI:One LLM API (OpenAI-compatible).
Fallback: regex extraction → sensible defaults.
"""

import json
import os
import re
from typing import Dict, Any, List, Optional


# ── ASI:One LLM parsing ─────────────────────────────────────────────────

_ASI_BASE_URL = "https://api.asi1.ai/v1/chat/completions"
_ASI_MODEL = "asi1-mini"

_SYSTEM_PROMPT = (
    "You extract disaster simulation parameters from a user message.\n"
    "Return ONLY a JSON object with these keys (all optional, use defaults when absent):\n"
    '  "num_agents": int (1-6, default 3),\n'
    '  "num_victims": int (1-15, default 6),\n'
    '  "modes": list of strings from ["greedy","coordinated","llm"] (default all three),\n'
    '  "steps": int (5-300, default 40),\n'
    '  "seed": int (default 42)\n'
    "No markdown fences, no commentary — only the JSON object."
)


def _call_asi_one(user_text: str) -> Optional[Dict[str, Any]]:
    """Call ASI:One chat completion; returns parsed dict or None on failure."""
    api_key = os.getenv("ASI1_API_KEY", "")
    if not api_key:
        return None

    try:
        import httpx

        resp = httpx.post(
            _ASI_BASE_URL,
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": _ASI_MODEL,
                "messages": [
                    {"role": "system", "content": _SYSTEM_PROMPT},
                    {"role": "user", "content": user_text},
                ],
                "temperature": 0.0,
                "max_tokens": 200,
            },
            timeout=10.0,
        )
        resp.raise_for_status()
        content = resp.json()["choices"][0]["message"]["content"]
        # Strip possible fences
        content = content.strip()
        m = re.search(r"\{.*\}", content, re.DOTALL)
        if m:
            return json.loads(m.group())
    except Exception:
        pass
    return None


# ── regex / default fallback ─────────────────────────────────────────────

_NUM_RE = re.compile(r"(\d+)\s*(?:rescue|agent|team|unit)", re.I)
_VIC_RE = re.compile(r"(\d+)\s*victim", re.I)
_MODE_RE = re.compile(r"\b(greedy|coordinated|hungarian|llm)\b", re.I)
_SEED_RE = re.compile(r"seed\s*[=:]?\s*(\d+)", re.I)
_STEP_RE = re.compile(r"(\d+)\s*step", re.I)


def _regex_fallback(user_text: str) -> Dict[str, Any]:
    """Best-effort extraction using simple patterns."""
    params: Dict[str, Any] = {}

    m = _NUM_RE.search(user_text)
    if m:
        params["num_agents"] = int(m.group(1))

    m = _VIC_RE.search(user_text)
    if m:
        params["num_victims"] = int(m.group(1))

    modes_found = _MODE_RE.findall(user_text)
    if modes_found:
        normalised = []
        for raw in modes_found:
            low = raw.lower()
            normalised.append("coordinated" if low == "hungarian" else low)
        params["modes"] = list(dict.fromkeys(normalised))  # dedup, preserve order

    m = _SEED_RE.search(user_text)
    if m:
        params["seed"] = int(m.group(1))

    m = _STEP_RE.search(user_text)
    if m:
        params["steps"] = int(m.group(1))

    return params


# ── public interface ─────────────────────────────────────────────────────

_DEFAULTS: Dict[str, Any] = {
    "num_agents": 3,
    "num_victims": 6,
    "modes": ["greedy", "coordinated", "llm"],
    "steps": 40,
    "seed": 42,
}


def parse_scenario(user_text: str) -> Dict[str, Any]:
    """
    Parse a natural-language scenario description into simulation parameters.

    Tries ASI:One LLM first, falls back to regex, then fills defaults.
    """
    parsed = _call_asi_one(user_text)

    if parsed is None:
        parsed = _regex_fallback(user_text)

    # Merge with defaults
    result = dict(_DEFAULTS)
    for key in _DEFAULTS:
        if key in parsed and parsed[key] is not None:
            result[key] = parsed[key]

    return result
