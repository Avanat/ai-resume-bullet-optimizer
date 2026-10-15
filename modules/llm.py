"""OpenAI-powered bullet rewriting and suggestions, with guardrails.

The model is *instructed* not to invent achievements, and its output is then
*verified* in code: any rewrite that introduces new numbers or new known skills
is flagged so the UI keeps it unselected by default. Prompting + verification is
more reliable than prompting alone.
"""
from __future__ import annotations

import json
import os
import re
import time
from typing import Callable, Dict, List, Optional, Sequence, Tuple

from .bullet_rules import offline_optimize
from .skills import extract_skills

try:
    import openai
    from openai import OpenAI
except ImportError:  # pragma: no cover
    openai = None  # type: ignore
    OpenAI = None  # type: ignore

DEFAULT_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
BATCH_SIZE = 8
MAX_JD_CHARS = 3500


class LLMError(Exception):
    """Any failure talking to the model, with a message safe to show to the user."""


STYLE_GUIDES = {
    "Balanced": "Professional and clear. Mix of action and outcome.",
    "Technical": "Emphasise tools, technologies, systems and engineering detail that is already in the bullet.",
    "Leadership": "Emphasise ownership, collaboration, mentoring and cross-team influence that is already implied.",
    "Concise": "As short as possible while keeping every fact. Aim for 12-18 words.",
}

BULLET_SYSTEM = """You are an expert resume writer and ATS specialist. Rewrite resume bullet points.

STRICT RULES - violating any of them makes the output useless:
1. NEVER invent facts: no new numbers, percentages, tools, technologies, employers, team sizes,
   scope, or outcomes that are not in the original bullet.
2. Preserve every number and metric from the original exactly.
3. Start with a strong past-tense action verb. One sentence, max 28 words, no first-person pronouns,
   no filler such as "responsible for" or "worked on".
4. If the original has no measurable impact, do NOT make one up. Keep the wording factual and put a short
   question in "metric_prompt" asking the user for a number they could add (otherwise null).
5. If a job description is supplied, you may mirror its terminology ONLY where it truthfully describes
   what the original bullet already says.
6. "changes" is a short list of what you improved (e.g. "Replaced weak opener", "Tightened wording").

Return ONLY JSON: {"results":[{"id":<int>,"optimized":"...","changes":["..."],"metric_prompt":null|"..."}]}"""

SUGGEST_SYSTEM = """You are a career coach reviewing a resume analysis. Using ONLY the JSON data provided
(do not invent facts about the candidate), write 5-8 specific, actionable improvement suggestions, most
important first. Never suggest claiming skills or achievements the candidate doesn't have; for missing skills,
suggest adding them only if the candidate genuinely has them, or learning them via a small project.
Return ONLY JSON: {"suggestions":[{"priority":"High|Medium|Low","area":"short label","action":"one or two sentences"}]}"""


# --------------------------------------------------------------------------- #
# Guardrails
# --------------------------------------------------------------------------- #
_NUM_RE = re.compile(r"\d[\d,]*\.?\d*")


def _numbers(text: str) -> set:
    return {m.replace(",", "").rstrip(".") for m in _NUM_RE.findall(text)}


def validate_rewrite(original: str, optimized: str) -> List[str]:
    """Return a list of warnings if the rewrite may contain invented or lost facts."""
    warnings: List[str] = []
    new_nums = _numbers(optimized) - _numbers(original)
    if new_nums:
        warnings.append(f"Introduces numbers not in the original: {', '.join(sorted(new_nums))}")
    lost = _numbers(original) - _numbers(optimized)
    if lost:
        warnings.append(f"Dropped numbers from the original: {', '.join(sorted(lost))}")
    new_skills = set(extract_skills(optimized)) - set(extract_skills(original))
    if new_skills:
        warnings.append(f"Mentions skills not in the original: {', '.join(sorted(new_skills))}")
    if not optimized.strip():
        warnings.append("Empty rewrite")
    return warnings


# --------------------------------------------------------------------------- #
# API plumbing
# --------------------------------------------------------------------------- #
def _get_client(api_key: Optional[str]):
    if OpenAI is None:
        raise LLMError("The 'openai' package is not installed. Run: pip install openai")
    if not api_key:
        raise LLMError("No OpenAI API key provided. Add one in the sidebar or set OPENAI_API_KEY.")
    return OpenAI(api_key=api_key, timeout=60.0)


def _chat_json(client, model: str, system: str, user: str, temperature: float = 0.3, retries: int = 3) -> dict:
    kwargs = dict(
        model=model,
        messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
        response_format={"type": "json_object"},
        temperature=temperature,
    )
    last: Optional[Exception] = None
    for attempt in range(retries):
        try:
            resp = client.chat.completions.create(**kwargs)
            return json.loads(resp.choices[0].message.content or "{}")
        except openai.AuthenticationError as exc:
            raise LLMError("OpenAI rejected the API key. Please check it and try again.") from exc
        except openai.RateLimitError as exc:
            last = exc
            if "quota" in str(exc).lower():
                raise LLMError("Your OpenAI account has no remaining quota/credits.") from exc
            time.sleep(2 ** attempt)
        except openai.APIConnectionError as exc:  # includes timeouts
            last = exc
            time.sleep(2 ** attempt)
        except openai.BadRequestError as exc:
            if "temperature" in str(exc).lower() and "temperature" in kwargs:
                kwargs.pop("temperature")        # some models only accept the default
                continue
            raise LLMError(f"OpenAI rejected the request: {exc}") from exc
        except openai.NotFoundError as exc:
            raise LLMError(f"Model '{model}' was not found or you don't have access to it.") from exc
        except openai.APIStatusError as exc:
            last = exc
            if getattr(exc, "status_code", 500) < 500:
                raise LLMError(f"OpenAI API error ({exc.status_code}): {exc}") from exc
            time.sleep(2 ** attempt)
        except json.JSONDecodeError as exc:
            last = exc
    raise LLMError(f"Could not get a valid response from OpenAI after {retries} attempts: {last}")


# --------------------------------------------------------------------------- #
# Public functions
# --------------------------------------------------------------------------- #
def optimize_bullets(
    bullets: Sequence[Tuple[int, str, str]],
    api_key: Optional[str],
    model: str = DEFAULT_MODEL,
    style: str = "Balanced",
    job_description: str = "",
    progress: Optional[Callable[[int, int], None]] = None,
) -> List[dict]:
    """bullets: (id, text, role_context). Returns one result dict per bullet."""
    client = _get_client(api_key)
    results: Dict[int, dict] = {}
    total = len(bullets)
    jd = (job_description or "").strip()[:MAX_JD_CHARS]

    for start in range(0, total, BATCH_SIZE):
        batch = list(bullets[start:start + BATCH_SIZE])
        payload = {
            "style": f"{style}: {STYLE_GUIDES.get(style, STYLE_GUIDES['Balanced'])}",
            "job_description": jd or None,
            "bullets": [{"id": i, "role_context": ctx, "text": t} for i, t, ctx in batch],
        }
        data = _chat_json(client, model, BULLET_SYSTEM, json.dumps(payload))
        for item in data.get("results", []):
            try:
                results[int(item["id"])] = item
            except (KeyError, TypeError, ValueError):
                continue
        if progress:
            progress(min(start + BATCH_SIZE, total), total)

    final: List[dict] = []
    fallback = {r["id"]: r for r in offline_optimize([(i, t) for i, t, _ in bullets])}
    for bid, text, _ in bullets:
        item = results.get(bid)
        optimized = (item or {}).get("optimized", "").strip() if item else ""
        if not optimized:                      # model skipped it -> safe local fallback
            fb = fallback[bid]
            fb["warnings"] = ["AI returned nothing for this bullet; used local rules instead."]
            final.append(fb)
            continue
        final.append({
            "id": bid, "original": text, "optimized": optimized,
            "changes": [str(c) for c in item.get("changes", [])][:5],
            "metric_prompt": item.get("metric_prompt") or None,
            "warnings": validate_rewrite(text, optimized), "source": "ai",
        })
    return final


def generate_suggestions(context: dict, api_key: Optional[str], model: str = DEFAULT_MODEL) -> List[dict]:
    client = _get_client(api_key)
    data = _chat_json(client, model, SUGGEST_SYSTEM, json.dumps(context, default=str), temperature=0.4)
    out = []
    for s in data.get("suggestions", []):
        if isinstance(s, dict) and s.get("action"):
            pr = s.get("priority", "Medium")
            out.append({"priority": pr if pr in ("High", "Medium", "Low") else "Medium",
                        "area": str(s.get("area", "General")), "action": str(s["action"])})
    if not out:
        raise LLMError("The model returned no suggestions. Please try again.")
    return out
