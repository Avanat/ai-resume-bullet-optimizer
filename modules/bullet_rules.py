"""Rule-based bullet analysis (drives scoring) and an offline rewriter (no API key needed).

The analyzer is the "ground truth" quality check used both before and after the LLM
rewrite, so improvements are measured with the same yardstick.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, List, Tuple

ACTION_VERBS = set("""
accelerated achieved administered advised analyzed annotated applied architected assembled audited
authored automated benchmarked boosted built calculated championed cleaned coached collaborated
communicated compiled completed conceived conducted configured consolidated constructed consulted
containerized contributed converted coordinated created curated cut debugged decreased defined
delivered deployed derived designed detected developed devised diagnosed directed documented doubled
drafted drove earned edited eliminated enabled engineered enhanced established estimated evaluated
examined executed expanded expedited extracted facilitated forecasted formulated founded generated
grew guided identified implemented improved increased initiated innovated instrumented integrated
introduced investigated labeled launched led leveraged maintained managed mapped measured mentored
migrated minimized modeled modernized monitored negotiated normalized onboarded operated optimized
orchestrated organized overhauled oversaw owned partnered performed piloted pioneered planned
presented prioritized processed produced programmed prototyped provided published ran rebuilt
redesigned reduced refactored replaced reported researched resolved restructured retrained reviewed
revised saved scaled scheduled secured shipped simplified solved sourced spearheaded standardized
streamlined strengthened supervised supported surpassed synthesized taught tested tracked trained
transformed translated troubleshot tuned unified upgraded validated visualized won wrote
""".split())

GERUND_TO_PAST: Dict[str, str] = {
    "managing": "Managed", "developing": "Developed", "leading": "Led", "building": "Built",
    "creating": "Created", "designing": "Designed", "implementing": "Implemented",
    "maintaining": "Maintained", "analyzing": "Analyzed", "coordinating": "Coordinated",
    "supporting": "Supported", "handling": "Handled", "overseeing": "Oversaw",
    "delivering": "Delivered", "writing": "Wrote", "testing": "Tested", "monitoring": "Monitored",
    "training": "Trained", "optimizing": "Optimized", "improving": "Improved",
    "organizing": "Organized", "preparing": "Prepared", "processing": "Processed",
    "reviewing": "Reviewed", "conducting": "Conducted", "planning": "Planned", "running": "Ran",
    "driving": "Drove", "launching": "Launched", "automating": "Automated",
    "deploying": "Deployed", "integrating": "Integrated", "researching": "Researched",
    "documenting": "Documented", "ensuring": "Ensured", "providing": "Provided",
    "performing": "Performed", "executing": "Executed", "administering": "Administered",
    "resolving": "Resolved", "troubleshooting": "Troubleshot", "migrating": "Migrated",
    "refactoring": "Refactored", "debugging": "Debugged", "collaborating": "Collaborated",
    "mentoring": "Mentored", "cleaning": "Cleaned", "analysing": "Analysed", "gathering": "Gathered",
}

_WEAK_GERUND = re.compile(
    r"^(?:i\s+)?(?:was\s+|were\s+)?(?:responsible for|in charge of|tasked with|charged with|duties included)\s+(\w+ing)\b",
    re.I,
)
_WEAK_SIMPLE = [
    (re.compile(r"^(?:i\s+)?worked on\s+", re.I), "Contributed to "),
    (re.compile(r"^(?:i\s+)?(?:helped with|assisted with|assisted in|helped in)\s+", re.I), "Supported "),
    (re.compile(r"^(?:i\s+)?(?:was\s+)?involved in\s+", re.I), "Contributed to "),
]
_WEAK_DETECT = re.compile(
    r"^(?:i\s+)?(?:was\s+|were\s+)?(responsible for|in charge of|tasked with|worked on|helped|assisted|involved in|duties included|participated in)\b",
    re.I,
)
_FIRST_PERSON = re.compile(r"\b(i|my|me|myself)\b", re.I)
_PASSIVE = re.compile(r"\b(was|were|been|being)\s+\w+ed\b", re.I)
_YEAR = re.compile(r"\b(?:19|20)\d{2}\b")


@dataclass
class BulletAnalysis:
    text: str
    score: float
    has_action_verb: bool
    has_metric: bool
    word_count: int
    issues: List[str] = field(default_factory=list)


def starts_with_action_verb(text: str) -> bool:
    m = re.match(r"[A-Za-z]+", text.strip())
    if not m:
        return False
    w = m.group(0).lower()
    return w in ACTION_VERBS or (w + "d") in ACTION_VERBS or (w + "ed") in ACTION_VERBS


def has_metric(text: str) -> bool:
    stripped = _YEAR.sub("", text)
    return bool(re.search(r"\d", stripped)) or bool(re.search(r"\b(doubled|tripled|halved|quadrupled)\b", text, re.I))


def analyze_bullet(text: str) -> BulletAnalysis:
    text = text.strip()
    words = len(text.split())
    issues: List[str] = []
    score = 0.0

    verb = starts_with_action_verb(text)
    if verb:
        score += 35
    else:
        issues.append("Doesn't start with a strong action verb")

    metric = has_metric(text)
    if metric:
        score += 30
    else:
        issues.append("No measurable impact (numbers, %, time, scale)")

    if 10 <= words <= 28:
        score += 15
    elif 7 <= words < 10 or 28 < words <= 35:
        score += 8
        issues.append("Slightly too short" if words < 10 else "A bit long - aim for under 28 words")
    else:
        issues.append("Too short to show impact" if words < 7 else "Too long - split or tighten")

    penalty = 0
    if _WEAK_DETECT.match(text):
        penalty += 10
        issues.append("Weak opening phrase (e.g. 'responsible for', 'worked on')")
    if _FIRST_PERSON.search(text):
        penalty += 5
        issues.append("Uses first-person pronouns")
    if _PASSIVE.search(text):
        penalty += 5
        issues.append("Passive voice")
    score += max(0, 20 - penalty)
    return BulletAnalysis(text, score, verb, metric, words, issues)


def rewrite_rule_based(text: str) -> Tuple[str, List[str]]:
    """Conservative rewrite: fixes weak openers only; never adds facts."""
    original = text.strip()
    new, changes = original, []

    m = _WEAK_GERUND.match(new)
    if m and m.group(1).lower() in GERUND_TO_PAST:
        verb = GERUND_TO_PAST[m.group(1).lower()]
        new = verb + new[m.end():]
        changes.append(f"Replaced weak opener with action verb '{verb}'")
    else:
        for pattern, repl in _WEAK_SIMPLE:
            if pattern.match(new):
                new = pattern.sub(repl, new, count=1)
                changes.append(f"Replaced weak opener with '{repl.strip()}'")
                break

    if re.match(r"^i\s+", new, re.I):
        new = re.sub(r"^i\s+", "", new, flags=re.I)
        changes.append("Removed first-person pronoun")
    if new and new[0].islower():
        new = new[0].upper() + new[1:]
        if not changes:
            changes.append("Capitalized first word")
    new = re.sub(r"\s+", " ", new).strip()
    return new, changes


def offline_optimize(bullets: List[Tuple[int, str]]) -> List[dict]:
    """Same result schema as llm.optimize_bullets, but fully local and deterministic."""
    results = []
    for bid, text in bullets:
        new, changes = rewrite_rule_based(text)
        metric_prompt = None
        if not has_metric(new):
            metric_prompt = "Can you quantify this? e.g. % improvement, time saved, users/records affected, team size."
        if not starts_with_action_verb(new):
            changes.append("Still lacks a strong opening verb - AI mode can suggest one")
        results.append({
            "id": bid, "original": text, "optimized": new, "changes": changes or ["No safe automatic change found"],
            "metric_prompt": metric_prompt, "warnings": [], "source": "rules",
        })
    return results
