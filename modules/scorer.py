"""Resume scoring (relevance, clarity, skills, ATS-friendliness) and suggestions."""
from __future__ import annotations

from dataclasses import dataclass, field
from statistics import mean
from typing import Dict, List, Optional

from .bullet_rules import analyze_bullet
from .nlp_matcher import MatchResult
from .resume_parser import YEAR_RE, Bullet, ParsedResume

WEIGHTS_WITH_JD = {"relevance": 0.30, "clarity": 0.25, "skills": 0.20, "ats": 0.25}
WEIGHTS_NO_JD = {"clarity": 0.35, "skills": 0.30, "ats": 0.35}


@dataclass
class ResumeScore:
    overall: float
    relevance: Optional[float]
    clarity: float
    skills: float
    ats: float
    ats_checks: List[dict] = field(default_factory=list)
    bullet_scores: List[dict] = field(default_factory=list)
    weights: Dict[str, float] = field(default_factory=dict)

    def components(self) -> Dict[str, float]:
        d = {"Clarity": self.clarity, "Skills": self.skills, "ATS": self.ats}
        if self.relevance is not None:
            d = {"Relevance": self.relevance, **d}
        return d


def _ats_checks(p: ParsedResume) -> List[dict]:
    non_ascii = sum(1 for c in p.raw_text if ord(c) > 127 and c != "•")
    ratio = non_ascii / max(1, len(p.raw_text))
    n_bullets = len(p.bullets)
    checks = [
        ("Email address found", bool(p.email), 10, "Add a professional email in the header."),
        ("Phone number found", bool(p.phone), 5, "Add a phone number in the header."),
        ("Standard 'Experience' heading", "experience" in p.sections, 15,
         "Use a standard heading such as 'Experience' or 'Work Experience'."),
        ("Standard 'Education' heading", "education" in p.sections, 10,
         "Add a clearly labelled 'Education' section."),
        ("Standard 'Skills' heading", "skills" in p.sections, 15,
         "Add a 'Skills' section so ATS can read your keywords."),
        ("Reasonable length (250-1000 words)", 250 <= p.word_count <= 1000, 10,
         "Aim for 1-2 pages: 250-1000 words."),
        ("Two pages or fewer", p.num_pages <= 2, 5, "Trim to two pages or fewer."),
        ("Dates present for roles/education", len(YEAR_RE.findall(p.raw_text)) >= 2, 10,
         "Include start/end dates (e.g. 'Jun 2022 - Present')."),
        ("Achievements written as bullets", n_bullets >= 3, 10,
         "Use bullet points to describe achievements."),
        ("Few unusual characters", ratio < 0.02, 5,
         "Avoid icons, emojis and decorative symbols; ATS parsers can choke on them."),
        ("LinkedIn / GitHub link", bool(p.links), 5, "Add a LinkedIn or GitHub URL."),
    ]
    return [{"label": l, "passed": ok, "points": pts if ok else 0, "max_points": pts, "tip": tip}
            for l, ok, pts, tip in checks]


def score_resume(p: ParsedResume, bullets: List[Bullet], match: Optional[MatchResult] = None) -> ResumeScore:
    analyses = [(b, analyze_bullet(b.text)) for b in bullets]
    clarity = mean(a.score for _, a in analyses) if analyses else 0.0

    n_skills = len(p.skills)
    skills = min(1, n_skills / 15) * 70 + min(1, len(p.skill_categories) / 4) * 20 \
        + (10 if "skills" in p.sections else 0)

    checks = _ats_checks(p)
    ats = sum(c["points"] for c in checks)

    relevance = match.overall if match else None
    weights = WEIGHTS_WITH_JD if relevance is not None else WEIGHTS_NO_JD
    parts = {"relevance": relevance, "clarity": clarity, "skills": skills, "ats": ats}
    overall = sum(parts[k] * w for k, w in weights.items())
    return ResumeScore(
        overall=float(overall), relevance=relevance, clarity=float(clarity), skills=float(skills),
        ats=float(ats), ats_checks=checks, weights=weights,
        bullet_scores=[{"id": b.id, "text": b.text, "score": a.score, "issues": a.issues} for b, a in analyses],
    )


def rule_based_suggestions(p: ParsedResume, score: ResumeScore, gap: Optional[dict] = None,
                           match: Optional[MatchResult] = None, recs: Optional[List[dict]] = None) -> List[dict]:
    out: List[dict] = []
    add = lambda pr, area, act: out.append({"priority": pr, "area": area, "action": act})  # noqa: E731

    bs = score.bullet_scores
    if not bs:
        add("High", "Content", "No bullet points were detected. Describe each role with 3-5 bullet points "
                                "that start with an action verb and show an outcome.")
    else:
        n = len(bs)
        no_verb = sum(1 for b in bs if any("action verb" in i for i in b["issues"]))
        no_metric = sum(1 for b in bs if any("measurable" in i for i in b["issues"]))
        long_ = sum(1 for b in bs if any("long" in i.lower() for i in b["issues"]))
        short = sum(1 for b in bs if any("short" in i.lower() for i in b["issues"]))
        weak = sum(1 for b in bs if any("Weak opening" in i for i in b["issues"]))
        if no_verb / n > 0.2:
            add("High", "Clarity", f"{no_verb} of {n} bullets don't start with a strong action verb. "
                                   "Lead with verbs like Built, Led, Reduced, Automated.")
        if no_metric / n > 0.4:
            add("High", "Impact", f"{no_metric} of {n} bullets have no measurable result. Add real numbers you can "
                                  "defend: % improvement, hours saved, users, revenue, dataset size.")
        if weak:
            add("Medium", "Clarity", f"{weak} bullets open with 'responsible for' / 'worked on' style phrases. "
                                     "Replace them with the specific action you took.")
        if long_:
            add("Medium", "Clarity", f"{long_} bullets are too long. Keep each to one line or two (under ~28 words).")
        if short:
            add("Low", "Clarity", f"{short} bullets are very short. Add the 'how' and the outcome.")

    for c in score.ats_checks:
        if not c["passed"]:
            add("High" if c["max_points"] >= 10 else "Medium", "ATS", c["tip"])

    if len(p.skills) < 8:
        add("Medium", "Skills", "Only a few recognisable skills were found. List your tools and technologies "
                                "explicitly in a Skills section (only ones you can discuss in an interview).")

    if gap and gap.get("missing"):
        top = [r for r in (recs or []) if r["priority"] in ("High", "Medium")][:4] or \
              [{"skill": s} for s in gap["missing"][:4]]
        names = ", ".join(r["skill"] for r in top)
        add("High", "Job fit", f"The job asks for {names}, which your resume doesn't mention. If you have real "
                               "experience, add it to your bullets/skills; if not, consider a small project.")
    if match:
        missing_kw = [k["term"] for k in match.keywords if not k["in_resume"]][:6]
        if match.overall < 60 and missing_kw:
            add("High", "Job fit", "Mirror the posting's language where it is truthful. Terms from the job that "
                                   f"don't appear in your resume: {', '.join(missing_kw)}.")
        weak_rel = [b for b in match.bullet_relevance if b["relevance"] < 15]
        if weak_rel and len(weak_rel) >= max(2, len(match.bullet_relevance) // 3):
            add("Low", "Job fit", f"{len(weak_rel)} bullets are barely related to this job. Reorder so the most "
                                  "relevant achievements come first, or trim the rest.")

    if not out:
        add("Low", "Overall", "Your resume is in good shape. Proofread, and tailor the top bullets for each application.")
    order = {"High": 0, "Medium": 1, "Low": 2}
    return sorted(out, key=lambda s: order[s["priority"]])
