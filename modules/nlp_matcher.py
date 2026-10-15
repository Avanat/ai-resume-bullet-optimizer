"""Resume <-> job description matching with real NLP.

Two interchangeable text-similarity back-ends:
  * TF-IDF (unigrams + bigrams) with cosine similarity  - fast, no downloads
  * Sentence embeddings (all-MiniLM-L6-v2)              - semantic, optional install

The final match score blends text similarity with *skill coverage* (from the skills
taxonomy), because raw cosine scores between two documents are naturally small.
"""
from __future__ import annotations

import importlib.util
import re
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Generic job-post filler that carries no signal for matching.
JD_NOISE = {
    "experience", "ability", "work", "working", "team", "teams", "strong", "years", "year", "skills",
    "skill", "responsibilities", "requirements", "including", "etc", "will", "looking", "candidate",
    "role", "job", "company", "position", "preferred", "required", "must", "plus", "knowledge",
    "understanding", "good", "excellent", "join", "opportunity", "seeking", "ideal", "environment",
    "based", "using", "use", "new", "like", "help", "make", "able", "well", "across", "within",
    "qualifications", "nice", "have", "bonus", "familiarity", "hands", "hands-on", "proven",
}
_STOP = set(ENGLISH_STOP_WORDS) | JD_NOISE
_TOKEN_RE = re.compile(r"[a-z][a-z0-9+#]*(?:[.\-/][a-z0-9+#]+)*")

# Calibration constants (documented in README): map raw similarity to a 0-100 scale.
TFIDF_FULL_MATCH = 0.35        # cosine at/above this counts as a 100% text match
EMB_FLOOR, EMB_SPAN = 0.20, 0.45


def tokenize(text: str) -> List[str]:
    out = []
    for t in _TOKEN_RE.findall((text or "").lower()):
        if t in _STOP or (len(t) < 2 and t not in {"r", "c"}):
            continue
        out.append(t)
    return out


def _vectorizer(ngram=(1, 2)) -> TfidfVectorizer:
    return TfidfVectorizer(tokenizer=tokenize, lowercase=False, token_pattern=None,
                           ngram_range=ngram, sublinear_tf=True)


def tfidf_cosine(a: str, b: str) -> float:
    try:
        X = _vectorizer().fit_transform([a, b])
    except ValueError:  # empty vocabulary
        return 0.0
    return float(cosine_similarity(X[0], X[1])[0, 0])


def _sentences(text: str) -> List[str]:
    parts = re.split(r"[\n\r]+|(?<=[.!?;])\s+", text or "")
    cleaned = [p.strip(" -•*\t") for p in parts]
    return [p for p in cleaned if len(p.split()) >= 2]


def extract_keywords(jd: str, resume: str, top_n: int = 30) -> List[dict]:
    """Rank JD terms by TF-IDF *within the JD* (sentences as documents), flag those in the resume."""
    sents = _sentences(jd) or [jd]
    vec = _vectorizer()
    try:
        X = vec.fit_transform(sents)
    except ValueError:
        return []
    weights = np.asarray(X.sum(axis=0)).ravel()
    terms = vec.get_feature_names_out()
    resume_grams = set(vec.build_analyzer()(resume))
    order = np.argsort(-weights)
    top = weights[order[0]] if len(order) else 1.0
    chosen: List[dict] = []
    for idx in order:
        term = terms[idx]
        if term.replace(".", "").isdigit() or len(term) < 2:
            continue
        # prefer the bigram over its own unigrams ("machine learning" over "machine")
        if " " not in term and any(term in c["term"].split() for c in chosen):
            continue
        chosen.append({"term": term, "weight": float(weights[idx] / top), "in_resume": term in resume_grams})
        if len(chosen) >= top_n:
            break
    return chosen


def bullet_relevance(bullets: Sequence[Tuple[int, str]], jd: str) -> List[dict]:
    """TF-IDF cosine of every bullet against the JD, scaled to 0-100."""
    if not bullets or not jd.strip():
        return []
    texts = [jd] + [t for _, t in bullets]
    try:
        X = _vectorizer().fit_transform(texts)
    except ValueError:
        return []
    sims = cosine_similarity(X[0], X[1:]).ravel()
    return [{"id": bid, "text": t, "relevance": float(min(1.0, s / 0.30) * 100)}
            for (bid, t), s in zip(bullets, sims)]


# ----------------------------- embeddings (optional) ----------------------- #
def embeddings_available() -> bool:
    return importlib.util.find_spec("sentence_transformers") is not None


@lru_cache(maxsize=1)
def _load_model(name: str = "all-MiniLM-L6-v2"):
    from sentence_transformers import SentenceTransformer  # heavy import, lazy
    return SentenceTransformer(name)


def embedding_coverage(resume: str, jd: str) -> float:
    """Mean over JD sentences of the best-matching resume sentence (cosine)."""
    model = _load_model()
    r = _sentences(resume)[:150]
    j = _sentences(jd)[:100]
    if not r or not j:
        return 0.0
    R = model.encode(r, normalize_embeddings=True)
    J = model.encode(j, normalize_embeddings=True)
    return float((J @ R.T).max(axis=1).mean())


# ----------------------------- orchestration -------------------------------- #
@dataclass
class MatchResult:
    method: str
    raw_similarity: float
    text_score: float
    skill_coverage: Optional[float]
    overall: float
    keywords: List[dict] = field(default_factory=list)
    bullet_relevance: List[dict] = field(default_factory=list)
    note: str = ""


def match_resume(
    resume_text: str,
    jd_text: str,
    resume_skills: Sequence[str],
    jd_skills: Sequence[str],
    bullets: Sequence[Tuple[int, str]],
    method: str = "tfidf",
) -> MatchResult:
    if len(jd_text.split()) < 15:
        raise ValueError("The job description is too short to analyze - paste the full posting.")

    note = ""
    used = "TF-IDF cosine"
    if method == "embeddings":
        try:
            raw = embedding_coverage(resume_text, jd_text)
            text_score = float(np.clip((raw - EMB_FLOOR) / EMB_SPAN, 0, 1) * 100)
            used = "Sentence embeddings (MiniLM)"
        except Exception as exc:  # noqa: BLE001 - missing package / no model download
            note = f"Embeddings unavailable ({type(exc).__name__}); fell back to TF-IDF."
            method = "tfidf"
    if method != "embeddings":
        raw = tfidf_cosine(resume_text, jd_text)
        text_score = float(min(1.0, raw / TFIDF_FULL_MATCH) * 100)

    jd_set, resume_set = set(jd_skills), set(resume_skills)
    coverage = (len(jd_set & resume_set) / len(jd_set) * 100) if jd_set else None
    overall = text_score if coverage is None else 0.5 * text_score + 0.5 * coverage
    return MatchResult(
        method=used, raw_similarity=raw, text_score=text_score, skill_coverage=coverage,
        overall=float(overall), keywords=extract_keywords(jd_text, resume_text),
        bullet_relevance=bullet_relevance(bullets, jd_text), note=note,
    )
