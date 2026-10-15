# 📄 AI Resume Bullet Optimizer

A Streamlit app that parses a PDF resume, rewrites its bullet points with the OpenAI API **without inventing
achievements**, matches it against a job description using TF-IDF or sentence embeddings, finds skill gaps, scores
the resume, and exports the improved content.

It works end-to-end **even without an API key** (local rule-based rewriting), so you can demo it anywhere.

## Features

| # | Feature | How it works |
|---|---------|--------------|
| 1 | **Resume upload** | `pypdf` text extraction → heading detection → bullet/entry grouping; extracts contact info, skills, education, experience, projects |
| 2 | **Bullet optimizer** | OpenAI rewrite with strict prompt **+ programmatic fact-check** (flags invented numbers/skills). Offline rule-based mode included |
| 3 | **Job matching** | TF-IDF (uni+bigrams) cosine similarity *or* MiniLM sentence embeddings, blended with skill coverage; per-bullet relevance |
| 4 | **Skills gap** | Taxonomy of ~200 skills + aliases, "required vs nice-to-have" detection, prioritised learning recommendations |
| 5 | **Resume scoring** | Relevance, Clarity, Skills, ATS-friendliness (transparent rubric below) |
| 6 | **Suggestions** | Rule-based (always on) + optional AI coaching grounded in your computed data |
| 7 | **Dashboard** | Plotly gauges, radar, skill-gap bars, keyword weights, bullet relevance, before/after quality |
| 8 | **Download** | Word (.docx), Markdown, text, or a JSON analysis report; only rewrites you accept are included |

## Quick start

```bash
# 1. Create an environment (Python 3.10+)
python -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\activate

# 2. Install
pip install -r requirements.txt

# 3. (Optional) add your OpenAI key
cp .env.example .env                 # then edit .env   (or paste the key in the app sidebar)

# 4. Run
streamlit run app.py
```

Click **Use sample resume** on the first page and **Load sample job description** on the Job Match page to try
everything in under a minute.

**Optional semantic matching:** `pip install sentence-transformers` (downloads ~90 MB model on first use).
If it's not installed, the app falls back to TF-IDF automatically.

**Deploying:** push to GitHub and deploy on Streamlit Community Cloud; add `OPENAI_API_KEY` under *Secrets*.

## How it works

```
PDF ──► resume_parser ──► ParsedResume ─┬─► skills.py ─────────► skill gap + learning plan
        (pypdf, regex)    (sections,    ├─► nlp_matcher.py ────► match score, keywords, bullet relevance
                           bullets,     ├─► llm.py / bullet_rules ► rewrites (+ fact-check guardrails)
                           skills)      ├─► scorer.py ─────────► 4-part score + suggestions
                                        └─► export.py ─────────► docx / md / txt / json
                      app.py (Streamlit UI) + charts.py (Plotly)
```

### The NLP/ML parts
- **TF-IDF + cosine similarity** (`scikit-learn`) with a custom tokenizer that preserves `c++`, `node.js`, `ci/cd`.
- **Keyword extraction:** the job description is split into sentences that act as "documents", so IDF is meaningful
  *within* the posting; unigrams covered by a chosen bigram are dropped ("machine learning" beats "machine").
- **Sentence embeddings (optional):** for each JD sentence, take its best-matching resume sentence (cosine) and average.
- **Skill extraction:** one compiled regex over a taxonomy with aliases (`k8s`→Kubernetes), longest-match-first,
  case-sensitivity for ambiguous words (`Swift`, `Excel`, `Go`).
- **Score calibration:** raw cosine between two documents is small by nature, so it is scaled (TF-IDF: 0.35 ⇒ 100%;
  embeddings: 0.20–0.65 ⇒ 0–100%). Overall match = 50% text similarity + 50% skill coverage. These constants live at
  the top of `nlp_matcher.py` and are easy to tune.

### Avoiding made-up achievements
1. The system prompt forbids new facts and requires numbers to be preserved.
2. If a bullet has no metric, the model must **ask** (`metric_prompt`) rather than invent one.
3. `llm.validate_rewrite()` checks every rewrite for **new numbers, dropped numbers, and new known skills**.
   Flagged rewrites show a warning and are **unticked by default**.
4. Every rewrite is editable; nothing is exported unless the user ticks **Use this version**.

### Scoring rubric
- **Clarity** - mean bullet score: action verb (35) + measurable impact (30) + length 10-28 words (15) + no weak
  phrases/first person/passive (20).
- **Skills** - number of distinct skills (up to 15), category variety, presence of a Skills section.
- **ATS** - 11 checks: contact info, standard headings, length, dates, bullets, unusual characters, links.
- **Relevance** - the job match score (only when a job description is provided; weights are re-balanced otherwise).

## Project structure

```
app.py                  Streamlit UI (7 pages, session-state driven)
modules/
  resume_parser.py      PDF → text → structured resume
  skills.py             Taxonomy, extraction, JD analysis, gap, learning recs
  nlp_matcher.py        TF-IDF / embeddings, keywords, bullet relevance
  bullet_rules.py       Bullet analyzer + offline rewriter
  llm.py                OpenAI calls, retries, guardrails
  scorer.py             Resume scoring + rule-based suggestions
  charts.py             Plotly figures
  export.py             docx / md / txt / json builders
  samples.py            Fictional demo resume + job description
tests/test_core.py      Unit tests + headless UI smoke test
```

## Tests

```bash
pytest -q        # no API key or network needed
```

Covers skill extraction edge cases, PDF round-trip, parser structure, scoring, matching, the hallucination guard,
exports, and a headless run of the whole Streamlit app.

## Limitations (be upfront about these in interviews)
- Resume parsing is heuristic; multi-column or heavily designed PDFs may parse imperfectly (hence the editable
  extracted text and "extra bullets" box). Scanned/image PDFs need OCR, which is not included.
- The skill taxonomy is finite; skills outside it aren't scored (they're shown as "Other listed").
- The hallucination guard catches new numbers and known skills, not every possible fabricated claim: users must
  review rewrites.
- ATS checks are best-practice heuristics, not a simulation of any specific ATS.

## Ideas to extend
OCR for scanned PDFs · spaCy NER for companies/titles · cover-letter generator · multi-JD comparison ·
persistent user accounts · A/B evaluation of rewrite quality with an LLM judge.
