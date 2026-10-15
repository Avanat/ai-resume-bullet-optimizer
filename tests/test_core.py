"""Run with:  pytest -q   (no network or API key needed)."""
import io
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from modules import export, llm, nlp_matcher, samples, scorer
from modules import skills as sk
from modules.bullet_rules import analyze_bullet, offline_optimize, rewrite_rule_based
from modules.resume_parser import ResumeParseError, extract_text_from_pdf, parse_resume


@pytest.fixture(scope="module")
def parsed():
    return parse_resume(samples.RESUME, 1)


def make_pdf(text: str) -> bytes:
    from reportlab.lib.pagesizes import A4
    from reportlab.pdfgen import canvas
    buf = io.BytesIO()
    c = canvas.Canvas(buf, pagesize=A4)
    y = 800
    for line in text.splitlines():
        c.drawString(40, y, line.replace("•", "-"))
        y -= 13
        if y < 40:
            c.showPage()
            y = 800
    c.save()
    return buf.getvalue()


# ---- skills ----
def test_skill_extraction_aliases_and_ambiguity():
    s = sk.extract_skills("Built apps in C++ and C#, used React Native, k8s, scikit-learn. Go to market. R programming.")
    assert {"C++", "C#", "React Native", "Kubernetes", "scikit-learn", "R"} <= set(s)
    assert "Go" not in s and "React" not in s       # 'Go to market' is not Go; React Native consumed 'React'
    assert "Excel" not in sk.extract_skills("She will excel at this")


def test_jd_optional_detection():
    info = sk.analyze_job_description(samples.JOB_DESCRIPTION)
    assert info["Python"]["required"] >= 1
    assert info["Terraform"]["required"] == 0 and info["Terraform"]["optional"] == 1


# ---- parser ----
def test_parser_structure(parsed):
    assert parsed.name == "Alex Morgan" and parsed.email == "alex.morgan@example.com"
    assert parsed.phone and len(parsed.experience) == 2 and len(parsed.projects) == 2
    assert len(parsed.bullets) == 10 and "Python" in parsed.skills


def test_pdf_roundtrip():
    text, pages = extract_text_from_pdf(make_pdf(samples.RESUME))
    p = parse_resume(text, pages)
    assert p.email and len(p.bullets) >= 8 and "scikit-learn" in p.skills


def test_bad_pdfs():
    with pytest.raises(ResumeParseError):
        extract_text_from_pdf(b"not a pdf")
    with pytest.raises(ResumeParseError):
        extract_text_from_pdf(b"")


# ---- bullets ----
def test_rule_rewrite_and_scoring():
    new, changes = rewrite_rule_based("Responsible for building weekly dashboards")
    assert new.startswith("Built") and changes
    assert analyze_bullet("Reduced report time by 40% using Python automation scripts").score > \
        analyze_bullet("Responsible for reports").score
    assert offline_optimize([(1, "Worked on cleaning data")])[0]["metric_prompt"]


def test_hallucination_guard():
    assert llm.validate_rewrite("Cut costs", "Cut costs by 30%")                   # invented number
    assert llm.validate_rewrite("Built a model", "Built a PyTorch model")          # invented skill
    assert not llm.validate_rewrite("Cut costs by 30%", "Reduced costs by 30%")    # clean


# ---- matching, scoring, export ----
def test_matching_and_gap(parsed):
    info = sk.analyze_job_description(samples.JOB_DESCRIPTION)
    m = nlp_matcher.match_resume(parsed.raw_text, samples.JOB_DESCRIPTION, list(parsed.skills), list(info),
                                 [(b.id, b.text) for b in parsed.bullets])
    assert 0 < m.overall < 100 and m.keywords and m.bullet_relevance
    gap = sk.skill_gap(parsed.skills, info)
    assert "Kubernetes" in gap["missing"] and "Python" in gap["matched"]
    with pytest.raises(ValueError):
        nlp_matcher.match_resume(parsed.raw_text, "too short", [], [], [])
    sc = scorer.score_resume(parsed, parsed.bullets, m)
    assert 0 <= sc.overall <= 100 and sc.relevance is not None
    assert scorer.rule_based_suggestions(parsed, sc, gap, m)


def test_export(parsed):
    reps = {0: "Built weekly Tableau sales dashboards for regional managers"}
    txt = export.to_text(parsed, reps, [])
    assert "Built weekly Tableau sales dashboards" in txt and "Responsible for building" not in txt
    assert export.to_docx(parsed, reps, [])[:2] == b"PK"
    assert export.to_markdown(parsed, reps, []).startswith("# Alex Morgan")


# ---- UI smoke test (headless) ----
def test_app_end_to_end():
    from streamlit.testing.v1 import AppTest
    at = AppTest.from_file(str(Path(__file__).resolve().parent.parent / "app.py"), default_timeout=60).run()
    assert not at.exception
    at.button[[b.label for b in at.button].index("Use sample resume")].click().run()
    assert not at.exception
    for page in ["2 · Bullet Optimizer", "3 · Job Match", "5 · Score & Suggestions", "6 · Analytics Dashboard", "7 · Download"]:
        at.session_state["page"] = page
        at.run()
        assert not at.exception, page
